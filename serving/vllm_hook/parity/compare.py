"""Compare vLLM-Hook captures against the HF activations the probes were fit on (CPU).

Reads <data>/prompts.jsonl, <data>/hf_layer20.npy (stored, what the paper trained
on) and any extra capture files given as name=path.npy (same row order).

Reports
  1. vector agreement per source vs stored: cosine and relative L2 percentiles
  2. probe impact under the paper's exact protocol (missing_cells.py: pre-action
     steps < 3, case score = max over its rows, stratified 5-fold seed 13).
     Fold models are always fit on the STORED vectors, then held-out cases are
     scored with each source's vectors. stored->stored must reproduce the
     published cells (commitment/logistic 0.853, naive/logistic 0.840).
  3. per-case |delta p| and verdict flips at 0.5 vs stored

Usage: python compare.py <latent-state-auditing repo> <data dir> [name=path.npy ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

repo, data = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / "analysis"))
from missing_cells import ACTION_STEP, FOLDS, SEED, auroc, logreg_fit, logreg_score  # noqa: E402
from latent_agent_auditing.experiments import summarize_paper_evidence as spe  # noqa: E402

rows = [json.loads(l) for l in open(data / "prompts.jsonl")]
sources = {"stored": np.load(data / "hf_layer20.npy").astype(np.float64)}
for arg in sys.argv[3:]:
    name, path = arg.split("=", 1)
    sources[name] = np.load(path).astype(np.float64)
    assert sources[name].shape == sources["stored"].shape, (name, sources[name].shape)

ref = sources["stored"]
report = {"vectors": {}, "probe": {}, "scores": {}}
for name, X in sources.items():
    if name == "stored":
        continue
    cos = (X * ref).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(ref, axis=1))
    rel = np.linalg.norm(X - ref, axis=1) / np.linalg.norm(ref, axis=1)
    def pct(v):
        return {"min": float(v.min()), "p1": float(np.percentile(v, 1)), "median": float(np.median(v)),
                "p99": float(np.percentile(v, 99)), "max": float(v.max())}
    report["vectors"][name] = {"cosine": pct(cos), "rel_l2": pct(rel)}

case_rows = {}
labels, attacked = {}, {}
for i, r in enumerate(rows):
    labels[r["case_id"]] = bool(r["attack_success"])
    attacked[r["case_id"]] = bool(r["attack_present"])
    if r["step"] < ACTION_STEP:
        case_rows.setdefault(r["case_id"], []).append(i)
ids = sorted(labels)
att = [c for c in ids if attacked[c]]


def oof(pool, lab):
    folds = spe._stratified_case_folds(pool, lab, folds=FOLDS, seed=SEED)
    out = {name: {} for name in sources}
    for test_ids in folds:
        test = set(test_ids)
        tr = [i for c in pool if c not in test for i in case_rows.get(c, [])]
        m = logreg_fit(ref[tr], np.array([lab[rows[i]["case_id"]] for i in tr], dtype=float))
        for c in test_ids:
            idx = case_rows.get(c, [])
            for name, X in sources.items():
                out[name][c] = float(np.max(logreg_score(m, X[idx]))) if idx else 0.5
    return out


contrasts = {
    "commitment": (att, {c: labels[c] for c in att}),
    "naive": (ids, {c: labels[c] for c in ids}),
}
y = [labels[c] for c in att]
for contrast, (pool, lab) in contrasts.items():
    sc = oof(pool, lab)
    report["probe"][contrast] = {name: round(auroc([s[c] for c in att], y), 3) for name, s in sc.items()}
    for name, s in sc.items():
        if name == "stored":
            continue
        d = np.array([abs(s[c] - sc["stored"][c]) for c in att])
        flips = sum((s[c] >= 0.5) != (sc["stored"][c] >= 0.5) for c in att)
        report["scores"][f"{contrast}/{name}"] = {
            "max_abs_dp": float(d.max()), "median_abs_dp": float(np.median(d)), "verdict_flips_at_0.5": int(flips), "n": len(att)}

print(json.dumps(report, indent=2))
(data / "parity_report.json").write_text(json.dumps(report, indent=2))
