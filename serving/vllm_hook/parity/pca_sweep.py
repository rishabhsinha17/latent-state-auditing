"""Probe robustness with a PCA front end (CPU). Same protocol as l2_sweep.py.

Per fold: center on the training rows, project onto the top-k principal
directions of the training rows, then the paper's logreg_fit. Folding the PCA
into the probe gives an ordinary (mean, scale, weights, bias) artifact.
"""
import json, sys
from pathlib import Path
import numpy as np
repo = Path(__file__).resolve().parents[3]  # repository root
sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / "analysis"))
from missing_cells import ACTION_STEP, FOLDS, SEED, auroc, logreg_fit, logreg_score
from latent_agent_auditing.experiments import summarize_paper_evidence as spe

rows = [json.loads(l) for l in open("data/exp_ws_full/prompts.jsonl")]
S = {"stored": np.load("data/exp_ws_full/hf_layer20.npy").astype(np.float64),
     "vllm_full": np.load("out/vllm_full.npy").astype(np.float64),
     "hf_fresh": np.load("out/hf_fresh.npy").astype(np.float64)}
labels, attacked, case_rows = {}, {}, {}
for i, r in enumerate(rows):
    labels[r["case_id"]] = bool(r["attack_success"]); attacked[r["case_id"]] = bool(r["attack_present"])
    if r["step"] < ACTION_STEP:
        case_rows.setdefault(r["case_id"], []).append(i)
att = [c for c in sorted(labels) if attacked[c]]
lab = {c: labels[c] for c in att}
folds = spe._stratified_case_folds(att, lab, folds=FOLDS, seed=SEED)
y = [lab[c] for c in att]
out = {}
for k in [int(x) for x in sys.argv[1:]]:
    sc = {n: {} for n in S}
    for test_ids in folds:
        t = set(test_ids)
        tr = [i for c in att if c not in t for i in case_rows[c]]
        Xtr = S["stored"][tr]; mu = Xtr.mean(0)
        _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
        P = Vt[:k].T
        m = logreg_fit((Xtr - mu) @ P, np.array([lab[rows[i]["case_id"]] for i in tr], float))
        for c in test_ids:
            for n, X in S.items():
                sc[n][c] = float(np.max(logreg_score(m, (X[case_rows[c]] - mu) @ P)))
    res = {"auroc_stored": round(auroc([sc["stored"][c] for c in att], y), 3)}
    for n in ("vllm_full", "hf_fresh"):
        d = [abs(sc[n][c] - sc["stored"][c]) for c in att]
        res[n] = {"auroc": round(auroc([sc[n][c] for c in att], y), 3),
                  "flips": sum((sc[n][c] >= .5) != (sc["stored"][c] >= .5) for c in att),
                  "max_dp": round(max(d), 3)}
    out[k] = res
    print(k, json.dumps(res), flush=True)
Path("out/pca_sweep.json").write_text(json.dumps(out, indent=2))
