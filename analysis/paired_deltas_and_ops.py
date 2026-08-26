"""Paired AUROC deltas, operating points, and the pooled transfer cell.

Reuses the exact fold construction, models, and loaders of the released
commitment_probes.py / reviewer_response_analyses.py / cross_transfer.py, then:
  (1) SANITY GATE: recomputed base AUROCs must reproduce the published values
      (EXP_commitment.json / EXP_reviewer_response.json / EXP_xfer_*.json) to
      3 decimals, else the script aborts and writes nothing;
  (2) paired case-level (and user-task cluster) bootstrap CIs on AUROC
      DIFFERENCES behind the "parity" / "equal or lead" claims;
  (3) FPR@90TPR (out-of-fold, seed-13 stratified folds) for the
      commitment-contrast monitors;
  (4) one pooled transfer cell (held-out templates + travel, scored together).

Writes EXP_paired_deltas.json. Run from analysis/:
  python3 paired_deltas_and_ops.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "repo-latent-state-auditing"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from latent_agent_auditing.detectors.activation_probe import CentroidActivationProbe  # noqa: E402
from latent_agent_auditing.detectors.text_classifier import BagOfWordsNaiveBayes  # noqa: E402
from latent_agent_auditing.eval.metrics import binary_metrics  # noqa: E402
from latent_agent_auditing.experiments import summarize_paper_evidence as spe  # noqa: E402
from latent_agent_auditing.logging.jsonl import read_jsonl  # noqa: E402
from reviewer_response_analyses import TfidfLogreg  # noqa: E402
from commitment_probes import logreg_fit, logreg_score  # noqa: E402

SEED = 13
FOLDS = 5
RESAMPLES = 1000
ACTION_STEP = 3
RUNS = HERE.parents[1] / "pod-harvest" / "lsa" / "runs"


def auroc(scores, labels):
    return binary_metrics(list(scores), list(labels)).auroc


def load(run):
    case_rows = read_jsonl(run / "case_scores.jsonl")
    cases = {str(c["id"]): c for c in read_jsonl(run / "cases.jsonl")}
    audits = read_jsonl(run / "audits.jsonl")
    labels = {str(r["case_id"]): bool(r["attack_success"]) for r in case_rows}
    attacked = {cid: bool(c.get("attack_present")) for cid, c in cases.items()}
    text_pre = defaultdict(list)
    for a in audits:
        act = a["activation"]
        if act["step"] < ACTION_STEP:
            for e in a.get("nla_explanations", []):
                t = e if isinstance(e, str) else (e.get("text") or e.get("explanation") or "")
                if t:
                    text_pre[act["case_id"]].append(t)
    acts_all = spe._activations_by_case(audits)
    acts_pre = {cid: [r for r in recs if getattr(r, "step", 99) < ACTION_STEP] for cid, recs in acts_all.items()}
    att_ids = sorted(c for c in labels if attacked[c])

    def vis(cid):
        c = cases[cid]
        return f"{c.get('user_instruction', '')} {c.get('external_content', '')}"

    return att_ids, {c: labels[c] for c in att_ids}, text_pre, acts_pre, vis


def user_task(cid):
    m = re.search(r"(user_task_\d+)", cid)
    return m.group(1) if m else cid


def fit_all(train_ids, lab, text_pre, acts_pre, vis):
    tn = BagOfWordsNaiveBayes()
    tn.fit([" ".join(text_pre.get(c, [])) for c in train_ids], [lab[c] for c in train_ids])
    tt = BagOfWordsNaiveBayes()
    tt.fit([vis(c) for c in train_ids], [lab[c] for c in train_ids])
    tf = TfidfLogreg()
    tf.fit([vis(c) for c in train_ids], [lab[c] for c in train_ids])
    ta, tl = [], []
    for c in train_ids:
        for r in acts_pre.get(c, []):
            ta.append(r)
            tl.append(lab[c])
    apb = CentroidActivationProbe()
    apb.fit(ta, tl)
    lg = logreg_fit(np.array([r.vector for r in ta]), np.array(tl, dtype=float))
    return tn, tt, tf, apb, lg


def score_all(ids, models, text_pre, acts_pre, vis):
    tn, tt, tf, apb, lg = models
    out = defaultdict(dict)
    tf_scores = tf.score_many([vis(c) for c in ids])
    for i, c in enumerate(ids):
        out["nla_text"][c] = tn.score(" ".join(text_pre.get(c, [])))
        out["transcript"][c] = tt.score(vis(c))
        out["tfidf"][c] = float(tf_scores[i])
        s = [apb.score(r) for r in acts_pre.get(c, [])]
        out["centroid_max"][c] = max(s) if s else 0.5
        out["logreg_max"][c] = (
            float(np.max(logreg_score(lg, np.array([r.vector for r in acts_pre[c]]))))
            if acts_pre.get(c) else 0.5
        )
    return out


def run_folds(att_ids, lab, fold_lists, text_pre, acts_pre, vis):
    oof = defaultdict(dict)
    for test_ids in fold_lists:
        train_ids = [c for c in att_ids if c not in set(test_ids)]
        models = fit_all(train_ids, lab, text_pre, acts_pre, vis)
        part = score_all(test_ids, models, text_pre, acts_pre, vis)
        for m, d in part.items():
            oof[m].update(d)
    return oof


def paired_delta(ids, sa, sb, y, clusters=None):
    """Bootstrap CI on auroc(sa) - auroc(sb) over the SAME resamples."""
    a = np.array([sa[c] for c in ids])
    b = np.array([sb[c] for c in ids])
    yy = np.array([y[c] for c in ids])
    point = auroc(a, yy) - auroc(b, yy)
    rng = random.Random(SEED)
    deltas = []
    if clusters is None:
        idx = list(range(len(ids)))
        for _ in range(RESAMPLES):
            pick = [rng.choice(idx) for _ in idx]
            if len(set(yy[pick])) < 2:
                continue
            deltas.append(auroc(a[pick], yy[pick]) - auroc(b[pick], yy[pick]))
    else:
        groups = defaultdict(list)
        for i, c in enumerate(ids):
            groups[clusters(c)].append(i)
        keys = sorted(groups)
        for _ in range(RESAMPLES):
            pick = [i for k in (rng.choice(keys) for _ in keys) for i in groups[k]]
            if len(set(yy[pick])) < 2:
                continue
            deltas.append(auroc(a[pick], yy[pick]) - auroc(b[pick], yy[pick]))
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas)) - 1]
    return {"delta": round(point, 3), "lower": round(lo, 3), "upper": round(hi, 3), "resamples": len(deltas)}


def gate(name, got, want, tol=0.0015):
    ok = abs(got - want) <= tol
    print(f"  gate {name}: recomputed {got:.3f} vs published {want:.3f} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    out = {}
    ws = RUNS / "exp_ws_full"
    att_ids, lab, text_pre, acts_pre, vis = load(ws)
    y = {c: lab[c] for c in att_ids}
    yy = [lab[c] for c in att_ids]

    # ---------- stratified seed-13 folds ----------
    folds = spe._stratified_case_folds(att_ids, lab, folds=FOLDS, seed=SEED)
    oof = run_folds(att_ids, lab, folds, text_pre, acts_pre, vis)
    base = {m: round(auroc([sc[c] for c in att_ids], yy), 3) for m, sc in oof.items()}
    print("stratified base:", base)

    comm = json.load(open(HERE / "EXP_commitment.json"))["stratified_seed13"]
    rr = json.load(open(HERE / "EXP_reviewer_response.json"))
    gates = [
        gate("logreg", base["logreg_max"], comm["logreg_max"]["auroc"]),
        gate("centroid", base["centroid_max"], comm["centroid_max"]["auroc"]),
        gate("nla_text", base["nla_text"], comm["nla_text"]["auroc"]),
        gate("nb_visible", base["transcript"], comm["transcript"]["auroc"]),
        gate("tfidf", base["tfidf"], rr["tfidf_stratified"]["auroc"]),
    ]
    if not all(gates):
        sys.exit("SANITY GATE FAILED - nothing written")

    out["base_stratified"] = base
    out["paired_deltas_stratified"] = {
        "logreg_minus_tfidf": paired_delta(att_ids, oof["logreg_max"], oof["tfidf"], y),
        "centroid_minus_tfidf": paired_delta(att_ids, oof["centroid_max"], oof["tfidf"], y),
        "logreg_minus_nb_visible": paired_delta(att_ids, oof["logreg_max"], oof["transcript"], y),
        "logreg_minus_tfidf_cluster": paired_delta(att_ids, oof["logreg_max"], oof["tfidf"], y, clusters=user_task),
    }
    out["fpr_at_90tpr_stratified_oof"] = {
        m: round(binary_metrics([sc[c] for c in att_ids], yy).fpr_at_90_tpr, 3) for m, sc in oof.items()
    }

    # ---------- grouped folds (seed 13 assignment) ----------
    tasks = sorted({user_task(c) for c in att_ids})
    rng = random.Random(SEED)
    rng.shuffle(tasks)
    gfolds = [[c for c in att_ids if user_task(c) in set(tasks[i::FOLDS])] for i in range(FOLDS)]
    goof = run_folds(att_ids, lab, gfolds, text_pre, acts_pre, vis)
    gbase = {m: round(auroc([sc[c] for c in att_ids], yy), 3) for m, sc in goof.items()}
    print("grouped (seed 13) base:", gbase)
    comm_g = json.load(open(HERE / "EXP_commitment.json"))["grouped_per_seed"]["13"]
    if not (gate("grouped logreg", gbase["logreg_max"], comm_g["logreg_max"]) and
            gate("grouped tfidf", gbase["tfidf"], rr["tfidf_grouped_per_seed"]["13"])):
        sys.exit("SANITY GATE FAILED (grouped) - nothing written")
    out["base_grouped_seed13"] = gbase
    out["paired_deltas_grouped_seed13"] = {
        "logreg_minus_tfidf": paired_delta(att_ids, goof["logreg_max"], goof["tfidf"], y, clusters=user_task),
        "centroid_minus_tfidf": paired_delta(att_ids, goof["centroid_max"], goof["tfidf"], y, clusters=user_task),
    }

    # ---------- pooled transfer cell ----------
    models = fit_all(att_ids, lab, text_pre, acts_pre, vis)
    pooled_ids, pooled_y, pooled_scores = [], {}, defaultdict(dict)
    per_set = {}
    for name, run in (("templates", RUNS / "exp_ws_templates"), ("travel", RUNS / "exp_travel")):
        t_ids, t_lab, t_text, t_acts, t_vis = load(run)
        sc = score_all(t_ids, models, t_text, t_acts, t_vis)
        per_set[name] = {m: round(auroc([sc[m][c] for c in t_ids], [t_lab[c] for c in t_ids]), 3) for m in sc}
        for c in t_ids:
            key = f"{name}:{c}"
            pooled_ids.append(key)
            pooled_y[key] = t_lab[c]
            for m in sc:
                pooled_scores[m][key] = sc[m][c]
    print("per-set transfer (recomputed):", per_set)
    xt = json.load(open(HERE / "EXP_xfer_templates.json"))["monitors"]
    xv = json.load(open(HERE / "EXP_xfer_travel.json"))["monitors"]
    if not (gate("xfer templates centroid", per_set["templates"]["centroid_max"], xt["centroid_max"]["auroc"]) and
            gate("xfer travel centroid", per_set["travel"]["centroid_max"], xv["centroid_max"]["auroc"])):
        sys.exit("SANITY GATE FAILED (transfer) - nothing written")

    py = [pooled_y[c] for c in pooled_ids]
    pooled = {}
    for m in pooled_scores:
        a = np.array([pooled_scores[m][c] for c in pooled_ids])
        yv = np.array(py)
        point = auroc(a, yv)
        rngp = random.Random(SEED)
        groups = defaultdict(list)
        for i, c in enumerate(pooled_ids):
            groups[user_task(c)].append(i)
        keys = sorted(groups)
        boots = []
        for _ in range(RESAMPLES):
            pick = [i for k in (rngp.choice(keys) for _ in keys) for i in groups[k]]
            if len(set(yv[pick])) < 2:
                continue
            boots.append(auroc(a[pick], yv[pick]))
        boots.sort()
        pooled[m] = {
            "auroc": round(point, 3),
            "lower": round(boots[int(0.025 * len(boots))], 3),
            "upper": round(boots[int(0.975 * len(boots)) - 1], 3),
            "n": len(pooled_ids), "unsafe": int(sum(py)),
        }
    out["pooled_transfer_cluster_bootstrap"] = pooled
    out["per_set_transfer_recomputed"] = per_set

    (HERE / "EXP_paired_deltas.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
