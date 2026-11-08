"""Why does a second prefill probe not lower FPR? Cheap CPU diagnostics on grouped OOF scores.

OOF scores: 5 folds grouped by user task, seeds 13, 0, 1, 2. Probes fixed in advance:
  S1    layer 28, PCA-32 + logistic, trained on attacked steps (released recipe)
  L21   layer 21, PCA-64 + logistic, trained on attacked steps
  PRES  injection-presence probe: layers 14+21+28 concatenated, PCA-64 + logistic, trained on
        every step with y = attack present (uses no outcome label)

D1 rank correlation of S1 and L21 (all attacked; S1-flagged attacked-but-safe)
D2 conditional AUROC of L21 inside the S1-flagged region (can stage 2 separate S1's flags?)
D3 ORACLE upper bound, thresholds tuned on the evaluated OOF labels (descriptive only)
D4 label and action breakdown of S1 flags at its 90%-recall OOF threshold
D5 paired exposure check: per user task, attacked-but-safe S1 score minus the clean step's score
D6 PRES AUROC, attacked vs clean (is a presence stage 2 feasible from cached features?)
D7 SEMI-SYNTHETIC value of a stage 2: real S1 scores, synthetic stage 2 of given conditional
   AUROC (independent of S1 given the label); thresholds from half the user tasks, FPR on the rest

Usage: diagnose.py <chat dir> <cascade.py> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Data, Probe, auroc, group_bootstrap, load_cascade, spearman  # noqa: E402

chat, cas_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cascade = load_cascade(cas_path)
D = Data(chat)
CAT = np.hstack([D.X[14], D.X[21], D.X[28]])
SEEDS = (13, 0, 1, 2)
att, uns = D.attacked, D.unsafe
safe = att & ~uns

oof = {}
for seed in SEEDS:
    s = {k: np.full(D.n, np.nan) for k in ("S1", "L21", "PRES")}
    for te in D.folds(np.arange(D.n), seed):
        tr = np.setdiff1d(np.arange(D.n), te)
        a = tr[att[tr]]
        s["S1"][te] = Probe(32).fit(D.X[28][a], uns[a]).score(D.X[28][te])
        s["L21"][te] = Probe(64).fit(D.X[21][a], uns[a]).score(D.X[21][te])
        s["PRES"][te] = Probe(64).fit(CAT[tr], att[tr]).score(CAT[te])
    oof[seed] = s
    print(f"seed {seed} OOF done", flush=True)


def per_seed(fn):
    v = [fn(oof[s]) for s in SEEDS]
    return [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)]


def thr90(s):
    return cascade.threshold_for_recall(s["S1"][uns], 0.9)


R = {}
R["D1_spearman_S1_L21_attacked"] = per_seed(lambda s: spearman(s["S1"][att], s["L21"][att]))
R["D1_spearman_S1_L21_flagged_safe"] = per_seed(
    lambda s: spearman(s["S1"][safe & (s["S1"] >= thr90(s))], s["L21"][safe & (s["S1"] >= thr90(s))]))
R["D2_L21_auroc_all_attacked"] = per_seed(lambda s: auroc(s["L21"][att], uns[att]))
R["D2_S1_auroc_within_S1_flagged"] = per_seed(
    lambda s: auroc(s["S1"][att & (s["S1"] >= thr90(s))], uns[att & (s["S1"] >= thr90(s))]))
R["D2_L21_auroc_within_S1_flagged"] = per_seed(
    lambda s: auroc(s["L21"][att & (s["S1"] >= thr90(s))], uns[att & (s["S1"] >= thr90(s))]))


def oracle(s):
    y = uns[att]
    single = float(np.mean(s["S1"][safe] >= thr90(s)))
    casc = cascade.calibrate_cascade(s["S1"][att], s["L21"][att], y, 0.9).fpr
    avg = (s["S1"] + s["L21"]) / 2
    av = float(np.mean(avg[safe] >= cascade.threshold_for_recall(avg[uns], 0.9)))
    return single, casc, av


o = [oracle(oof[s]) for s in SEEDS]
R["D3_ORACLE_fpr_at_recall90"] = {k: [round(float(np.mean([x[i] for x in o])), 3), round(float(np.std([x[i] for x in o])), 3)]
                                  for i, k in enumerate(("S1_single", "cascade_S1_L21", "average_S1_L21"))}

# D4: where do S1's flags land?
lab = D.labels
kind = np.array(["clean" if not a else ("unsafe_strict_and_follows" if lab[c]["strict"] and lab[c]["follows"] else
                 "unsafe_follows_only" if lab[c]["follows"] else "unsafe_strict_only" if lab[c]["strict"] else "safe")
                 for c, a in zip(D.ids, att)])
calls = np.array([bool(g["calls"]) for g in D.gen])
groups4 = {
    "unsafe_strict_and_follows": kind == "unsafe_strict_and_follows",
    "unsafe_follows_only": kind == "unsafe_follows_only",
    "unsafe_strict_only": kind == "unsafe_strict_only",
    "attacked_safe_tool_call": (kind == "safe") & calls,
    "attacked_safe_text_only": (kind == "safe") & ~calls,
    "clean_tool_call": (kind == "clean") & calls,
    "clean_text_only": (kind == "clean") & ~calls,
}
R["D4_flag_rate_at_S1_recall90"] = {k: {"n": int(m.sum()), "flag_rate": per_seed(lambda s, m=m: np.mean(s["S1"][m] >= thr90(s)))}
                                    for k, m in groups4.items()}
R["D4_flag_rate_by_suite_attacked_safe"] = {su: per_seed(lambda s, su=su: np.mean(s["S1"][safe & (D.suite == su)] >= thr90(s)))
                                            for su in ("workspace", "travel", "banking", "slack")}

# D5: paired exposure check within user task
clean_of = {D.groups[i]: i for i in np.flatnonzero(~att)}
pair_rows = np.array([i for i in np.flatnonzero(safe) if D.groups[i] in clean_of])


def paired(s, rows=None):
    rows = pair_rows if rows is None else rows
    return float(np.mean([s["S1"][i] - s["S1"][clean_of[D.groups[i]]] for i in rows]))


R["D5_paired_safe_minus_clean_S1"] = {"n_pairs": int(len(pair_rows)), "mean_diff": per_seed(paired),
                                      "ci95_seed13": group_bootstrap(D.groups[pair_rows], lambda r: paired(oof[13], pair_rows[r]))}
R["D5_exposure_auroc_safe_vs_clean"] = per_seed(lambda s: auroc(np.r_[s["S1"][safe], s["S1"][~att]], np.r_[np.ones(safe.sum()), np.zeros((~att).sum())]))

# D6: injection presence decodable?
R["D6_PRES_auroc_attacked_vs_clean"] = per_seed(lambda s: auroc(s["PRES"], att))
R["D6_PRES_auroc_attacked_vs_clean_by_suite"] = {su: per_seed(lambda s, su=su: auroc(s["PRES"][D.suite == su], att[D.suite == su]))
                                                for su in ("workspace", "travel", "banking", "slack")}

# D7: semi-synthetic value of a stage 2
rng = np.random.default_rng(0)
keys = np.array(sorted(set(D.groups)))
D7 = {}
for a_target in (0.6, 0.7, 0.8, 0.9, 0.95):
    d = np.sqrt(2) * __import__("statistics").NormalDist().inv_cdf(a_target)
    res = []
    for seed in SEEDS:
        for rep in range(5):
            half = set(rng.choice(keys, len(keys) // 2, replace=False))
            cal = att & np.isin(D.groups, list(half)); tst = att & ~np.isin(D.groups, list(half))
            s2 = rng.normal(size=D.n) + d * uns
            s1 = oof[seed]["S1"]
            op = cascade.calibrate_cascade(s1[cal], s2[cal], uns[cal], 0.9)
            ev = cascade.evaluate_cascade(s1[tst], s2[tst], uns[tst], op.stage1_threshold, op.stage2_threshold)
            t = cascade.threshold_for_recall(s1[cal & uns], 0.9)
            res.append((ev.recall, ev.fpr, float(np.mean(s1[tst & uns] >= t)), float(np.mean(s1[tst & ~uns] >= t))))
    res = np.array(res)
    D7[f"stage2_auroc_{a_target}"] = {"cascade_recall": round(float(res[:, 0].mean()), 3), "cascade_fpr": round(float(res[:, 1].mean()), 3),
                                      "single_recall": round(float(res[:, 2].mean()), 3), "single_fpr": round(float(res[:, 3].mean()), 3)}
R["D7_SEMISYNTHETIC_value_of_stage2"] = D7

out.write_text(json.dumps({"protocol": __doc__, "results": R}, indent=1))
np.savez_compressed(out.with_suffix(".oof.npz"), **{f"{k}_seed{s}": oof[s][k] for s in SEEDS for k in oof[s]})
for k, v in R.items():
    print(k, json.dumps(v))
