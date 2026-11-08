"""POST HOC, DESCRIPTIVE (after eval_r3b): decision probe vs presence x tool-name at MATCHED recall.

eval_r3b compared the two cascades at their calibrated operating points, where C-PT landed 2.6
recall points higher, so the FPR gap mixes information with operating point. Here, per seed, the
held-out (outer-fold) scores of S1, PRES28*DEC and PRES28*TOOL are pooled, the stage-1 gate is set
at 97% recall of the held-out positives, and the stage-2 threshold is swept to read the
attacked-but-safe FPR at recall 0.86, 0.88, 0.90 and 0.92. Using test labels to place thresholds
makes this a comparison of score quality, not a deployable result.
Usage: posthoc_r3b.py <same args as eval_r3b.py>
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
exec(open(Path(__file__).resolve().parent / "eval_r3b.py").read().split("flags = {}")[0])   # data, recipes, fit_score

held = {s: {k: np.full(D.n, np.nan) for k in ("S1", "PD", "PT")} for s in SEEDS}
for seed in SEEDS:
    for te in D.folds(rows_all, seed):
        for k, v in fit_score(np.setdiff1d(rows_all, te), te).items():
            held[seed][k][te] = v
RECALLS = (0.86, 0.88, 0.90, 0.92)


def fpr_at(s1, s2, rows, r):
    y = uns[rows]; a, b = s1[rows], s2[rows]
    t1 = C.threshold_for_recall(a[y], 0.97)
    passed = np.sort(b[(a >= t1) & y])[::-1]
    need = int(np.ceil(r * y.sum() - 1e-9))
    if need > passed.size:
        return np.nan
    t2 = passed[need - 1]
    f = (a >= t1) & (b >= t2)
    return float(f[att[rows] & ~y].mean())


res = {}
for k in ("PD", "PT"):
    res[k] = {str(r): round(float(np.mean([fpr_at(held[s]["S1"], held[s][k], rows_all, r) for s in SEEDS])), 3) for r in RECALLS}
rng = np.random.default_rng(0)
keys = np.array(sorted(set(D.groups[rows_all]))); mem = {g: rows_all[D.groups[rows_all] == g] for g in keys}
ci = {}
for r in RECALLS:
    d = []
    for _ in range(1000):
        rows = np.concatenate([mem[g] for g in rng.choice(keys, len(keys))])
        d.append(np.mean([fpr_at(held[s]["S1"], held[s]["PD"], rows, r) - fpr_at(held[s]["S1"], held[s]["PT"], rows, r) for s in SEEDS]))
    ci[str(r)] = [round(float(np.nanpercentile(d, 2.5)), 3), round(float(np.nanpercentile(d, 97.5)), 3)]
outp = {"protocol": __doc__, "fpr_attacked_safe_at_matched_recall": res, "ci95_PD_minus_PT": ci}
(out / "posthoc_r3b.json").write_text(json.dumps(outp, indent=1))
print(json.dumps(res)); print(json.dumps(ci))
