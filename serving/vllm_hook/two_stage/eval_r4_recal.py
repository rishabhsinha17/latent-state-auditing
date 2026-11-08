"""r4: unseen tool suite WITH in-domain recalibration (CPU).

PRE-SPECIFIED, committed before this script was first run on any dataset.
Same data, recipes and detectors as eval_r4.py. For each held-out suite: probes trained on the
other three suites only (as split B, no retraining); the suite's user tasks are split in half
(10 random halvings per suite, seeds 0-9); thresholds are calibrated on one half's scores
(labelled, target recall 0.90; cascades with calibrate_cascade as configured) and evaluated
on the other half. This is the per-deployment recalibration recipe: only thresholds move.
Pooled over suites and halvings: recall, FPR attacked-but-safe, FPR clean.
Pre-specified paired comparisons (bootstrap 2000 over the evaluation halves' user tasks,
averaged over halvings): C-PD/gate97 - S1, C-PD/joint - S1, C-PRES/joint - S1.
Usage: eval_r4_recal.py <eval_r4.py args>
"""
import json, sys
from pathlib import Path
import numpy as np
src = open(Path(__file__).resolve().parent / "eval_r4.py").read()
exec(src.split("report = {\"protocol\"")[0])

HALVINGS = 10
flagsets = [ {n: np.zeros(len(ids), bool) for n, _, _ in DETS} for _ in range(HALVINGS)]
evaluated = [np.zeros(len(ids), bool) for _ in range(HALVINGS)]
for su in sorted(set(suite[rows_all])):
    tr = rows_all[suite[rows_all] != su]
    held = rows_all[suite[rows_all] == su]
    sc = fit_score(tr, held)
    full = {k: np.full(len(ids), np.nan) for k in sc}
    for k, v in sc.items():
        full[k][held] = v
    keys = sorted(set(groups[held]))
    for h in range(HALVINGS):
        ks = keys[:]
        random.Random(h).shuffle(ks)
        cal = held[np.isin(groups[held], ks[: len(ks) // 2])]
        ev = held[~np.isin(groups[held], ks[: len(ks) // 2])]
        y = uns[cal]
        if y.sum() < 5 or (~y).sum() < 5:
            continue
        for name, k2, gate in DETS:
            if k2 is None:
                f = full["S1"][ev] >= C.threshold_for_recall(full["S1"][cal][y], 0.9)
            else:
                try:
                    op = C.calibrate_cascade(full["S1"][cal], full[k2][cal], y, 0.9, stage1_recall=gate)
                except ValueError:
                    continue
                f = C.cascade_flags(full["S1"][ev], full[k2][ev], op.stage1_threshold, op.stage2_threshold)
            flagsets[h][name][ev] = f
        evaluated[h][ev] = True
    print(f"{su} done", flush=True)

rows_eval = np.flatnonzero(np.any(evaluated, 0))


def mean_metrics(n, rows):
    vals = [metrics(flagsets[h][n], rows[evaluated[h][rows]]) for h in range(HALVINGS) if evaluated[h][rows].any()]
    return np.nanmean(vals, 0)


summary = {n: {k: round(float(v), 3) for k, v in zip(("recall", "fpr_attacked_safe", "fpr_clean"), mean_metrics(n, rows_eval))}
           for n, _, _ in DETS}
rng = np.random.default_rng(0)
gk = np.array(sorted(set(groups[rows_eval])))
mem = {g: rows_eval[groups[rows_eval] == g] for g in gk}
pairs = [("C-PD/gate97", "S1"), ("C-PD/joint", "S1"), ("C-PRES/joint", "S1")]
draws = {p: [] for p in pairs}
for _ in range(BOOT):
    r = np.concatenate([mem[g] for g in rng.choice(gk, len(gk))])
    for a, b in pairs:
        draws[(a, b)].append(mean_metrics(a, r) - mean_metrics(b, r))
paired = {f"{a} - {b}": {k: [round(float(np.nanpercentile(np.array(d)[:, i], 2.5)), 3), round(float(np.nanpercentile(np.array(d)[:, i], 97.5)), 3)]
                         for i, k in enumerate(("d_recall", "d_fpr_attacked_safe", "d_fpr_clean"))} for (a, b), d in draws.items()}
out.mkdir(parents=True, exist_ok=True)
(out / f"eval_r4_recal_{TAG}.json").write_text(json.dumps({"protocol": __doc__, "tag": TAG, "summary": summary, "paired_95ci": paired}, indent=1))
print(json.dumps(summary)); print(json.dumps(paired))
