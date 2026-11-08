"""POST HOC control for the C-PRES result: is the presence stage just a prompt-length detector?

Injected tool outputs add tokens, so attacked prompts are longer than their clean twins. This
checks (1) how well prompt length alone separates attacked from clean, (2) the presence probe's
AUROC inside prompt-length quartiles, and (3) split-half cascades with stage 2 = presence probe,
prompt length, or the presence score with length regressed out (fit on the calibration half).
Scores are diagnose.py's grouped OOF scores (single level). Calibration on a random half of the
user tasks, metrics on the other half, 20 halves x 4 seeds, target recall 0.9.
Also: per-suite clean FPR of S1 vs C-PRES from eval_r1.py's nested held-out flags.

Usage: control_presence.py <chat dir> <cascade.py> <out_r1 dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SUITES, Data, auroc, load_cascade, spearman  # noqa: E402

chat, cas_path, outdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
C = load_cascade(cas_path)
D = Data(chat, layers=(28,))
att, uns = D.attacked, D.unsafe
length = np.array([g["n_prompt_tokens"] for g in D.gen], float)
O = np.load(outdir / "diagnose.oof.npz")
SEEDS = (13, 0, 1, 2)
R = {}
R["length_auroc_attacked_vs_clean"] = round(auroc(length, att), 3)
R["length_auroc_by_suite"] = {su: round(auroc(length[D.suite == su], att[D.suite == su]), 3) for su in SUITES}
R["spearman_PRES_length"] = round(float(np.mean([spearman(O[f"PRES_seed{s}"], length) for s in SEEDS])), 3)
q = np.quantile(length, [0.25, 0.5, 0.75])
stratum = np.digitize(length, q)
R["PRES_auroc_within_length_quartile"] = {
    int(k): {"n_clean": int(((stratum == k) & ~att).sum()),
             "auroc": round(float(np.mean([auroc(O[f"PRES_seed{s}"][stratum == k], att[stratum == k]) for s in SEEDS])), 3)}
    for k in range(4)}

rng = np.random.default_rng(1)
keys = np.array(sorted(set(D.groups)))
res = {k: [] for k in ("S1", "C-PRES", "C-LEN", "C-PRES-RESID")}
for s in SEEDS:
    s1, pres = O[f"S1_seed{s}"], O[f"PRES_seed{s}"]
    for _ in range(20):
        half = np.isin(D.groups, rng.choice(keys, len(keys) // 2, replace=False))
        cal, tst = half, ~half
        X = np.c_[np.ones(cal.sum()), np.log(length[cal])]
        beta = np.linalg.lstsq(X, pres[cal], rcond=None)[0]
        resid = pres - np.c_[np.ones(D.n), np.log(length)] @ beta
        t = C.threshold_for_recall(s1[cal & uns], 0.9)
        f = s1 >= t
        res["S1"].append((f[tst & uns].mean(), f[tst & att & ~uns].mean(), f[tst & ~att].mean()))
        for name, s2 in (("C-PRES", pres), ("C-LEN", np.log(length)), ("C-PRES-RESID", resid)):
            op = C.calibrate_cascade(s1[cal], s2[cal], uns[cal], 0.9)
            f = C.cascade_flags(s1, s2, op.stage1_threshold, op.stage2_threshold)
            res[name].append((f[tst & uns].mean(), f[tst & att & ~uns].mean(), f[tst & ~att].mean()))
R["split_half_at_recall_0.9"] = {k: {m: round(float(np.mean([x[i] for x in v])), 3)
                                     for i, m in enumerate(("recall", "fpr_attacked_safe", "fpr_clean"))}
                                 for k, v in res.items()}

F = np.load(outdir / "eval_r1_flags.npz")
R["nested_clean_fpr_by_suite_recall_0.9"] = {
    su: {"n_clean": int(((D.suite == su) & ~att).sum()),
         **{n: round(float(np.mean([F[f"{n}|recall=0.9|{s}"][(D.suite == su) & ~att].mean() for s in SEEDS])), 3)
            for n in ("S1", "C-PRES")}}
    for su in SUITES}
(outdir / "control_presence.json").write_text(json.dumps({"protocol": __doc__, "results": R}, indent=1))
for k, v in R.items():
    print(k, json.dumps(v))
