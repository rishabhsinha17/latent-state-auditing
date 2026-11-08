"""Per-suite clean and attacked-but-safe FPR, and recall, at the 0.9 recall target from eval_r1 nested flags.

Usage: per_suite_r1.py <chat dir> <out_r1 dir>   (writes <out_r1>/per_suite_r1.json)
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SUITES, Data  # noqa: E402

chat, o = Path(sys.argv[1]), Path(sys.argv[2])
D = Data(chat, layers=(28,))
F = {**dict(np.load(o / "eval_r1_flags.npz")), **dict(np.load(o / "eval_r1_posthoc_flags.npz"))}
res = {}
for su in SUITES:
    m = D.suite == su
    res[su] = {"n_unsafe": int((m & D.unsafe).sum()), "n_attacked_safe": int((m & D.attacked & ~D.unsafe).sum()),
               "n_clean": int((m & ~D.attacked).sum())}
    for n in ("S1", "C-PRES", "C-PRES28"):
        f = [F[f"{n}|recall=0.9|{s}"] for s in (13, 0, 1, 2)]
        res[su][n] = {k: round(float(np.mean([x[m & sel].mean() for x in f])), 3) for k, sel in
                      (("recall", D.unsafe), ("fpr_attacked_safe", D.attacked & ~D.unsafe), ("fpr_clean", ~D.attacked))}
(o / "per_suite_r1.json").write_text(json.dumps(res, indent=1))
for su, v in res.items():
    print(su, json.dumps(v))
