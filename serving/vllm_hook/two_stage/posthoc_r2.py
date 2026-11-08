"""POST HOC (after eval_r2 results were seen): is the decision-time stage more than the tool name,
and how do the r2 cascades split by suite? Reads eval_r2_flags.npz; same kept steps and seeds.

Usage: posthoc_r2.py <chat dir> <dec npz> <out_r2 dir>
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SUITES, Data  # noqa: E402

chat, dec_path, o = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
D = Data(chat, layers=(28,))
E = np.load(dec_path)
dec = E["dec_layer_28"]
keep = E["same_action"].astype(bool) & ~np.isnan(dec.astype(np.float32)).all(2).all(1)
F = dict(np.load(o / "eval_r2_flags.npz"))
SEEDS = (13, 0, 1, 2)
rows_all = np.flatnonzero(keep)
att, uns = D.attacked, D.unsafe


def m(n, rows, mode="recall=0.9"):
    v = [F[f"{n}|{mode}|{s}"] for s in SEEDS]
    return np.mean([[f[rows][uns[rows]].mean(), f[rows][att[rows] & ~uns[rows]].mean(),
                     f[rows][~att[rows]].mean() if (~att[rows]).any() else np.nan] for f in v], 0)


rng = np.random.default_rng(0)
keys = np.array(sorted(set(D.groups[rows_all])))
mem = {k: rows_all[D.groups[rows_all] == k] for k in keys}
pairs = [("C-DEC", "C-TOOL"), ("C-PD", "C-TOOL"), ("DEC", "TOOL")]
draws = {p: [] for p in pairs}
for _ in range(2000):
    r = np.concatenate([mem[k] for k in rng.choice(keys, len(keys))])
    for a, b in pairs:
        draws[(a, b)].append(m(a, r) - m(b, r))
res = {"paired_95ci_recall0.9": {f"{a}-{b}": {k: [round(float(np.nanpercentile(np.array(d)[:, i], 2.5)), 3),
                                                  round(float(np.nanpercentile(np.array(d)[:, i], 97.5)), 3)]
                                              for i, k in enumerate(("d_recall", "d_fpr_attacked_safe", "d_fpr_clean"))}
                                for (a, b), d in draws.items()}}
res["per_suite_recall0.9"] = {su: {"n_unsafe": int((keep & (D.suite == su) & uns).sum()),
                                   "n_attacked_safe": int((keep & (D.suite == su) & att & ~uns).sum()),
                                   "n_clean": int((keep & (D.suite == su) & ~att).sum()),
                                   **{n: [round(float(x), 3) for x in m(n, rows_all[D.suite[rows_all] == su])]
                                      for n in ("S1", "C-PRES28", "C-DEC", "C-PD")}}
                              for su in SUITES}
(o / "posthoc_r2.json").write_text(json.dumps({"protocol": __doc__, **res}, indent=1))
print(json.dumps(res, indent=0)[:2500])
