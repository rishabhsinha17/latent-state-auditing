"""POST HOC, DESCRIPTIVE (r8, after eval_r8 on Qwen): is the attention share only a length effect?

Among attacked AgentDojo steps with a tool call, the AUROC (unsafe vs safe) of the pre-registered ATT
score (mean over the fixed-depth heads of tool / (tool + user) at Q1, else Q2), of a length-only score
n_tool / (n_tool + n_user) tokens (the share a head would give if it attended uniformly), and of ATT
divided by that length share. Not used by any pre-registered decision.
Usage: posthoc_r8_length.py <ad chat> <labels jsonl> <attn npz> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    r = np.argsort(np.argsort(s)) + 1.0
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=r) / cnt)[inv]
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum()))


chat, lab_p, att_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
Z = np.load(att_p)
lab = {json.loads(l)["case_id"]: json.loads(l)["unsafe"] for l in open(lab_p)}
attk = {json.loads(l)["case_id"]: json.loads(l)["attack_present"] for l in open(chat / "suite_full.jsonl")}
ids = list(Z["case_id"])
m = np.array([attk[c] for c in ids])
y = np.array([bool(lab.get(c)) for c in ids])[m]
heads = [tuple(h) for h in Z["heads"]]
fixed = sorted({h[0] for h in heads if sum(1 for x in heads if x[0] == h[0]) >= 28})
cols = [i for i, h in enumerate(heads) if h[0] in fixed]
f = Z["tool"][:, :, cols] / np.maximum(Z["tool"][:, :, cols] + Z["user"][:, :, cols], 1e-12)
with np.errstate(all="ignore"):
    s = np.nanmean(f[:, 0, :], 1)
    s = np.where(np.isnan(s), np.nanmean(f[:, 1, :], 1), s)
    s = np.nan_to_num(s, nan=0.0)                       # as eval_r8: no usable query -> 0
ln = Z["n_tool_tok"] / (Z["n_tool_tok"] + Z["n_user_tok"])
res = {"attacked_calls": int(m.sum()), "unsafe": int(y.sum()), "auroc_ATT": round(auroc(s[m], y), 3),
       "auroc_length_share": round(auroc(ln[m], y), 3), "auroc_ATT_over_length_share": round(auroc((s / ln)[m], y), 3),
       "corr_ATT_length_share": round(float(np.corrcoef(s[m], ln[m])[0, 1]), 3)}
out.write_text(json.dumps({"protocol": __doc__, "results": res}, indent=1))
print(json.dumps(res))
