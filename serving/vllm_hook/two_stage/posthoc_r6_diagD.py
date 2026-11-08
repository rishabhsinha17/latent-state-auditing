"""POST HOC, DIAGNOSTIC (r6, after eval_r5): why do decision stages lose recall on held-out injection tasks?

Reruns eval_r5's split D (5 folds grouped by injection task, seeds 13, 0, 1, 2, same recipes and
calibration) for S1, C-PTF, C-PSPAN, C-PNAME and C-PT, on all steps and on calls only, and keeps
every test step's flag. For each unsafe test step it records the called tool and whether that
tool was the tool of any unsafe training step in the same fold ("seen unsafe tool"). Recall is
then reported separately for seen and unseen unsafe tools, and per injection task.
Usage: posthoc_r6_diagD.py <chat dir> <cascade.py> <r4 npz> <r5 npz> <texts> <labels> <layer> <tag> <out dir>
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FOLDS, Data, Probe, load_cascade, logreg_fit  # noqa: E402

chat, cas_path, r4_path, r5_path, texts_path, lab_path = map(Path, sys.argv[1:7])
L, TAG, out = int(sys.argv[7]), sys.argv[8], Path(sys.argv[9])
C = load_cascade(cas_path)
gen = [json.loads(l) for l in open(chat / "suite_full.jsonl")]
ids = [g["case_id"] for g in gen]
att = np.array([g["attack_present"] for g in gen])
lab = {json.loads(l)["case_id"]: bool(json.loads(l)["unsafe"]) for l in open(lab_path)}
uns = np.array([lab.get(c, False) and a for c, a in zip(ids, att)])
E, F = np.load(r4_path), np.load(r5_path)
X = E[f"prefill_layer_{L}"].astype(np.float64)
dec = E[f"dec_layer_{L}"].astype(np.float64)
has = ~np.isnan(dec).all(2)
tf = F[f"fixed_layer_{L}"].astype(np.float64)
tfh = ~np.isnan(tf).all(2)
TF = np.where(tfh[..., None], np.nan_to_num(tf), 0).sum(1) / np.maximum(tfh.sum(1, keepdims=True), 1)
WIN = {"TF": TF, "SPAN": F[f"span_layer_{L}"].astype(np.float64), "NAME": F[f"name_layer_{L}"].astype(np.float64)}
HC = F["call_tok"] >= 0
texts = {json.loads(l)["case_id"]: json.loads(l) for l in open(texts_path)}
keep = ~np.isnan(X).any(1) & (has.sum(1) >= 1) & np.array([not texts[c].get("skipped_too_long") for c in ids])
name_re = re.compile(r'"name"\s*:\s*"([^"]+)"')
tool = np.array([(m.group(1) if (m := name_re.search(texts[c]["text"][:600])) else
                  ("<partial>" if "tool_call" in texts[c]["text"][:600] else "<text>")) for c in ids])
groups = np.array([Data.group_key(c) for c in ids])
injg = np.array([(c.split("_user_task")[0] + ":" + re.search(r"injection_task_\d+", c).group(0)) if a
                 else "clean:" + g for c, a, g in zip(ids, att, groups)])
SEEDS = (13, 0, 1, 2)


def gfolds(rows, seed, g):
    keys = sorted(set(g[rows]))
    random.Random(seed).shuffle(keys)
    return [rows[np.isin(g[rows], keys[i::FOLDS])] for i in range(FOLDS)]


class OneHot:
    def fit(self, names, y):
        self.vocab = {v: i for i, v in enumerate(sorted(set(names)))}
        self.m = logreg_fit(self._x(names), np.asarray(y, float))
        return self

    def _x(self, names):
        Z = np.zeros((len(names), len(self.vocab)))
        for r, v in enumerate(names):
            if v in self.vocab:
                Z[r, self.vocab[v]] = 1.0
        return Z

    def score(self, names):
        mu, sd, w, b = self.m
        return 1.0 / (1.0 + np.exp(-(((self._x(names) - mu) / sd) @ w + b)))


def fit_score(tr, te):
    a = tr[att[tr]]
    pres = Probe(64).fit(X[tr], att[tr]).score(X[te])
    s = {"S1": Probe(32).fit(X[a], uns[a]).score(X[te]), "PT": pres * OneHot().fit(tool[a], uns[a]).score(tool[te])}
    for k, Z in WIN.items():
        s["P" + k] = pres * Probe(64).fit(Z[a], uns[a]).score(Z[te])
    return s


DETS = (("S1", None), ("C-PTF/gate97", "PTF"), ("C-PSPAN/gate97", "PSPAN"), ("C-PNAME/gate97", "PNAME"),
        ("C-PT/gate97", "PT"))


def run_split(tr, te, inner_seed):
    oof = {}
    for f in gfolds(tr, inner_seed, groups):
        for k, v in fit_score(np.setdiff1d(tr, f), f).items():
            oof.setdefault(k, np.full(len(ids), np.nan))[f] = v
    test = fit_score(tr, te)
    y = uns[tr]
    res = {}
    for name, k2 in DETS:
        if k2 is None:
            res[name] = test["S1"] >= C.threshold_for_recall(oof["S1"][tr][y], 0.9)
        else:
            op = C.calibrate_cascade(oof["S1"][tr], oof[k2][tr], y, 0.9, stage1_recall=0.97)
            res[name] = C.cascade_flags(test["S1"], test[k2], op.stage1_threshold, op.stage2_threshold)
    return res


report = {"protocol": __doc__, "tag": TAG}
for pop, rows in (("all_steps", np.flatnonzero(keep)), ("calls_only", np.flatnonzero(keep & HC))):
    flag = {n: np.zeros(len(ids)) for n, _ in DETS}
    seen = np.zeros(len(ids))
    for seed in SEEDS:
        for te in gfolds(rows, seed, injg):
            tr = np.setdiff1d(rows, te)
            r = run_split(tr, te, seed + 1000)
            for n in flag:
                flag[n][te] += r[n] / len(SEEDS)
            seen_tools = set(tool[tr][uns[tr]])
            seen[te] += np.isin(tool[te], list(seen_tools)) / len(SEEDS)
        print(TAG, pop, "seed", seed, flush=True)
    u = rows[uns[rows]]
    grp = {"seen_unsafe_tool": u[seen[u] >= 0.5], "unseen_unsafe_tool": u[seen[u] < 0.5]}
    rep = {k: {"n": int(v.size), **{n: round(float(flag[n][v].mean()), 3) if v.size else None for n, _ in DETS}}
           for k, v in grp.items()}
    per_task = defaultdict(list)
    for i in u:
        per_task[injg[i]].append(i)
    rep["per_injection_task"] = {k: {"n_unsafe": len(v), "tools": sorted(set(tool[v])),
                                     "seen_fraction": round(float(seen[v].mean()), 2),
                                     **{n: round(float(flag[n][v].mean()), 3) for n, _ in DETS}}
                                 for k, v in sorted(per_task.items())}
    report[pop] = rep
    print(TAG, pop, json.dumps({k: rep[k] for k in grp}), flush=True)
out.mkdir(parents=True, exist_ok=True)
(out / f"posthoc_r6_diagD_{TAG}.json").write_text(json.dumps(report, indent=1))
