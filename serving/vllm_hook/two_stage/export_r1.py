"""Export a local two-stage cascade for Qwen2.5-7B-Instruct (NOT published).

stage1  the released tool-call risk probe (../export/probe.npz, copied unchanged)
stage2  injection-presence probe on layer 28 (PCA-64 + logistic, attack present vs clean,
        fit on all 1826 steps; no outcome label is used)
Thresholds: calibrate_cascade at target recall 0.9 on grouped 5-fold OOF scores (seed 13)
of the same two recipes, written with save_cascade_config.

Checks: the folded stage-2 export reproduces the fitted probe; the released stage-1 artifact
matches this recipe refit on all attacked steps; IN-SAMPLE operating point printed only as a
sanity check (the held-out numbers are in eval_r1_posthoc.json, detector C-PRES28).

Usage: export_r1.py <chat dir> <cascade.py> <released export dir> <out dir>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Data, Probe, load_cascade  # noqa: E402

chat, cas_path, rel, out = map(Path, sys.argv[1:5])
C = load_cascade(cas_path)
D = Data(chat, layers=(28,))
X, att, uns = D.X[28], D.attacked, D.unsafe

s1, s2 = np.full(D.n, np.nan), np.full(D.n, np.nan)
for te in D.folds(np.arange(D.n), 13):
    tr = np.setdiff1d(np.arange(D.n), te)
    a = tr[att[tr]]
    s1[te] = Probe(32).fit(X[a], uns[a]).score(X[te])
    s2[te] = Probe(64).fit(X[tr], att[tr]).score(X[te])
op = C.calibrate_cascade(s1, s2, uns, 0.9)
print("OOF calibration point:", json.dumps(op.to_dict()))

pres = Probe(64).fit(X, att)
mean, scale, w, b = pres.export()
ref = pres.score(X)
got = 1 / (1 + np.exp(-(((X - mean) / scale) @ w + b)))
assert np.allclose(got, ref, atol=1e-6), "stage-2 export does not reproduce the fitted probe"

d = out / "qwen2.5-7b-instruct"
(d / "stage1").mkdir(parents=True, exist_ok=True)
(d / "stage2").mkdir(parents=True, exist_ok=True)
for name in ("probe.npz", "probe.json"):
    shutil.copy(rel / name, d / "stage1" / name)
np.savez(d / "stage2" / "probe.npz", mean=mean.astype(np.float32), scale=scale.astype(np.float32),
         weights=w.astype(np.float32), bias=np.float32(b), layer=np.int64(28))
(d / "stage2" / "probe.json").write_text(json.dumps({
    "model_name": "Qwen/Qwen2.5-7B-Instruct", "layer": 28, "threshold": round(op.stage2_threshold, 6),
    "probe": "pca64", "label": "a prompt injection is present in the context (attacked step vs its clean twin)",
    "training_data": "1826 AgentDojo v1 agent steps (1729 attacked, 97 clean), Qwen2.5 chat template, vLLM-Hook capture, bf16",
    "note": "threshold is the cascade stage-2 threshold from cascade.json, not a standalone operating point"}, indent=2))

# Released stage 1 vs this recipe refit on all attacked steps (SVD vs Gram-eigh PCA).
r = np.load(rel / "probe.npz")
p_rel = 1 / (1 + np.exp(-(((X - r["mean"]) / r["scale"]) @ r["weights"] + r["bias"])))
p_new = Probe(32).fit(X[att], uns[att]).score(X)
print("released stage1 vs refit: max |dp| =", round(float(np.abs(p_rel - p_new).max()), 5))

C.save_cascade_config(str(d / "cascade.json"), str(d / "stage1/probe.npz"), str(d / "stage2/probe.npz"), op,
                      target_recall=0.9, data="AgentDojo v1 suite_full (1826 steps)",
                      split="grouped 5-fold OOF by user task, seed 13",
                      held_out_eval="eval_r1_posthoc.json detector C-PRES28")
ins = C.evaluate_cascade(p_rel, ref, uns, op.stage1_threshold, op.stage2_threshold)
f = C.cascade_flags(p_rel, ref, op.stage1_threshold, op.stage2_threshold)
print("IN-SAMPLE sanity (not evidence): recall", round(ins.recall, 3),
      "fpr attacked-safe", round(float(f[att & ~uns].mean()), 3), "fpr clean", round(float(f[~att].mean()), 3))
np.save(out / "insample_flags.npy", f)
