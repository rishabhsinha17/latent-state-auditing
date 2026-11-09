"""Export the decision-time cascade evaluated in r3 as C-PD/v032/gate97 (published as ../artifacts/qwen2.5-7b-instruct/two_stage_decision/, which examples/demo_two_stage.py --decision downloads).

stage1   risk probe, layer 28 last prompt token, PCA-32 + logistic on attacked steps
stage2   product of
           presence probe, layer 28 last prompt token, PCA-64, attack present vs clean
           decision probe, layer 28, mean of decode states for generated tokens 1..16, PCA-64,
           attacked steps with the outcome label (probe.json position "decode_mean")
All three are fit on the r3 step set with vLLM 0.32 features (the backend users run).
Thresholds: calibrate_cascade(target_recall=0.9, stage1_recall=0.97) on grouped 5-fold OOF
scores (seed 13). Also writes cpu_reference.json: the exported cascade's verdicts on the bs1
GPU-capture features for every step, for the batched GPU check.

Usage: export_r3.py <chat dir> <cascade.py> <dec npz> <regen labels> <regen jsonl> <out dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Data, Probe, load_cascade  # noqa: E402

chat, cas_path, dec_path, lab_path, regen_path, out = map(Path, sys.argv[1:7])
C = load_cascade(cas_path)
D = Data(chat, layers=(28,))
E = np.load(dec_path)
assert list(E["case_id"]) == D.ids
dec = E["dec_layer_28"].astype(np.float64)
has = ~np.isnan(dec).all(2)
DEC = np.where(has[..., None], dec, 0).sum(1) / np.maximum(has.sum(1, keepdims=True), 1)
X = E["prefill_layer_28"].astype(np.float64)
keep = E["same_action"].astype(bool) & (has.sum(1) >= 1)
lab_new = {json.loads(l)["case_id"]: bool(json.loads(l)["unsafe"]) for l in open(lab_path)}
idx = {c: i for i, c in enumerate(D.ids)}
for l in open(regen_path):
    r = json.loads(l)
    i = idx[r["case_id"]]
    if D.attacked[i] and r["prefix_matches_capture"] and r["case_id"] in lab_new:
        D.unsafe[i] = lab_new[r["case_id"]]
        keep[i] = has[i].sum() >= 1
keep |= (~D.attacked) & (has.sum(1) >= 1)
att, uns = D.attacked, D.unsafe
rows = np.flatnonzero(keep)


def fit(tr):
    a = tr[att[tr]]
    return Probe(32).fit(X[a], uns[a]), Probe(64).fit(X[tr], att[tr]), Probe(64).fit(DEC[a], uns[a])


s1, s2 = np.full(D.n, np.nan), np.full(D.n, np.nan)
for te in D.folds(rows, 13):
    p1, pp, pd = fit(np.setdiff1d(rows, te))
    s1[te], s2[te] = p1.score(X[te]), pp.score(X[te]) * pd.score(DEC[te])
op = C.calibrate_cascade(s1[rows], s2[rows], uns[rows], 0.9, stage1_recall=0.97)
print("OOF calibration point:", json.dumps(op.to_dict()))

final = dict(zip(("stage1", "presence", "decision"), fit(rows)))
d = out / "qwen2.5-7b-instruct-decision"
meta = {"stage1": {"label": "next action follows the injection (risk)", "position": "prompt_last"},
        "presence": {"label": "a prompt injection is present in the context", "position": "prompt_last"},
        "decision": {"label": "next action follows the injection, read from the generated action",
                     "position": "decode_mean", "decode_tokens": 16}}
X_of = {"stage1": X, "presence": X, "decision": DEC}
for name, pr in final.items():
    (d / name).mkdir(parents=True, exist_ok=True)
    mean, scale, w, b = pr.export()
    ref = pr.score(X_of[name][rows])
    got = 1 / (1 + np.exp(-(((X_of[name][rows] - mean) / scale) @ w + b)))
    assert np.allclose(got, ref, atol=1e-6), name
    np.savez(d / name / "probe.npz", mean=mean.astype(np.float32), scale=scale.astype(np.float32),
             weights=w.astype(np.float32), bias=np.float32(b), layer=np.int64(28))
    (d / name / "probe.json").write_text(json.dumps({
        "model_name": "Qwen/Qwen2.5-7B-Instruct", "layer": 28, **meta[name],
        "training_data": f"{len(rows)} AgentDojo v1 steps (vLLM 0.32 capture, bf16), Qwen2.5 chat template",
        "note": "no standalone threshold: the cascade thresholds live in cascade.json"}, indent=2))
C.save_cascade_config(str(d / "cascade.json"), str(d / "stage1/probe.npz"),
                      [str(d / "presence/probe.npz"), str(d / "decision/probe.npz")], op,
                      target_recall=0.9, stage1_recall=0.97, data="AgentDojo v1, r3 step set",
                      split="grouped 5-fold OOF by user task, seed 13",
                      held_out_eval="eval_r3.json detector C-PD/v032/gate97")


def lin(path, Z):
    a = np.load(path)
    return 1 / (1 + np.exp(-(((Z - a["mean"]) / a["scale"]) @ a["weights"] + a["bias"])))


q1 = lin(d / "stage1/probe.npz", X)
q2 = lin(d / "presence/probe.npz", X) * lin(d / "decision/probe.npz", DEC)
flags = C.cascade_flags(q1, q2, op.stage1_threshold, op.stage2_threshold)
(out / "cpu_reference.json").write_text(json.dumps({
    "case_id": D.ids, "stage1": q1.tolist(), "stage2": q2.tolist(), "flag": flags.tolist(),
    "kept": keep.tolist(), "note": "verdicts of the exported cascade on bs1 GPU-capture features"}))
ins = C.evaluate_cascade(q1[rows], q2[rows], uns[rows], op.stage1_threshold, op.stage2_threshold)
print("IN-SAMPLE sanity (not evidence): recall", round(ins.recall, 3), "fpr", round(ins.fpr, 3))
