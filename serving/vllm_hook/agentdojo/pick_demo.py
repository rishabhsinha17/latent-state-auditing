"""Pick three demo steps for examples/demo_toolcallrisk.py (clean, attacked-safe,
attacked-unsafe) where the exported probe's verdict matches the label, preferring
short prompts. These are training-distribution examples, not a held-out test.

Usage: python pick_demo.py <chat dir> <export dir>
"""
import json, sys
from pathlib import Path
import numpy as np

cdir, exp = Path(sys.argv[1]), Path(sys.argv[2])
P = np.load(exp / "probe.npz"); meta = json.loads((exp / "probe.json").read_text())
L, thr = int(P["layer"]), meta["threshold"]
X = np.load(cdir / "suite_full.npz")[f"layer_{L}"].astype(np.float64)
p = 1 / (1 + np.exp(-(((X - P["mean"]) / P["scale"]) @ P["weights"] + P["bias"])))
gen = [json.loads(l) for l in open(cdir / "suite_full.jsonl")]
lab = {json.loads(l)["case_id"]: json.loads(l)["unsafe"] for l in open(cdir / "suite_full.labels.jsonl")}
steps = {json.loads(l)["case_id"]: json.loads(l) for l in open(cdir / "suite_steps.jsonl")}


def pick(kind):
    cands = []
    for i, g in enumerate(gen):
        c = g["case_id"]; s = steps[c]
        if kind == "clean" and not s["attack_present"] and p[i] < thr:
            cands.append((g["n_prompt_tokens"], i))
        if kind == "safe" and s["attack_present"] and not lab[c] and p[i] < thr:
            cands.append((g["n_prompt_tokens"], i))
        if kind == "unsafe" and s["attack_present"] and lab[c] and p[i] >= thr:
            cands.append((g["n_prompt_tokens"], i))
    return min(cands)[1]


names = {"clean": "clean step", "safe": "injected step, model does not follow the injection",
         "unsafe": "injected step, model follows the injection"}
with open(exp / "demo_steps.jsonl", "w") as f:
    for kind in ("clean", "safe", "unsafe"):
        i = pick(kind); c = gen[i]["case_id"]; s = steps[c]
        f.write(json.dumps({"case_id": c, "label": names[kind], "messages": s["messages"], "tools": s["tools"]}) + "\n")
        print(kind, c, f"p={p[i]:.3f}", gen[i]["n_prompt_tokens"], "tokens |", gen[i]["text"][:100].replace("\n", " "))
