"""POST HOC, DESCRIPTIVE (after eval_r4): how early does each model write its tool call?

The decision stage reads generated tokens 1..16 (roughly the first 80 characters). For each r4
dataset, the share of unsafe actions with a JSON "name" key within the first 80 characters of the
action, and anywhere in it.
Usage: posthoc_r4_callpos.py <gpu-20261108-r4/raw dir> <out json>
"""
import json, sys
from pathlib import Path
R, out = Path(sys.argv[1]), Path(sys.argv[2])
res = {}
for tag, lab in (("qwen_full", "qwen_full.labels.jsonl"), ("granite2_full", "granite2_full.labels_permissive.jsonl"),
                 ("hermes_full", "hermes_full.labels.jsonl")):
    T = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(R / f"{tag}.texts.jsonl")}
    L = {json.loads(l)["case_id"]: json.loads(l)["unsafe"] for l in open(R / lab)}
    uns = [c for c, u in L.items() if u]
    res[tag] = {"unsafe": len(uns), "call_within_80_chars": sum('"name"' in T[c][:80] for c in uns),
                "call_anywhere": sum('"name"' in T[c] for c in uns)}
out.write_text(json.dumps({"protocol": __doc__, "results": res}, indent=1)); print(json.dumps(res))
