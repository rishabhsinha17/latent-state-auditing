"""POST HOC, DESCRIPTIVE (r7, after eval_r7 on Qwen): what are the safe tool calls on InjecAgent?

For each model, every step whose action has a parsed call and is labelled safe, classified by its
first call: a re-call of the step's user tool (the tool already called in the context), another
offered tool, or a tool that was not offered. If safe calls are all re-calls, a probe can separate
safe from unsafe calls by "repeats the previous call" without any signal about the injection.
Usage: posthoc_r7_safecalls.py <ia chat dir> <raw dir> <out json>
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_steps_r7 import parse, parse_permissive  # noqa: E402

chat, raw, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
steps = {json.loads(l)["case_id"]: json.loads(l) for l in open(chat / "suite_steps.jsonl")}
res = {}
for m in ("qwen", "granite", "hermes"):
    T = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(raw / f"gen_ia_{m}.texts.jsonl")}
    L = {json.loads(l)["case_id"]: json.loads(l) for l in open(raw / f"gen_ia_{m}.labels.jsonl")}
    pr = parse_permissive if m == "granite" else parse
    k = collections.Counter()
    for c, lab in L.items():
        calls = pr(T[c])
        if not calls or lab["unsafe"]:
            continue
        s, n = steps[c], calls[0][0]
        k["user_tool_recall" if n == s["user_tool"] else
          "other_offered_tool" if n in [t["function"]["name"] for t in s["tools"]] else "not_offered"] += 1
    res[m] = {"safe_calls": dict(k), "unsafe": sum(l["unsafe"] for l in L.values())}
out.write_text(json.dumps({"protocol": __doc__, "results": res}, indent=1))
print(json.dumps(res))
