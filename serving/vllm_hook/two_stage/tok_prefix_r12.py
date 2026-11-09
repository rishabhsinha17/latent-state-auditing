"""r12 (CPU, tokenizers only): is carrying the presence score forward the same as scoring one prompt
at several positions?

For every attacked step after arrival (depth d), and every earlier depth j = 1 .. d-1 of the same
trajectory, render the step's own messages cut after j calls, with the generation prompt, exactly as
gen_capture_r4.py renders a step (Granite through granite_messages). If those token ids are an exact
prefix of the full step's ids, then in a causal model the hidden state at the last of them, inside
the full prompt, is the prompt state of the earlier step, so the maximum presence score over the
earlier steps can be read from one forward pass at those positions (no per-task state at the caller).
Also reports how often the cut messages equal the stored earlier step (cd / v1 / r11 / r12 files),
exactly and with AgentDojo's wall-clock timestamps masked.
Usage: python -I tok_prefix_r12.py <model id> <adapter none|granite> <out json> <chat dir> ...
"""
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_capture_r4 import granite_messages  # noqa: E402

model, adapter, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
dirs = [Path(d) for d in sys.argv[4:]]
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
render = granite_messages if adapter == "granite" else (lambda m: m)
TS = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\.\d+")
steps = {}
for d in dirs:
    for line in open(d / "suite_steps.jsonl"):
        s = json.loads(line)
        steps[s["case_id"]] = s


def ids_of(messages, tools):
    return tok.apply_chat_template(render(messages), tools=tools, add_generation_prompt=True, tokenize=True,
                                   return_dict=False)


def stored(case, j):
    """The stored step of the same trajectory at depth j, if any."""
    s = steps[case]
    if s.get("injection_depth") is None:
        return None
    base = case.rsplit("_persist", 1)[0]
    k = s["injection_depth"]
    if j == k:
        return steps.get(base)
    if j > k:
        return steps.get(f"{base}_persist{j}")
    return steps.get(f"{s['suite']}_{s['user_task']}_cleandepth{j}")


dump = lambda m: json.dumps(m, sort_keys=True, default=str)
n_pairs = n_prefix = n_same = n_same_ts = n_stored = 0
fails = []
after = [c for c, s in steps.items() if s.get("injection_depth") is not None and s["depth"] > s["injection_depth"]]
for c in after:
    s = steps[c]
    full = ids_of(s["messages"], s["tools"])
    for j in range(1, s["depth"]):
        cut = s["messages"][: 2 + 2 * j]
        part = ids_of(cut, s["tools"])
        n_pairs += 1
        ok = len(part) < len(full) and full[: len(part)] == part
        n_prefix += ok
        if not ok and len(fails) < 5:
            k = next((i for i, (x, y) in enumerate(zip(part, full)) if x != y), min(len(part), len(full)))
            fails.append({"case": c, "j": j, "first_diff": k, "part": tok.decode(part[max(0, k - 8):k + 8]),
                          "full": tok.decode(full[max(0, k - 8):k + 8])})
        ref = stored(c, j)
        if ref is not None:
            n_stored += 1
            n_same += dump(ref["messages"]) == dump(cut)
            n_same_ts += TS.sub("<ts>", dump(ref["messages"])) == TS.sub("<ts>", dump(cut))
rep = {"model": model, "after_arrival_steps": len(after), "pairs": n_pairs, "prefix": n_prefix,
       "stored_earlier_steps_compared": n_stored, "stored_equal": n_same, "stored_equal_timestamps_masked": n_same_ts,
       "fails": fails}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items() if k != "fails"}), "fails shown:", len(fails))
