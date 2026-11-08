"""Print eval_r6a / eval_r6b tables from their JSON (no computation). Usage: summarize_r6.py <json> ..."""
import json
import sys


def row(name, v):
    f = lambda x: "n.a." if x is None or x != x else f"{x:.3f}"
    return f"  {name:22s} " + " / ".join(f(v[k]) for k in ("recall", "fpr_attacked_safe", "fpr_clean", "fpr_attacked_safe_call"))


def block(title, rep):
    print(f"\n{title}: recall / fpr_AS / fpr_clean / fpr_AS_call")
    for d, v in rep["summary"].items():
        print(row(d, v))
    for k, ci in rep["paired_95ci"].items():
        print(f"    {k:42s} dR {ci['d_recall']} dAS {ci['d_fpr_attacked_safe']} dASc {ci['d_fpr_attacked_safe_call']}")
    for extra in ("recall_seen_vs_unseen_unsafe_tool", "recall_strict_vs_follows_only", "unsafe_with_tool_seen_unsafe_in_training"):
        if extra in rep:
            print(f"    {extra}: {json.dumps(rep[extra])}")


for path in sys.argv[1:]:
    r = json.load(open(path))
    print(f"\n### {path}  tag={r['tag']}")
    for k in ("n", "parity", "n_old", "n_new"):
        if k in r:
            print(k, json.dumps(r[k]))
    for top in ("all_steps", "calls_only"):
        if top in r:
            for s, rep in r[top].items():
                block(f"{top} {s}", rep)
    for part in ("old", "new"):
        if part in r:
            for pop, v in r[part].items():
                if "summary" in v:
                    block(f"{part} {pop}", v)
                else:
                    for s, rep in v.items():
                        block(f"{part} {pop} {s}", rep)
