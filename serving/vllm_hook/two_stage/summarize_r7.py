"""Print the eval_r7 scaling tables (reads the JSON only). Usage: summarize_r7.py out_r7/eval_r7_<tag>.json ..."""
import json
import sys

KS = ("3", "6", "12", "24", "max")
for path in sys.argv[1:]:
    r = json.load(open(path))
    print(f"\n### {r['tag']}  n={json.dumps(r['n'])}  underpowered={r['underpowered']}")
    for pop in ("calls_only", "all_steps"):
        p = r[pop]
        print(f"\n{pop}  (skipped cells {p['skipped_cells']} of {p['cells']})   recall / FPR attacked-safe call / coverage")
        for d in ("S1", "C-DEC", "C-SPAN", "C-TOOL", "C-PROV", "C-RULE"):
            cells = [p["summary"][f"{d}@{k}"] for k in KS]
            print(f"  {d:7s} " + "  ".join(f"k={k}: {c['recall']:.3f}/{c['fpr_attacked_safe_call']:.3f}/{c['coverage']:.2f}" for k, c in zip(KS, cells)))
        ref = p["seen_goal_reference"]
        print("  seen-goal reference: " + "  ".join(f"{d} {ref[d]['recall']:.3f}/{ref[d]['fpr_attacked_safe_call']:.3f}" for d in ref))
        for k, ci in p["paired_95ci"].items():
            print(f"    {k:22s} dR {ci['d_recall']} dASc {ci['d_fpr_attacked_safe_call']}")
        for kd, v in p["by_kind_at_max"].items():
            print(f"  {kd} at max: " + "  ".join(f"{d} {v[d+'@max']['recall']:.3f}/{v[d+'@max']['fpr_attacked_safe_call']:.3f}" for d in ("S1", "C-DEC", "C-SPAN", "C-PROV")))
    if "agentdojo_D_calls_only" in r:
        a = r["agentdojo_D_calls_only"]
        print("\nAgentDojo split D, calls only:")
        for d, v in a["summary"].items():
            print(f"  {d:20s} {v['recall']:.3f} / {v['fpr_attacked_safe']:.3f}")
        for k, ci in a["paired_95ci"].items():
            print(f"    {k:40s} dR {ci['d_recall']} dAS {ci['d_fpr_attacked_safe']}")
