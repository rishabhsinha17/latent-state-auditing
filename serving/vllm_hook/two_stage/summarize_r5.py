"""Print the eval_r5 tables (no computation beyond reading the JSON). Usage: summarize_r5.py out_r5/eval_r5_<tag>.json ..."""
import json
import sys

DETS = ("S1", "C-PRES/joint", "C-RULE/joint", "C-PD/gate97", "C-PTF/gate97", "C-PCALL/gate97", "C-PNAME/gate97",
        "C-PSPAN/gate97", "C-PPOS/gate97", "C-PT/gate97")
for path in sys.argv[1:]:
    r = json.load(open(path))
    print(f"\n### {r['tag']} layer {r['layer']}  n={r['n']}  parity={r['parity']['tf_vs_r4_decode_cos_median']}"
          f"/{r['parity']['prompt_vs_r4_cos_median']} comparable={r['parity']['comparable_to_r4']}")
    for pop in ("all_steps", "calls_only"):
        for s in ("A", "B", "C", "D"):
            sm = r[pop][s]["summary"]
            print(f"\n{pop} {s}: recall / fpr_AS / fpr_clean / fpr_AS_call")
            for d in DETS:
                v = sm[d]
                print(f"  {d:16s} " + " / ".join("n.a." if x != x else f"{x:.3f}" for x in
                                                  (v["recall"], v["fpr_attacked_safe"], v["fpr_clean"],
                                                   v["fpr_attacked_safe_call"])))
            for k, ci in r[pop][s]["paired_95ci"].items():
                print(f"    {k:34s} dR {ci['d_recall']} dAS {ci['d_fpr_attacked_safe']} dCl {ci['d_fpr_clean']}"
                      f" dASc {ci['d_fpr_attacked_safe_call']}")
