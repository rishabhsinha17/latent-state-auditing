"""Recompute per-suite/per-template export coverage from the released exports.

Writes EXP_suite_counts.json; verify_paper.py checks the Appendix compliance
table against it. Travel appears in two export files and is deduplicated by id.
Run from analysis/: python3 suite_counts.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent.parent / "pod-harvest" / "lsa" / "data_expanded"
FILES = [
    "mc_ws_important_full.jsonl",
    "mc_ws_templates_probe.jsonl",
    "mc_travel_important.jsonl",
    "mc_other_suites_full.jsonl",
]

seen = set()
cells = {}
for f in FILES:
    for line in open(DATA / f):
        r = json.loads(line)
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        suite = r.get("domain")
        atk = r.get("attack_name") or "clean"
        atk = "clean" if atk in ("clean", "none") else atk
        c = cells.setdefault((suite, atk), {"n": 0, "unsafe": 0})
        c["n"] += 1
        c["unsafe"] += bool(r.get("attack_success"))

out = {
    "unique_trajectories": len(seen),
    "attacked_total": sum(c["n"] for (s, a), c in cells.items() if a != "clean"),
    "clean_total": sum(c["n"] for (s, a), c in cells.items() if a == "clean"),
    "unsafe_total": sum(c["unsafe"] for c in cells.values()),
    "cells": {f"{s}:{a}": c for (s, a), c in sorted(cells.items())},
}
json.dump(out, open(HERE / "EXP_suite_counts.json", "w"), indent=1)
print(json.dumps(out, indent=1))
