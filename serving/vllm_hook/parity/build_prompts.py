"""Rebuild the exact prompts behind the stored HF activations (CPU, no model).

Replays audit_benchmark_records.py's case/trajectory construction and the
pipeline's critical-point filter, resolves each audit point's prompt with the
same resolve_activation_prompt the HF backend used, and checks it against the
stored prompt_hash. Writes prompts.jsonl (one row per stored activation) and
hf_layer20.npy (the stored vectors, same row order).

Usage: python build_prompts.py <latent-state-auditing repo> <run name> <out dir>
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

repo, run, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(repo))

from latent_agent_auditing.auditing.activation_capture import _hash_text, resolve_activation_prompt
from latent_agent_auditing.auditing.decision_points import DEFAULT_CRITICAL_POINTS, is_security_critical
from latent_agent_auditing.experiments.audit_benchmark_records import ADAPTERS, _read_records, _trajectory_from_record

run_dir = repo / "runs" / run
manifest = json.loads((run_dir / "run_manifest.json").read_text())
records = _read_records(repo / manifest["input_path"])
cases = ADAPTERS[manifest["adapter_name"]]().from_records(records)

prompts = {}
for case, record in zip(cases, records):
    traj = _trajectory_from_record(case, record)
    for event in traj.events:
        if not is_security_critical(event, DEFAULT_CRITICAL_POINTS):
            continue
        prompt, source = resolve_activation_prompt(case, event)
        prompts[(case.id, event.step, event.decision_point.value)] = (prompt, source)

with gzip.open(run_dir / "audits.jsonl.gz", "rt") as f:
    audits = [json.loads(l) for l in f if l.strip()]

rows, vecs, mismatch = [], [], 0
for a in audits:
    act = a["activation"]
    key = (act["case_id"], act["step"], act["decision_point"])
    prompt, source = prompts[key]
    ok = _hash_text(prompt) == act["metadata"]["prompt_hash"]
    mismatch += not ok
    case = next(c for c in cases if c.id == act["case_id"])
    rows.append({
        "case_id": act["case_id"], "step": act["step"], "decision_point": act["decision_point"],
        "prompt_source": source, "hash_ok": ok, "n_tokens_hf": act["metadata"]["n_tokens"],
        "attack_present": case.attack_present, "attack_success": a["metadata"]["attack_success"],
        "prompt": prompt,
    })
    vecs.append(act["vector"])

out.mkdir(parents=True, exist_ok=True)
with open(out / "prompts.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
np.save(out / "hf_layer20.npy", np.asarray(vecs, dtype=np.float32))
(out / "meta.json").write_text(json.dumps({
    "run": run, "model_name": manifest["model_name"], "dtype": manifest["dtype"],
    "max_length": manifest["max_length"], "hf_layer": manifest["layers"][0],
    "hidden_state_index": audits[0]["activation"]["metadata"]["hidden_state_index"],
}, indent=2))
trunc = sum(r["n_tokens_hf"] >= manifest["max_length"] for r in rows)
print(f"rows {len(rows)}  hash mismatches {mismatch}  truncated at max_length {trunc}")
print("by source", {s: sum(r["prompt_source"] == s for r in rows) for s in {r["prompt_source"] for r in rows}})
