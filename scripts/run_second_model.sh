#!/bin/bash
# Second-model replication: exposure/commitment contrast phenomenon on a
# non-Qwen model. Activations only (heuristic NLA runner); layer 23 of 32
# (proportional to Qwen's 20 of 28).
set -u
LOG=/workspace/logs
mkdir -p "$LOG"
stage() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1" >> "$LOG/STATUS"; }

stage "setup:start"
cd /workspace
git clone --quiet <THIS_REPOSITORY_URL> lsa >> "$LOG/setup.log" 2>&1
cd /workspace/lsa
pip install -q -e ".[ml]" >> "$LOG/setup.log" 2>&1
pip install -q agentdojo hf_transfer >> "$LOG/setup.log" 2>&1
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ $? -eq 0 ]; then stage "setup:ok"; else stage "setup:FAIL"; fi

stage "download:start"
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
python3 - >> "$LOG/download.log" 2>&1 <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
for candidate in ("mistralai/Mistral-7B-Instruct-v0.3", "HuggingFaceH4/zephyr-7b-beta"):
    try:
        snapshot_download(candidate)
        print("USING_MODEL=" + candidate, flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"failed {candidate}: {e}", flush=True)
sys.exit(1)
PYEOF
if [ $? -ne 0 ]; then stage "download:FAIL"; exit 1; fi
MODEL=$(grep -o 'USING_MODEL=.*' "$LOG/download.log" | tail -1 | cut -d= -f2)
stage "download:ok:$MODEL"

stage "export2:start"
python3 -m latent_agent_auditing.experiments.export_agentdojo_model_trajectories \
  --output data/mc_ws_second_model.jsonl \
  --benchmark-version v1 --suites workspace \
  --attacks important_instructions,tool_knowledge \
  --user-task-limit 40 --injection-task-limit 6 \
  --model-name "$MODEL" --dtype bfloat16 --max-new-tokens 80 --temperature 0 \
  >> "$LOG/export2.log" 2>&1
if [ $? -eq 0 ]; then stage "export2:ok"; else stage "export2:FAIL"; exit 1; fi

python3 - >> "$LOG/counts2.log" 2>&1 <<'PYEOF'
import json
from collections import Counter
rows = [json.loads(l) for l in open("data/mc_ws_second_model.jsonl")]
c = Counter((r.get("attack_name") or "clean", bool(r.get("attack_success"))) for r in rows)
print("records:", len(rows))
for k, v in sorted(map(lambda kv: (str(kv[0]), kv[1]), c.items())):
    print(" ", k, v, flush=True)
PYEOF
stage "counts2:done"

stage "audit2:start"
python3 -m latent_agent_auditing.experiments.audit_benchmark_records \
  --input data/mc_ws_second_model.jsonl \
  --adapter agentdojo --run-dir runs/second_model \
  --activation-backend transformers --nla-runner heuristic \
  --model-name "$MODEL" \
  --dtype bfloat16 --layers 23 --n-nla-samples 1 --max-new-tokens 80 \
  >> "$LOG/audit2.log" 2>&1
if [ $? -eq 0 ]; then stage "audit2:ok"; else stage "audit2:FAIL"; fi
stage "SECOND_MODEL:done"
