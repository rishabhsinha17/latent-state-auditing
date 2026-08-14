#!/bin/bash
# Expanded FLMSec data run: exports + primary audit, staged with STATUS logging.
set -u
LOG=/workspace/logs
mkdir -p "$LOG" /workspace/lsa/data/expanded
cd /workspace/lsa
export HF_HOME=/workspace/hf
stage() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1" >> "$LOG/STATUS"; }

stage "prefetch:start"
python3 - >> "$LOG/prefetch.log" 2>&1 <<'PYEOF'
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen2.5-7B-Instruct", "kitft/nla-qwen2.5-7b-L20-av"):
    snapshot_download(m)
    print("done", m, flush=True)
PYEOF
if [ $? -eq 0 ]; then stage "prefetch:ok"; else stage "prefetch:FAIL"; fi

run_export() {
  local name=$1 out=$2 suites=$3 attacks=$4 ulim=$5 ilim=$6
  stage "$name:start"
  python3 -m latent_agent_auditing.experiments.export_agentdojo_model_trajectories \
    --output "$out" \
    --benchmark-version v1 --suites "$suites" --attacks "$attacks" \
    --user-task-limit "$ulim" --injection-task-limit "$ilim" \
    --model-name Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \
    --max-new-tokens 80 --temperature 0 \
    >> "$LOG/$name.log" 2>&1
  if [ $? -eq 0 ]; then stage "$name:ok"; else stage "$name:FAIL"; fi
}

run_export exportA data/expanded/mc_ws_important_full.jsonl workspace important_instructions 40 6
run_export exportB data/expanded/mc_ws_templates_probe.jsonl workspace tool_knowledge,injecagent,ignore_previous,system_message 20 2
run_export exportC data/expanded/mc_other_suites_full.jsonl travel,banking,slack important_instructions 40 9

stage "counts:start"
python3 - >> "$LOG/counts.log" 2>&1 <<'PYEOF'
import json
from collections import Counter
for f in ("mc_ws_important_full", "mc_ws_templates_probe", "mc_other_suites_full"):
    p = f"data/expanded/{f}.jsonl"
    try:
        rows = [json.loads(l) for l in open(p)]
    except FileNotFoundError:
        print(f, "MISSING", flush=True); continue
    print("=" * 10, f, "records:", len(rows), flush=True)
    if rows:
        print("keys:", sorted(rows[0].keys()), flush=True)
    c = Counter()
    for r in rows:
        env = r.get("environment") or r.get("suite") or "?"
        atk = r.get("attack_name") or "clean"
        succ = r.get("attack_success")
        c[(env, atk, succ)] += 1
    for k, v in sorted(map(lambda kv: (str(kv[0]), kv[1]), c.items())):
        print("  ", k, v, flush=True)
PYEOF
stage "counts:done"

stage "auditPRIMARY:start"
python3 -m latent_agent_auditing.experiments.audit_benchmark_records \
  --input data/expanded/mc_ws_important_full.jsonl \
  --adapter agentdojo --run-dir runs/exp_ws_full \
  --activation-backend transformers --nla-runner local-av \
  --model-name Qwen/Qwen2.5-7B-Instruct --av-model-name kitft/nla-qwen2.5-7b-L20-av \
  --dtype bfloat16 --layers 20 --n-nla-samples 1 --max-new-tokens 80 \
  >> "$LOG/audit_primary.log" 2>&1
if [ $? -eq 0 ]; then stage "auditPRIMARY:ok"; else stage "auditPRIMARY:FAIL"; fi

stage "summarize:start"
python3 -m latent_agent_auditing.experiments.summarize_paper_evidence \
  --run-dir runs/exp_ws_full --folds 5 --bootstrap-resamples 1000 \
  >> "$LOG/summarize.log" 2>&1
if [ $? -eq 0 ]; then stage "summarize:ok"; else stage "summarize:FAIL"; fi

stage "transfer:start"
python3 -m latent_agent_auditing.experiments.evaluate_benchmark_transfer \
  --run-dir runs/exp_ws_full >> "$LOG/transfer.log" 2>&1
if [ $? -eq 0 ]; then stage "transfer:ok"; else stage "transfer:FAIL"; fi

stage "ALL:done"
