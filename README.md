# Latent Agent Auditing

This repo implements the dev scaffold for **latent-state auditing for tool-using language agents**:

```text
existing agent eval trace
  -> security-critical decision point hook
  -> activation capture
  -> NLA explanation
  -> latent risk labels
  -> cheap detector/probe
  -> tool-gate evaluation
```

The implementation is intentionally a thin augmentation layer over existing eval patterns such as AgentDojo, InjecAgent, ToolHijacker, ToolEmu, CaMeL-style flow control, CoT monitors, and AI-control monitor evals. It does not claim that NLAs are ground truth or that they should be used directly as a production firewall.

## What Runs Locally

The default path is fully offline and deterministic:

```bash
python -m latent_agent_auditing.experiments.run_injecagent_latent_audit --cases-per-domain 12
```

It creates an InjecAgent-style email/file benchmark slice, runs a vulnerable mock agent, captures mock residual-stream vectors at critical points, produces NLA-like explanations through a deterministic backend, labels latent risks, trains detector baselines, and writes JSONL artifacts under `runs/`.

This gives a complete end-to-end harness without requiring Qwen/Gemma checkpoints, Anthropic's released NLA checkpoints, or GPU access. Real model and NLA integrations plug into `ActivationBackend` and `NLARunner`.

## Main Modules

```text
latent_agent_auditing/
  adapters/        adapters for InjecAgent/AgentDojo/ToolHijacker-style records
  agent/           deterministic mock tool-using agent
  auditing/        decision hooks, activation capture, NLA runner, audit pipeline
  detectors/       transcript, tool-policy, NLA-text, activation-probe, hybrid baselines
  eval/            AUROC, AUPRC, FPR@TPR, lead-time, prevention metrics
  labels/          grounded latent-risk taxonomy and keyword labeler
  logging/         JSONL writer
  models/          dataclass schemas for cases, traces, activations, audits
  tools/           in-memory email/file/database/browser/shell tool surface
```

## Real NLA Integration Point

Use the mock path for CI and local development. To swap in actual NLA/model plumbing:

1. Implement or configure a concrete `ActivationBackend` in `latent_agent_auditing/auditing/activation_capture.py`.
2. Implement or configure a concrete `NLARunner` in `latent_agent_auditing/auditing/nla_runner.py`.
3. Keep the rest of the pipeline unchanged: adapters, decision points, labels, detectors, and metrics do not need to know whether activations came from a real model or the deterministic mock backend.

## Real Activation Smoke Test

On a GPU machine with PyTorch and Transformers installed:

```bash
python -m latent_agent_auditing.experiments.capture_transformer_smoke \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --run-dir runs/qwen_7b_smoke \
  --cache-dir /workspace/hf-cache \
  --dtype bfloat16 \
  --max-length 1024
```

This captures final-token hidden states at quartile layers for the
`BEFORE_TOOL_SELECTION` decision point and writes:

```text
runs/qwen_7b_smoke/
  activation_records.jsonl
  case.jsonl
  trajectory.jsonl
  summary.json
  activations/*.json
```

To verbalize the saved layer-20 Qwen activation with the released NLA AV:

```bash
python -m latent_agent_auditing.experiments.verbalize_saved_activation \
  --activation-records runs/qwen_7b_smoke/activation_records.jsonl \
  --output runs/qwen_7b_smoke/nla_layer20.jsonl \
  --av-model-name kitft/nla-qwen2.5-7b-L20-av \
  --layer 20 \
  --cache-dir /workspace/hf-cache \
  --dtype bfloat16 \
  --n-samples 1
```

This local Transformers runner is meant for correctness checks and tiny
batches. For throughput, use the released repo's SGLang serving path.

For the first 40-case batch:

```bash
python -m latent_agent_auditing.experiments.run_qwen_nla_batch \
  --cases-per-domain 20 \
  --run-dir runs/qwen_7b_nla_batch_40 \
  --cache-dir /workspace/hf-cache \
  --n-samples 1 \
  --max-new-tokens 160
```

To re-score saved generations after editing the taxonomy:

```bash
python -m latent_agent_auditing.experiments.relabel_nla_audits \
  --input runs/qwen_7b_nla_batch_40/nla_audits.jsonl \
  --output runs/qwen_7b_nla_batch_40/nla_audits_relabel.jsonl \
  --metrics runs/qwen_7b_nla_batch_40/metrics_relabel.json
```

To run multi-sample NLA on saved activations without recapturing Qwen
activations:

```bash
python -m latent_agent_auditing.experiments.verbalize_activation_batch \
  --activation-records runs/qwen_7b_nla_batch_40/activation_records.jsonl \
  --cases runs/qwen_7b_nla_batch_40/cases.jsonl \
  --trajectories runs/qwen_7b_nla_batch_40/trajectories.jsonl \
  --output-dir runs/qwen_7b_nla_batch_40_n3 \
  --cache-dir /workspace/hf-cache \
  --n-samples 3 \
  --max-new-tokens 160
```

To compare a trained NLA-text classifier against a raw activation probe:

```bash
python -m latent_agent_auditing.experiments.compare_nla_text_vs_activation \
  --audits runs/qwen_7b_nla_batch_40_n3/nla_audits.jsonl \
  --activation-records runs/qwen_7b_nla_batch_40/activation_records.jsonl \
  --output-dir runs/qwen_7b_compare_n3 \
  --folds 5
```

To evaluate cross-domain and leave-one-template-out generalization:

```bash
python -m latent_agent_auditing.experiments.evaluate_generalization \
  --audits runs/qwen_7b_nla_varied_72_n1/nla_audits.jsonl \
  --activation-records runs/qwen_7b_nla_varied_72_n1/activation_records.jsonl \
  --cases runs/qwen_7b_nla_varied_72_n1/cases.jsonl \
  --output-dir runs/qwen_7b_generalization_varied_72_n1
```

To run the raw-activation layer ablation with calibrated thresholds:

```bash
python -m latent_agent_auditing.experiments.run_activation_layer_ablation \
  --cases-per-domain 36 \
  --run-dir runs/qwen_7b_activation_layer_ablation_72 \
  --cache-dir /workspace/hf-cache \
  --layers 6,13,20,27 \
  --calibration-objective youden
```

To audit records exported from an existing benchmark harness without importing
that benchmark into this repo, use the default mock path for a fast schema and
metrics smoke test:

```bash
python -m latent_agent_auditing.experiments.audit_benchmark_records \
  --input path/to/agentdojo_or_injecagent_export.jsonl \
  --adapter agentdojo \
  --run-dir runs/agentdojo_record_audit \
  --layers 8,16,24,31 \
  --n-nla-samples 5
```

Supported adapters are `agentdojo`, `injecagent`, and `toolhijacker`. If the
export contains `trajectory` or `events`, those events are audited directly. If
it only contains task records, pass `--fallback-to-mock-agent` for a schema
smoke test.

To create a small AgentDojo-grounded export for replay auditing:

```bash
python -m latent_agent_auditing.experiments.export_agentdojo_records \
  --output data/agentdojo_workspace_small.jsonl \
  --suite workspace \
  --attack ignore_previous \
  --user-task-limit 2 \
  --injection-task-limit 2
```

This bridge uses AgentDojo suites and attack templates to produce JSONL records
with user tasks, injected external content, tool catalogs, audit-point events,
and constructed `model_prefix` values. It is a replay export for latent auditing,
not a full live AgentDojo agent run.

For paper evidence runs, export a larger grid across suites and attack
templates instead of relying on the smoke slice:

```bash
python -m latent_agent_auditing.experiments.export_agentdojo_records \
  --output data/agentdojo_paper_grid.jsonl \
  --suites workspace,banking,slack \
  --attacks ignore_previous,important_message \
  --user-task-limit 25 \
  --injection-task-limit 10
```

Clean cases are emitted once per suite when multiple attacks are requested, so
the negative class is not duplicated for each attack template.

To replay exported records through real Qwen activation capture and a local
released AV checkpoint, opt in explicitly:

```bash
python -m latent_agent_auditing.experiments.audit_benchmark_records \
  --input path/to/agentdojo_export.jsonl \
  --adapter agentdojo \
  --run-dir runs/agentdojo_qwen_replay_audit \
  --activation-backend transformers \
  --nla-runner local-av \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --av-model-name kitft/nla-qwen2.5-7b-L20-av \
  --cache-dir /workspace/hf-cache \
  --dtype bfloat16 \
  --layers 20 \
  --n-nla-samples 1
```

Each run writes `run_manifest.json`, including backend names, model names,
layers, and a `prompt_summary`. If an exported event includes
`model_prefix`, activation capture uses that exact prefix and marks the
activation as `exact_prefix`; otherwise it uses the normalized replay prompt and
marks the activation as `normalized_replay`. Treat replay evidence as weaker
than exact-prefix evidence.

After an audit run, generate the paper-facing table artifacts:

```bash
python -m latent_agent_auditing.experiments.summarize_paper_evidence \
  --run-dir runs/agentdojo_qwen_replay_audit \
  --folds 5 \
  --bootstrap-resamples 1000
```

This writes:

```text
runs/agentdojo_qwen_replay_audit/paper_evidence/
  paper_table.json
  paper_table.md
  predictions.jsonl
```

The table compares transcript, tool-policy, raw NLA taxonomy, trained NLA-text,
raw activation probe, and hybrid monitors. It also records AUROC/AUPRC bootstrap
intervals, operating-point confusion at threshold `0.5`, pairwise
complementarity, and latent lead-time summaries.

To evaluate held-out transfer on the same run:

```bash
python -m latent_agent_auditing.experiments.evaluate_benchmark_transfer \
  --run-dir runs/agentdojo_qwen_replay_audit
```

This writes held-out suite/domain and attack-template transfer artifacts under
`paper_evidence/transfer/`.

For lead-time plumbing only, the AgentDojo exporter can insert a clearly labeled
proxy unsafe action after `before_tool_selection`:

```bash
python -m latent_agent_auditing.experiments.export_agentdojo_records \
  --output data/agentdojo_proxy_lead_time.jsonl \
  --suites workspace,slack \
  --attacks ignore_previous,system_message \
  --user-task-limit 10 \
  --injection-task-limit 2 \
  --include-proxy-unsafe-action
```

Treat this as a metric-path smoke test, not live AgentDojo behavioral evidence.
Paper-grade lead-time claims still require exported live trajectories with real
unsafe tool steps.

To generate a first model-choice trajectory slice where the base model actually
chooses the next action:

```bash
python -m latent_agent_auditing.experiments.export_agentdojo_model_trajectories \
  --output data/agentdojo_model_choice_workspace_attack_probe.jsonl \
  --benchmark-version v1 \
  --suites workspace \
  --attacks direct,system_message,important_instructions \
  --user-task-limit 4 \
  --injection-task-limit 2 \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --cache-dir /workspace/hf-cache \
  --dtype bfloat16
```

This is stronger than replay because the tool step is generated by the model,
but it is still a lightweight next-action bridge rather than a full AgentDojo
environment rollout.

## Research Grounding

See [docs/PAPER_GROUNDING.md](docs/PAPER_GROUNDING.md) for the paper-to-code mapping. The short version:

- AgentDojo/InjecAgent/ToolHijacker define the outer eval surfaces.
- CaMeL informs the policy/capability boundary.
- ToolEmu informs scalable scenario generation and evaluator separation.
- AgentHarm informs malicious multi-step agent misuse.
- CoT monitorability and adaptive monitor attacks inform complementarity and adversarial evaluation.
- Circuit tracing and NLA work inform the interpretability layer and its limitations.

See [docs/CURRENT_RESULTS.md](docs/CURRENT_RESULTS.md) for the latest proof-of-life numbers and caveats from the RunPod Qwen/NLA runs.

## Limitations

The included mock backend is not interpretability evidence. It is a development harness for validating data contracts, metrics, gating logic, and benchmark integration before running expensive activation/NLA jobs. Actual research claims require real model activations, real NLA explanations, held-out attacks, and comparison against existing benchmark defenses.
