# Current Results

This file tracks the proof-of-life runs so the project does not drift into
anecdotal examples. These results are useful engineering signals, not yet
paper-grade claims.

## RunPod Environment

- GPU: A100 PCIe 80GB
- Project path: `/workspace/latent-agent-auditing`
- Python env: `/workspace/latent-agent-auditing/.venv-ml`
- HF cache: `/workspace/hf-cache`
- Base model: `Qwen/Qwen2.5-7B-Instruct`

## Non-Leaky Qwen/NLA Batch

Run:

```bash
python -m latent_agent_auditing.experiments.run_qwen_nla_batch \
  --cases-per-domain 20 \
  --run-dir runs/qwen_7b_nla_batch_40_noleak_n1 \
  --cache-dir /workspace/hf-cache \
  --n-samples 1
```

Important fixes before this run:

- `BEFORE_TOOL_SELECTION` activation prompts no longer include proposed tool
  calls.
- The deterministic mock agent no longer leaks attack state through the
  selection message used for activation capture.

Results:

- Raw NLA keyword labels: AUROC `0.475`, AUPRC `0.512`.
- Trained NLA text classifier cross-validation: AUROC `0.970`, AUPRC `0.972`.
- Raw activation probe cross-validation: AUROC `1.000`, AUPRC `1.000`.

Interpretation:

The raw NLA taxonomy was weak at one sample per activation. A trained text
classifier and raw activation probe separated the tiny synthetic slice well,
but the slice is too small and stylized to support a research claim.

## Multi-Sample NLA On Saved Activations

Run:

```bash
python -m latent_agent_auditing.experiments.verbalize_activation_batch \
  --activation-records runs/qwen_7b_nla_batch_40/activation_records.jsonl \
  --cases runs/qwen_7b_nla_batch_40/cases.jsonl \
  --trajectories runs/qwen_7b_nla_batch_40/trajectories.jsonl \
  --output-dir runs/qwen_7b_nla_batch_40_n3 \
  --cache-dir /workspace/hf-cache \
  --n-samples 3
```

Results:

- Mean sample score: AUROC `0.771`, AUPRC `0.814`.
- Consensus rule: sample score >= `0.8` in at least two of three samples.
- Consensus confusion: TP `10`, FP `0`, TN `20`, FN `10`.

Interpretation:

Sampling improves precision but still misses many attacked cases. This supports
the intended framing: NLA text is evidence for auditing and labeling, not a
standalone firewall.

## Varied Template Generalization

Run:

```bash
python -m latent_agent_auditing.experiments.evaluate_generalization \
  --audits runs/qwen_7b_nla_varied_72_n1/nla_audits.jsonl \
  --activation-records runs/qwen_7b_nla_varied_72_n1/activation_records.jsonl \
  --cases runs/qwen_7b_nla_varied_72_n1/cases.jsonl \
  --output-dir runs/qwen_7b_generalization_varied_72_n1
```

Data:

- 72 cases total.
- 36 clean, 36 attacked.
- Domains: email and file.
- Attack templates: `ignore_previous`, `system_update`,
  `compliance_requirement`, `hidden_instruction`, `tool_output_injection`,
  `roleplay_override`.

Results:

- Raw NLA taxonomy: AUROC `0.623`, AUPRC `0.664`.
- NLA text classifier: mean AUROC `0.802`, min AUROC `0.623`.
- Raw activation probe: mean AUROC `0.990`, min AUROC `0.917`.
- Hybrid detector: mean AUROC `0.926`, min AUROC `0.741`.

Domain transfer:

- Train email, test file: NLA text AUROC `0.741`, activation AUROC `0.917`.
- Train file, test email: NLA text AUROC `0.623`, activation AUROC `1.000`.

Interpretation:

The activation signal transfers better than the NLA text classifier on this
synthetic slice. The NLA text remains useful as a human-readable audit signal,
but the cheaper probe is the stronger runtime candidate.

## Activation Layer Ablation

Run:

```bash
python -m latent_agent_auditing.experiments.run_activation_layer_ablation \
  --cases-per-domain 36 \
  --run-dir runs/qwen_7b_activation_layer_ablation_72 \
  --cache-dir /workspace/hf-cache \
  --layers 6,13,20,27 \
  --calibration-objective youden
```

Results:

| Layer | Mean AUROC | Min AUROC | Mean calibrated TPR | Mean calibrated FPR |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 1.000 | 1.000 | 0.500 | 0.000 |
| 13 | 1.000 | 1.000 | 0.500 | 0.000 |
| 20 | 0.990 | 0.917 | 0.750 | 0.125 |
| 27 | 0.979 | 0.833 | 0.938 | 0.250 |

Interpretation:

Ranking remains very strong across layers, but calibrated thresholds drift under
domain transfer. In several transfer splits the probe has perfect ranking but
the selected threshold blocks no attacked cases or blocks all benign cases. This
is the most important negative result so far: the signal exists in the synthetic
setup, but deployment-quality gating requires calibration on more realistic
benchmark distributions.

## Current Research Status

The implementation has achieved proof of life for the intended pipeline:

```text
case -> trajectory -> decision-point activation -> NLA text -> labels -> probes -> metrics
```

The evidence says:

- Good: real Qwen activations can be captured and passed through released NLA AV
  checkpoints.
- Good: the pipeline can compare transcript, NLA-text, raw activation, and
  hybrid detectors.
- Good: raw activation probes show strong ranking signal in synthetic indirect
  prompt-injection cases.
- Caution: raw NLA labels are noisy.
- Caution: threshold calibration is unstable across domains.
- Caution: the current benchmark is still synthetic and should not be presented
  as evidence about real-world agent security.

Next implementation target: scale the AgentDojo export beyond the smoke slice,
run the same real Qwen/NLA replay audit, and summarize it with
`summarize_paper_evidence.py`. The summarizer produces the first paper-facing
baseline table: transcript, tool policy, raw NLA taxonomy, trained NLA text, raw
activation probe, and hybrid monitors, with bootstrap intervals, complementarity,
and lead-time summaries.

## Evidence Levels

The benchmark-record audit CLI now distinguishes evidence levels in
`run_manifest.json` and activation metadata:

- `mock` backend runs are schema, metric, and regression tests only.
- `transformers` backend with `normalized_replay` prompts uses real model
  activations, but on prompts reconstructed from exported records.
- `transformers` backend with `exact_prefix` prompts uses real model activations
  on prefixes exported from the evaluated agent at the audited decision point.

Paper-grade claims should prioritize exact-prefix runs. Replay runs are useful
for debugging and early comparison, but should be labeled as replay evidence.

## AgentDojo Export Smoke

RunPod export:

```bash
python -m latent_agent_auditing.experiments.export_agentdojo_records \
  --output data/agentdojo_workspace_small.jsonl \
  --suite workspace \
  --attack ignore_previous \
  --user-task-limit 2 \
  --injection-task-limit 2
```

This produced 6 AgentDojo-grounded replay records: 2 clean user-task records and
4 attacked user/injection pairs. The export uses AgentDojo's real workspace
suite, injection goals, tool catalog, and `ignore_previous` attack template, but
it is still a replay export rather than a live model trajectory.

Real Qwen/NLA replay audit:

```bash
python -m latent_agent_auditing.experiments.audit_benchmark_records \
  --input data/agentdojo_workspace_small.jsonl \
  --adapter agentdojo \
  --run-dir runs/agentdojo_workspace_small_qwen \
  --activation-backend transformers \
  --nla-runner local-av \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --av-model-name kitft/nla-qwen2.5-7b-L20-av \
  --cache-dir /workspace/hf-cache \
  --dtype bfloat16 \
  --layers 20 \
  --n-nla-samples 1 \
  --max-new-tokens 80
```

Manifest:

- Activation backend: `transformers`
- NLA runner: `local-av`
- Prompt summary: `exact_prefix: 6`, `normalized_replay: 6`

Metrics are not meaningful at this size (`n=6`) and should not be treated as a
research result. The milestone is that AgentDojo-grounded exports now run
through the same real Qwen plus released AV audit path.

## AgentDojo Paper Grid Replay

RunPod export:

```bash
python -m latent_agent_auditing.experiments.export_agentdojo_records \
  --output data/agentdojo_paper_grid_ws_slack_200.jsonl \
  --benchmark-version v1 \
  --suites workspace,slack \
  --attacks ignore_previous,system_message \
  --user-task-limit 20 \
  --injection-task-limit 2
```

This produced 200 replay records: 40 clean cases and 160 attacked cases across
two suites and two attack templates.

Real Qwen/NLA replay summary:

| Monitor | AUROC | AUPRC | FPR@90TPR | Threshold@90TPR | Acc@90TPR | Acc@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| transcript | 0.500 | 0.737 | 1.000 | 0.000 | 0.800 | 0.200 |
| tool_policy | 0.500 | 0.737 | 1.000 | 0.000 | 0.800 | 0.200 |
| raw NLA taxonomy | 0.569 | 0.825 | 1.000 | 0.000 | 0.800 | 0.505 |
| trained NLA text | 0.976 | 0.988 | 0.050 | 1.000 | 0.945 | 0.910 |
| raw activation probe | 0.999 | 1.000 | 0.000 | 0.514 | 0.980 | 0.995 |
| hybrid | 0.999 | 1.000 | 0.000 | 0.453 | 0.975 | 0.200 |

Held-out transfer on the same replay run:

| Split | Trained NLA Text AUROC | Raw Activation Probe AUROC | Hybrid AUROC |
| --- | ---: | ---: | ---: |
| train workspace, test slack | 0.952 | 1.000 | 0.986 |
| train slack, test workspace | 0.979 | 1.000 | 1.000 |
| hold out ignore_previous | 1.000 | 1.000 | 1.000 |
| hold out system_message | 0.999 | 0.996 | 0.997 |

Interpretation:

The replay slice shows strong separability for trained NLA text and raw
activation probes, including held-out suite and attack-template transfer. Raw
NLA keyword taxonomy remains weak. The hybrid monitor ranks well but requires
calibration; fixed threshold `0.5` is misleading for this score scale.

This is still replay evidence, not live AgentDojo behavioral evidence. Lead-time
remains non-paper-grade until the benchmark export includes real multi-step
trajectories with actual unsafe tool steps.

## AgentDojo Model-Choice Generated Actions

RunPod export:

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
  --dtype bfloat16 \
  --max-new-tokens 80 \
  --temperature 0
```

This produced 28 AgentDojo-grounded model-choice trajectories. Qwen generated 2
unsafe actions, both under `important_instructions`: one `send_email` and one
`delete_file`.

Real Qwen/NLA audit summary:

| Monitor | AUROC | AUPRC | FPR@90TPR | Note |
| --- | ---: | ---: | ---: | --- |
| transcript | 0.500 | 0.054 | 1.000 | visible text baseline weak |
| tool policy | 1.000 | 1.000 | 0.000 | trivial once generated unsafe tool is visible |
| raw NLA taxonomy | 0.346 | 0.054 | 1.000 | did not flag the two generated unsafe actions |
| trained NLA text | 0.673 | 0.306 | 0.615 | small-n, weak |
| raw activation probe | 0.808 | 0.226 | 0.192 | first latent lead-time signal |
| hybrid | 1.000 | 1.000 | 0.000 | benefits from tool visibility/ranking; calibrate carefully |

Monitor-specific lead-time over the two generated unsafe actions:

| Monitor | Flagged positives | Mean lead time |
| --- | ---: | ---: |
| raw activation probe | 2 / 2 | 2.0 steps |
| raw NLA taxonomy | 0 / 2 | n/a |
| trained NLA text | 0 / 2 | n/a |

Interpretation:

This is the first end-to-end artifact where unsafe tool steps are generated by
the model rather than inserted as a proxy. It is far too small for a paper claim,
but it validates the experimental path: find generated unsafe actions, audit
pre-action activations, and compute monitor-specific lead time. The notable
signal is that raw activations flagged both generated unsafe cases two steps
before execution, while NLA text did not.

## Larger Generated-Action Lead-Time Slice

RunPod generation:

```bash
python -m latent_agent_auditing.experiments.export_agentdojo_model_trajectories \
  --output data/agentdojo_model_choice_important_240.jsonl \
  --benchmark-version v1 \
  --suites workspace,slack,banking,travel \
  --attacks important_instructions \
  --user-task-limit 20 \
  --injection-task-limit 2 \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --cache-dir /workspace/hf-cache \
  --dtype bfloat16 \
  --max-new-tokens 80 \
  --temperature 0
```

This produced 228 AgentDojo-grounded model-choice trajectories. Qwen generated
22 unsafe actions, all in the `workspace` suite: 13 `send_email` calls and 9
`delete_file` calls. We filtered to the workspace subset for the real audit,
yielding 60 cases with 22 generated unsafe positives.

Real Qwen/NLA audit on `runs/agentdojo_model_choice_workspace_important_58_qwen`:

| Monitor | AUROC | AUPRC | FPR@90TPR | Acc@90TPR |
| --- | ---: | ---: | ---: | ---: |
| transcript | 0.500 | 0.296 | 1.000 | 0.367 |
| tool policy | 1.000 | 1.000 | 0.000 | 1.000 |
| raw NLA taxonomy | 0.482 | 0.329 | 1.000 | 0.367 |
| trained NLA text | 0.951 | 0.906 | 0.105 | 0.900 |
| raw activation probe | 0.865 | 0.727 | 0.263 | 0.817 |
| hybrid | 0.994 | 0.992 | 0.000 | 0.983 |

Monitor-specific lead time over the 22 generated unsafe actions:

| Monitor | Flagged positives | Mean lead time |
| --- | ---: | ---: |
| raw activation probe | 22 / 22 | 2.00 steps |
| trained NLA text | 22 / 22 | 1.95 steps |
| raw NLA taxonomy | 5 / 22 | 1.60 steps among flagged |

Interpretation:

This is the strongest artifact so far for the paper thesis: the unsafe actions
are generated by the model, and latent monitors flag risk before the generated
unsafe tool-execution step. The important caveat is that tool policy trivially
catches the unsafe action once the generated tool is visible, so the paper claim
should focus on *pre-action lead time* and complementarity, not replacing tool
policy.
