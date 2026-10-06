# Tool-call risk probe for vLLM-Hook

Training and parity code for the `tool_call_risk` analyzer in [vLLM-Hook](https://github.com/IBM/vLLM-Hook). The analyzer reads the residual stream at the last prompt token of an agent step and predicts whether the model's next action follows a prompt injection. vLLM-Hook contains only the inference side. Everything that produces the probe is in this directory.

## Artifact

`artifacts/qwen2.5-7b-instruct/` holds `probe.npz` (mean, scale, weights, bias, layer), `probe.json` (model, layer, threshold, metrics, provenance) and `demo_steps.jsonl` (three AgentDojo steps used by `examples/demo_toolcallrisk.py`). `layer` follows the vLLM-Hook convention, the 1-based output of the Nth decoder block. That matches HuggingFace `hidden_states[N]` except at the last block (see below).

## The released probe (agentdojo/)

Training data comes from serving AgentDojo v1 agent steps through vLLM-Hook. We fit the probe on the same features the analyzer reads at inference. `build_suite_steps.py` executes each user task's ground-truth tool calls on the injected environment until a tool output contains the injection, across all four suites and three attack templates (important_instructions_no_model_name, ignore_previous, injecagent), plus clean steps. That gives 1,826 steps, 1,729 of them attacked. `gen_capture.py` renders each step with the Qwen2.5 tool-calling chat template, generates the next action greedily, and captures the last prompt token in the same pass. `label_steps.py` marks a step unsafe if executing the generated calls passes the injection task's AgentDojo security check. It also marks it unsafe if a generated call matches an injection ground-truth call on an argument the user task itself never uses. 358 attacked steps are unsafe.

`fit_chat.py` compares four layers and four probes (the paper's logistic probe and PCA-32/64/128 front ends) on the commitment task, attacked steps only, unsafe vs attacked-safe (`results/fit_report.json`). Selection uses folds grouped by suite and user task, so no task appears in its own test fold, and a probe is eligible only if at most 1% of verdicts flip and no score moves more than 0.3 between the batched capture and a one-prompt-per-call recapture.

| | value |
| --- | --- |
| selected | layer 28, PCA-32 + logistic |
| commitment AUROC, grouped folds | 0.869 |
| commitment AUROC, stratified folds | 0.889 |
| false-positive rate on attacked-safe steps at 90% recall | 0.325 |
| exposure AUROC (attacked-safe vs clean) | 0.532 |
| largest score movement across captures | 0.067 |

The probe predicts the action, not the presence of an injection (exposure AUROC is near chance). Deploy it beside an injection filter rather than in place of one. At 90% recall it still flags a third of attacked-but-safe steps, so pick the threshold from the false-positive rate you can accept. Layer 28 is the last block of Qwen2.5-7B. vLLM-Hook captures it before the final norm, unlike HuggingFace `hidden_states[28]`, and the probe is only valid on vLLM-Hook captures.

For the paper, the plain-text-prompt probe does not transfer to chat-template steps (0.477 AUROC on these features), and Qwen2.5-7B follows the injection far less often in its native tool-calling format (4 to 9% of workspace steps per template) than under the paper's plain-text prompt (141 of 240).

Steps in `demo_steps.jsonl` are training-distribution examples picked by `pick_demo.py`, not a held-out test.

## Capture parity (parity/)

This checks that hidden states captured through vLLM-Hook match the HuggingFace capture behind the paper's probes. `build_prompts.py` rebuilds all 839 audit prompts of `runs/exp_ws_full` from the pipeline's own code and checks each against its stored prompt hash (0 mismatches). `capture.py` feeds identical token ids to vLLM-Hook and to HuggingFace on the same GPU, and `compare.py` scores the paper's probes on each capture with the paper's folds (it reproduces the published 0.853 and 0.840 on the stored vectors).

Results (`results/capture_agreement.json`, `results/parity_report.json`, RTX 4090, vLLM 0.21.0, torch 2.11).

| Comparison | median cosine | median relative L2 |
| --- | --- | --- |
| Qwen2.5-7B bf16, vLLM-Hook vs HF (same GPU) | 0.99969 | 2.5% |
| Qwen2.5-7B bf16, HF today vs HF stored | 0.99968 | 2.5% |
| Qwen2.5-7B bf16, vLLM batched vs one prompt per call | 0.99974 | 2.3% |
| Qwen2.5-1.5B fp32, vLLM-Hook vs HF | 0.999999 | 0.13% |
| Qwen2.5-1.5B bf16, HF vs HF fp32 | 0.99989 | 1.5% |

vLLM-Hook captures the same tensor as HuggingFace. In fp32 the two agree to 0.13%, and in bf16 they differ by as much as two HuggingFace runs, which is bf16 rounding. The paper's commitment probe scores 0.849 AUROC on vLLM-Hook features against 0.853 on the stored ones.

The paper's L2 logistic probe is fragile under that rounding. It standardizes every dimension, so low-variance dimensions amplify noise, and single-case scores move by up to 0.93 between captures. Stronger L2 trades AUROC for little stability (`results/l2_sweep.json`). A PCA front end fixes it (`results/pca_sweep.json`). With 128 components the worst score movement drops to 0.21. The released probe uses this front end, folded into the same (mean, scale, weights, bias) form.

## Serving cost (agentdojo/bench.py)

280 workspace agent steps (about 4,500 prompt tokens each), 64 output tokens, Qwen2.5-7B-Instruct bf16 on one RTX 4090 (`results/bench.json`).

| Configuration | seconds | requests/s |
| --- | --- | --- |
| vLLM, eager | 158.6 | 1.766 |
| vLLM, CUDA graphs | 158.5 | 1.767 |
| vLLM-Hook loaded, capture off | 158.6 | 1.765 |
| vLLM-Hook, one layer to memory | 159.1 | 1.760 |
| vLLM-Hook, one layer to disk | 158.6 | 1.766 |

Capturing one layer adds 0.3% to wall time on this workload. These steps are prefill-bound. CUDA graphs do not help them, and the eager mode vLLM-Hook requires does not slow them down. Decode-heavy workloads will show a larger eager gap.

## Reproduce

Everything runs on one 24 GB GPU. `parity/run_pod.sh` installs vLLM 0.21.0 and the vLLM-Hook branch and runs the parity captures. For the probe, build steps in a separate environment with `pip install agentdojo` (`agentdojo/build_suite_steps.py`), then `agentdojo/gen_capture.py` (vLLM-Hook environment), `agentdojo/label_steps.py` (AgentDojo environment) and `agentdojo/fit_chat.py`.
