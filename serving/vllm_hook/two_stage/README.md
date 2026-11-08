# Two-stage false-positive reduction for vLLM-Hook

Evaluation code and outputs behind the two-stage cascade in vLLM-Hook
([docs/use_cases/two_stage.md](https://github.com/IBM/vLLM-Hook/blob/main/docs/use_cases/two_stage.md)).
The cascade artifacts used by `examples/demo_two_stage.py` are in
`../artifacts/qwen2.5-7b-instruct/two_stage/` (risk probe, then injection-presence probe) and
`../artifacts/qwen2.5-7b-instruct/two_stage_decision/` (risk probe, then presence times a
decision-time probe).

From r1 on, each evaluation fixed its protocol in its script header before the data it reads
existed or was copied to the analysis machine (eval_r1.py to eval_r5.py say PRE-SPECIFIED).
r0 and the r1 diagnostics were exploratory, and post hoc analyses say so in their headers. The cascade
module is vendored at `vendor/mia/utils/two_stage/cascade.py`, identical to vLLM-Hook's
`mia/utils/two_stage/cascade.py`. The CPU scripts need only numpy (`requirements.txt`).

## Rounds

| Round | Question | Scripts | Output |
|---|---|---|---|
| r0 | Does a second probe on the same capture lower the risk probe's FPR? | eval_two_stage.py, eval_context_thresholds.py, within_suite_auroc.py, per_suite.py | out/ |
| r1 | Why does a second risk probe fail, and does an injection-presence probe help? | diagnose.py, eval_r1.py, control_presence.py, per_suite_r1.py, export_r1.py | out_r1/, export_r1/ |
| r2 | Decision-time stage 2 from the first 16 generated tokens (GPU capture) | capture_decision.py, eval_r2.py, regen_dropped.py, posthoc_r2.py | out_r2/, out_r2_all/ |
| r3 | Same on vLLM 0.32 features, fixed-gate calibration, tool-name control | eval_r3.py, eval_r3b.py, posthoc_r3b.py, export_r3.py | out_r3/, out_r3b/, export_r3/ |
| r4 | Three models (Qwen2.5-7B, Granite-3.1-8B, Hermes-3-Llama-3.1-8B), held-out suites and attack templates, Attention Tracker as stage 2 | gen_capture_r4.py, label_steps_r4.py, eval_r4.py, eval_r4_recal.py, attn_tracker_r4.py, eval_r4_at.py, posthoc_r4_callpos.py | out_r4/ |
| r5 | Decision stage read at the tool call instead of the first tokens, a call-only rule as baseline, held-out injection tasks | tf_capture_r5.py, eval_r5.py, summarize_r5.py, e2e_window_r5.py | out_r5/ |
| r6 | Why the decision stage loses recall on held-out injection tasks: tool-residualized and value-masked windows, an untrained provenance rule, AgentDojo v1.2's new injection tasks | posthoc_r6_diagD.py, eval_r6a.py, eval_r6b.py, r6lib.py, tf_capture_r6.py, build_steps_r6_v12.py, label_steps_r6.py, summarize_r6.py | out_r6/ |
| r7 | Does goal diversity make the decision stage generalize? InjecAgent's 62 goals as a scaling test, and InjecAgent added to AgentDojo training | build_steps_r7_injecagent.py, label_steps_r7.py, eval_r7.py, posthoc_r7_safecalls.py, summarize_r7.py | out_r7/ |
| r8 | A training-free decision stage from attention: the share of attention on tool outputs against the user's request while the call is written | attn_prov_r8.py, eval_r8.py, posthoc_r8_length.py | out_r8/ |
| r9 | False alarms on clean steps at every depth of each task (337 more per model), cross-fitted by user task | build_steps_r9_cleandepth.py, eval_r9.py | out_r9/ |
| r10 | Retraining the presence probe and calibration (and, separately, the risk probe) with those clean steps as negatives, against the current recipe | eval_r10.py | out_r10/ |
| r11 | Attacked steps after the one where the injection arrives (2,994 per model): does r10's gain come from the depth layout, and does the released cascade keep its recall there? | build_steps_r11_persist.py, eval_r11.py, posthoc_r11_sticky.py | out_r11/ |

GPU benchmarks and end-to-end checks: bench_gpu.py, bench_decision_gpu.py, bench_collect_gpu.py,
bench_poll_gpu.py, bench_delivery_r4.py, e2e_cascade_gpu.py, e2e_decision_gpu.py,
e2e_window_r5.py. CPU analyzer timing: bench_analyzer_cpu.py.

## Data

The AgentDojo v1 steps come from `../agentdojo/` (PR #41 of vLLM-Hook): 1,826 steps over four
suites and three attack templates, 1,729 attacked and 97 clean. Labels for the Qwen2.5-7B
actions are `../results/suite_full.labels.jsonl`.

`data/` holds the step list with the October Qwen2.5-7B actions (`suite_full.jsonl`) and their
labels (`suite_full.labels.jsonl`, the same file as `../results/`), the r4 actions and labels of all three models (`*_full.texts.jsonl`,
`*_full.labels.jsonl`; Granite has strict and permissive labels), the Attention Tracker scores
(`granite_at.json`), decode-delivery timings (`delivery_r4_*.json` and `delivery_r5_rotated_hybrid.json`
rotate modes over the same prompts with the prefix cache on, `delivery_r5_controlled_*.json` warm each batch
first), and the r5 served-path check (`e2e_window_r5.json`).

In the published copies of `suite_full.jsonl` and the action texts, one key-format string that a
benchmark prompt carries (a ToolBench RapidAPI key, which the models copied into some actions) is
replaced by `<redacted key-format string>`. Nothing the CPU evaluations compute depends on it, but
`reproduce.sh` checks the md5 of the unredacted `suite_full.jsonl` that `../agentdojo/gen_capture.py`
writes next to `suite_full.npz`, so run it on that file.

The hidden-state captures are not in git (100 to 300 MB each). They are regenerated by the GPU
scripts, which record their software versions in their logs:

| File | Made by | md5 |
|---|---|---|
| suite_full.npz | `../agentdojo/gen_capture.py` | 4819017b9713ca9bdc5c82922269bc83 |
| dec_full.npz | `capture_decision.py` | fb51c90697b1cb78fd8da55b5b77d665 |
| qwen_full.npz | `gen_capture_r4.py Qwen/Qwen2.5-7B-Instruct 28` | 1c3108d72c0662bcc06bf27fac7dd762 |
| granite2_full.npz | `gen_capture_r4.py ibm-granite/granite-3.1-8b-instruct 40 --adapter granite` | bd6901644cf52b3e912db4c36043d40e |
| hermes_full.npz | `gen_capture_r4.py NousResearch/Hermes-3-Llama-3.1-8B 32` | 148ebff8d2ad549f99657882a98fa6d0 |
| tf_granite.npz, tf_hermes.npz, tf_qwen.npz | `tf_capture_r5.py` on the r4 actions | see out_r5/README.md |

GPU runs used vLLM 0.32.0, torch 2.13.0+cu130 and one NVIDIA A40, with
`VLLM_USE_FLASHINFER_SAMPLER=0`. Greedy generation is batch dependent, so a recapture can produce
a few different actions. The r4 and r5 scripts keep the action next to the features it came
from. The labels describe the action the features saw.

## Reproduce

r0 and r1 in one command (CPU, about 15 minutes on an Apple M1 Pro):

```
cp /path/to/suite_full.npz data/
PYTHON=/path/to/python3.12 PROBE_DIR=../artifacts/qwen2.5-7b-instruct ./reproduce.sh data
```

It checks the input md5s, builds `.venv` with numpy 2.3.5, reruns r0 and r1 into a fresh
directory and compares each output with `expected_md5.txt`. Outputs were produced with CPython
3.12.14 and numpy 2.3.5 on macOS arm64. Another BLAS build can change last-digit rounding and
with it the checksums. `out_r1/eval_r1.json` is the original pre-specified run. Its embedded
protocol text predates a post hoc note in eval_r1.py, so its md5 differs from a rerun while
every number is identical.

Later rounds take their inputs as arguments, as in

```
python -I eval_r4.py data vendor/mia/utils/two_stage/cascade.py granite2_full.npz data/granite2_full.texts.jsonl data/granite2_full.labels_permissive.jsonl 40 granite out_r4
python -I eval_r5.py data vendor/mia/utils/two_stage/cascade.py granite2_full.npz tf_granite.npz data/granite2_full.texts.jsonl data/granite2_full.labels_permissive.jsonl 40 granite out_r5
python -I summarize_r5.py out_r5/eval_r5_granite.json
```

with the captures in the working directory. The GPU scripts also read `suite_steps.jsonl`, the
rendered steps, which `../agentdojo/build_suite_steps.py` regenerates (25 MB, md5
ff5c74c3a6c1437785854781530c951d).
