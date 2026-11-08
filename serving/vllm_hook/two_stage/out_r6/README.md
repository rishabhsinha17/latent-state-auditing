# r6 outputs

- `posthoc_r6_diagD_<model>.json`: post hoc diagnostic on r5 data (split D recall by whether the
  unsafe call's tool was the tool of an unsafe training step).
- `eval_r6a_<model>.json`: pre-registered (`eval_r6a.py`, commit 0f98294) on the r4 and r5 captures:
  tool-residualized probes and the provenance rule.
- `eval_r6b_<model>.json`: pre-registered (`eval_r6b.py`, same commit) on the r6 captures below:
  value-masked windows and the AgentDojo v1.2 new-goal test.
- `summarize_r6.py <json>` prints the tables.

Captures, not in git (`tf_capture_r6.py`, every step's call span token by token; `gen_capture_r4.py`
for the v1.2 steps; 0 fallback steps in every run):

| File | md5 |
|---|---|
| tf6old_qwen.npz | fd5b2b9163e5a3a0d5f2f7d9db478966 |
| tf6old_granite.npz | 4822ce2492f56a7e7db0ce1f1b0dcb4d |
| tf6old_hermes.npz | b19fcdbd2a681640a75e5fd237c0e22e |
| gen_new_qwen.npz | 72aa1e118d3a261501e36a768d3eb7a2 |
| gen_new_granite.npz | 5cd720b13b1e556eeab244ef867a0fc1 |
| gen_new_hermes.npz | 1230272f730138a0d3f5a96b7c2f56ce |
| tf6new_qwen.npz | 827119e1a760a582f9499fd5c68e0d04 |
| tf6new_granite.npz | 89b82eaf09c7bba4f4b0a6d20fe3b5fb |
| tf6new_hermes.npz | 6c91fce84c0cd9e5a7a37ff073f5a56b |

The v1.2 steps come from `build_steps_r6_v12.py` (936 steps, `suite_steps.jsonl` md5
21dea987e7f5897286f8f4944a2862b9, not in git); `../data/v12_suite_full.jsonl` is their step list,
and `../data/gen_new_<model>.texts.jsonl` / `.labels.jsonl` the generated actions and their
labels (`label_steps_r6.py`, Granite with `--permissive`). `../data/delivery_r6_pyspy.json` is
`bench_delivery_r5.py` run under py-spy, which inflates its timings.
