# r14 outputs

`eval_r14_<model>.json`: pre-registered (`eval_r14.py`, commit b6a19f1). A presence probe trained on whether the
latest tool output carries the injection (PO), maximized over the task's steps so far with thresholds
calibrated on that maximum (PO-K); CUR-Kc applies the same calibration to the released presence probe.
Test sets include new model-driven rollouts with AgentDojo's tool_knowledge template (`rollout_r13.py`,
unchanged). Disclosure: Hermes' outcomes on the existing steps were visible in a reproduction run before
the commit (see its message); Qwen, Granite and all r14 rollout data were not.

| File | Made by | md5 |
|---|---|---|
| roll_qwen.npz | `rollout_r13.py` tool_knowledge | 110332b769fdf613f22ddcce98f29d8d |
| roll_qwen.steps.jsonl | same run, steps with messages | 2a1c710ec2bc9de31956a5e4c7e87758 |
| roll_qwen.texts.jsonl | same run, the actions | 887e17ebaaba6d1f1535f5a370df1b45 |
| roll_qwen.labels.jsonl | `label_steps_r13.py` | 9e5ec8c4c5e1342558abcf31760aadab |
| roll_granite.npz | `rollout_r13.py` tool_knowledge | 42b9caf7dbe12c9427981e6f6b6f06a8 |
| roll_granite.steps.jsonl | same run, steps with messages | 1cd5a2480477834ae800743932dcb06b |
| roll_granite.texts.jsonl | same run, the actions | 9cd388fe5df6e3f95843dc57edbb5a4b |
| roll_granite.labels.jsonl | `label_steps_r13.py` | f2ae692a77b80fa1c4e8137ca69c1b57 |
| roll_hermes.npz | `rollout_r13.py` tool_knowledge | 1c3d1a9619576b2f4be5e7a6aa2b2bfd |
| roll_hermes.steps.jsonl | same run, steps with messages | 0519979c584e7a662c3afcc9aaf85c17 |
| roll_hermes.texts.jsonl | same run, the actions | 0e1233bbed9150d66ba6731bc3f17d2b |
| roll_hermes.labels.jsonl | `label_steps_r13.py` | da83dd4c152c938031a657beb1be0879 |

Rollout sizes and labels:

- qwen: steps 2363 {'clean': 308, 'pre': 1171, 'arrival': 346, 'after': 538}; with injection 884: strict 305, strict with no action 204, follows 231, unsafe 236
- granite: steps 1501 {'clean': 202, 'pre': 978, 'arrival': 284, 'after': 37}; with injection 321: strict 5, strict with no action 2, follows 9, unsafe 10
- hermes: steps 1421 {'clean': 194, 'pre': 939, 'arrival': 263, 'after': 25}; with injection 288: strict 2, strict with no action 1, follows 3, unsafe 3

## Test outcomes

| Model | PO-K | CUR-Kc |
|---|---|---|
| qwen | fails R1, R1D, C1 | fails R1D, C1 |
| granite | fails R1, R1D, C1, C2; R3 untestable | fails R1D, C1, C2; R3 untestable |
| hermes | fails R1, R1D, R2, C1, C2; R3 untestable | fails R1, R1D, C1; R3 untestable |
