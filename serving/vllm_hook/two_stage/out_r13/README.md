# r13 outputs

`eval_r13_<model>.json`: pre-registered (`eval_r13.py`, commit 37747df; the env call in `rollout_r13.py` was
fixed to apply PYTHONHASHSEED in 9ab3db1, before any run). Model-driven AgentDojo rollouts (`rollout_r13.py`,
`env_r13.py`): every v1 user task with no injection and per injection task with
important_instructions_no_model_name, up to 8 actions, captured at every step; labels from
`label_steps_r13.py` (the step's own action, with an empty-action control). Tests under "tests".

| File | Made by | md5 |
|---|---|---|
| roll_qwen.npz | `rollout_r13.py` | 7a7dc91e55cab14138ddfd415cf48c74 |
| roll_qwen.steps.jsonl | same run, the steps with their messages | ebedcdce4673c92500496d81f5ce50f0 |
| roll_qwen.texts.jsonl | same run, the actions | d129f401a5f62c4cbb88963e246e8465 |
| roll_qwen.labels.jsonl | `label_steps_r13.py` | 8d28c840e02d4dd52f2287bf1d3cb719 |
| roll_granite.npz | `rollout_r13.py` | 61f6dbdd9b611bb59e0798293c5bb67a |
| roll_granite.steps.jsonl | same run, the steps with their messages | cab1bcfd841914c34ef41d0270c9c7a4 |
| roll_granite.texts.jsonl | same run, the actions | 9f9a7a5d730e7e5a2d7891a983964a1a |
| roll_granite.labels.jsonl | `label_steps_r13.py` | 9724c4017a279079530f9b5d47727db2 |
| roll_hermes.npz | `rollout_r13.py` | a75a4e776d6bc496fe8724bcc8d6b90c |
| roll_hermes.steps.jsonl | same run, the steps with their messages | 21b12137325efab9370e7924e494e01b |
| roll_hermes.texts.jsonl | same run, the actions | 1be597bda3372ff3bc7dff6f65e8f915 |
| roll_hermes.labels.jsonl | `label_steps_r13.py` | a45d5a22d19e873ef55b54e8993a281f |

Rollout sizes and labels (`label_steps_r13.py` output):

- qwen: steps 2336 {'clean': 305, 'pre': 1286, 'arrival': 270, 'after': 475}; with injection 745: strict 260, strict with no action 188, follows 194, unsafe 201
- granite: steps 1515 {'clean': 200, 'pre': 1002, 'arrival': 277, 'after': 36}; with injection 313: strict 2, strict with no action 1, follows 4, unsafe 4
- hermes: steps 1423 {'clean': 193, 'pre': 947, 'arrival': 249, 'after': 34}; with injection 283: strict 4, strict with no action 2, follows 4, unsafe 4

## Test outcomes

| Model | F1 | F2 after / arrival | PD: B1, B2, B3 (V1A, GT_after, RA) | adopted |
|---|---|---|---|---|
| qwen | holds | loses / holds | False, False, (False, False, False) | False |
| granite | not shown | loses / holds | True, False, (True, False, True) | False |
| hermes | holds | inconclusive / holds | True, True, (True, False, False) | False |

Granite and Hermes act on the injection in only 4 rollout steps each (Qwen 201), so their rollout
recall readings (B1, B3 on RA, F2) rest on 0 to 4 unsafe steps and carry no information.
