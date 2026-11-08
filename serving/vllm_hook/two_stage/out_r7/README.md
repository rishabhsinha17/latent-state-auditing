# r7 outputs

- `eval_r7_<model>.json`: pre-registered (`eval_r7.py`, commit c09bef8; the seen-goal reference
  got the cells' skip rule in 02ac2a2): goal-diversity scaling on InjecAgent and AgentDojo split D
  with InjecAgent added to the decision probe's training data. `summarize_r7.py` prints the tables.
- `posthoc_r7_safecalls.json`: post hoc, what the safe tool calls on InjecAgent are.

Note on the protocol text: it says every InjecAgent step is added to the AgentDojo decision probe's
training data. The code adds InjecAgent's tool-call steps (`keep & hc`): 408 for Qwen, 65 for
Granite, 13 for Hermes (`n_injecagent_training_steps` in each JSON).

InjecAgent source: github.com/uiuc-kang-lab/InjecAgent at f19c9f2c79a4, data/ (MIT). Steps are
rebuilt with `build_steps_r7_injecagent.py <InjecAgent data dir> <out dir>`. Source md5s:
test_cases_dh_base.json d4272124a6f0f262a19cd4512a7d0438, test_cases_dh_enhanced.json
a8a9afd66d14b5c324849bffbe0ff60d, test_cases_ds_base.json 4202da2a059577ee232de88177262cdc,
test_cases_ds_enhanced.json e0ec6093269a3bcc7016c32ec1f22486, tools.json d8d1d689b650ce3f96f13dda9277897f.

Captures, not in git:

| File | md5 |
|---|---|
| gen_ia_qwen.npz | 5d10fe18c063562336b94f0179b708b1 |
| tf6ia_qwen.npz | e8a4b3252b17cfac2f31cbf1c4fb7098 |
| gen_ia_granite.npz | aced443fb0a23633bd60197566bcbee0 |
| tf6ia_granite.npz | ba3cb6e62e05ab4b333d8d894c107735 |
| gen_ia_hermes.npz | f7c6dbd40686638781898b4b142cb689 |
| tf6ia_hermes.npz | 3d770a7f25ad9ce3bbdfeb45757638b0 |

`../data/injecagent_suite_full.jsonl` is the step list, and `../data/gen_ia_<model>.texts.jsonl` / `.labels.jsonl` the generated actions and their labels (`label_steps_r7.py`, Granite with `--permissive`).
