# r15 outputs

`eval_r15_<model>.json`: pre-registered (`eval_r15.py`, commit e6b8c6e). Clean false alarms of the risk probe
alone (S1), the presence cascade (CUR) and the decision-time cascade (PD), all trained on AgentDojo v1, on
1,201 clean steps from 703 ToolACE conversations (`build_steps_r15_toolace.py`; Team-ACE/ToolACE data.json,
Apache-2.0, md5 60f9402ab106213b06e4b6b8accedd34). Test X1 under "tests". Before the commit, checks used a
fake ToolACE capture only, so the outcome was blind.

| File | Made by | md5 |
|---|---|---|
| gen_tx_qwen.npz | `gen_capture_r4.py` on ../chat_toolace (qwen) | efe51993eaadaabd13ab0109b59d7ec8 |
| gen_tx_qwen.texts.jsonl | same run, the actions | 4bcfdaa4e10ac406434d9d4ce0539d7c |
| gen_tx_granite.npz | `gen_capture_r4.py` on ../chat_toolace (granite) | 166b53b7ebf17f2a0438699863b65aeb |
| gen_tx_granite.texts.jsonl | same run, the actions | 4801b2bb5d4e1ec595719a3a7fabae61 |
| gen_tx_hermes.npz | `gen_capture_r4.py` on ../chat_toolace (hermes) | b1f99b379dc151a5ad495f7392459b5c |
| gen_tx_hermes.texts.jsonl | same run, the actions | 44208c7dcdd4886db306eefbf6d3641a |
