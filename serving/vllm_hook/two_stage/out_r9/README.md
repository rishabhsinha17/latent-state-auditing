# r9 outputs

- `eval_r9_<model>.json`: pre-registered (`eval_r9.py`, commit 5043004): false alarms of every detector
  on 337 clean steps per model at every depth of each user task's ground-truth trajectory
  (`build_steps_r9_cleandepth.py`; 339 built, 2 too long for the context limit), cross-fitted by user
  task, with the 97 v1 clean steps for comparison.

Captures not in git: `gen_cd_<model>.npz` (`gen_capture_r4.py`):
- gen_cd_qwen.npz c95bafd9d0a4b0afcdc5e90180184403
- gen_cd_granite.npz 7cdbd7dac7d658404676cce4a5334a71
- gen_cd_hermes.npz 531072bad014fabf4866ae45714caa81

In `../data/`: `cleandepth_suite_full.jsonl` (the step list), `gen_cd_<model>.texts.jsonl` (the actions) and `attn_r9/attn_cd_<model>.npz` (attention captures, `attn_prov_r8.py`).
