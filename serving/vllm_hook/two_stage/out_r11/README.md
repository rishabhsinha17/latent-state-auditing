# r11 outputs

`eval_r11_<model>.json`: pre-registered (`eval_r11.py`, commit 5ce4996). Attacked AgentDojo v1 steps
at every depth after the injection first reaches the context (`build_steps_r11_persist.py`, 2,994
steps over 75 user tasks, AgentDojo's own attack templates and injection tasks), scored by the
current recipe (CUR), r10's retrained presence probe (NEW-P), a re-weighted variant (NEW-PW) and a
variant that also trains on attacked steps at every depth (NEW-PA). Test outcomes are under "tests".

Inputs: the r4 v1 captures and labels and the r9 clean-depth captures (md5s in ../README.md and
../out_r9/README.md), and the r11 later-depth captures and labels:

| File | Made by | md5 |
|---|---|---|
| gen_ps_qwen.npz | `gen_capture_r4.py` on ../chat_persist (qwen) | f81b4cdb3a276a41175c2d75280a73dd |
| gen_ps_qwen.texts.jsonl | same run, the actions | 7cc4cc1cbb626f33a19f8df5b8db2864 |
| gen_ps_granite.npz | `gen_capture_r4.py` on ../chat_persist (granite) | 219099b45bdaca2edb638e019e224083 |
| gen_ps_granite.texts.jsonl | same run, the actions | 9ecee69cc497cf69026020e3298cd517 |
| gen_ps_hermes.npz | `gen_capture_r4.py` on ../chat_persist (hermes) | c94122bf9fdeef17b923d2e6902f028f |
| gen_ps_hermes.texts.jsonl | same run, the actions | 31025549fefe89f0987d24faaf5cfe93 |
| gen_ps_qwen.labels.jsonl | `label_steps_r4.py` | 0f96c22e527d1b21af3dc1ece37f2bb8 |
| gen_ps_granite.labels_permissive.jsonl | `label_steps_r4.py` --permissive | 2097380f04a6ec52d10bcdd83f848779 |
| gen_ps_hermes.labels.jsonl | `label_steps_r4.py` | d9fa088b9270a439496d36b27b4cae1c |

`posthoc_r11_sticky_<model>.json`: `posthoc_r11_sticky.py` (commit a8e743b), post hoc for Qwen and
pre-specified for Granite and Hermes: stage 2 takes the maximum presence score over the task's
steps so far.

CUR and NEW-P reproduce eval_r10's numbers exactly on the v1 and clean-depth steps in splits A and D.
