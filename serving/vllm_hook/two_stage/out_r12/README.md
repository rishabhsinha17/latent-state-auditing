# r12 outputs

`eval_r12_<model>.json`: pre-registered (`eval_r12.py`, commit 8010cf7). Two AgentDojo templates no
earlier round used (`tool_knowledge`, `system_message`; `build_steps_r12_templates.py`, 3,542 steps,
1,258 of them arrival steps), scored by the risk probe alone (S1), the released presence cascade (CUR),
r10's retrained one (NEW-P), both with sticky presence (CUR-K, NEW-P-K), and the decision-time cascade
(PD, PD-K). The bar's outcome per model is under "tests". Suffix "/noskip": without the skip rule.

`boundary_r12_<model>.json`: `boundary_r12.py` (same commit), the one-pass header reading against the
separately captured steps (E1). `tok_prefix_<model>.json`: `tok_prefix_r12.py`, committed with the
pre-registration.

Inputs: the r4 v1, r9 clean-depth and r11 later-depth captures and labels (md5s in ../README.md,
../out_r9/README.md and ../out_r11/README.md), and the r12 captures and labels:

| File | Made by | md5 |
|---|---|---|
| gen_nw_qwen.npz | `gen_capture_r4.py` on ../chat_r12 (qwen) | ae75db7470d985357b4eb045acc9c1a0 |
| gen_nw_qwen.texts.jsonl | same run, the actions | 1308b2bd8f9be4ab2856cc38e4d11167 |
| bnd_qwen.npz | `hs_boundary_r12.py` (qwen) | 091bae7f00f88a7ee61e0331b78c0d14 |
| gen_nw_granite.npz | `gen_capture_r4.py` on ../chat_r12 (granite) | 34ecb4a1bf94ead5d908ee868054ee8f |
| gen_nw_granite.texts.jsonl | same run, the actions | 79793c5b7a990527a9d2606972207c9b |
| bnd_granite.npz | `hs_boundary_r12.py` (granite) | e699bb03b8e21be8a7a141eaffc2c266 |
| gen_nw_hermes.npz | `gen_capture_r4.py` on ../chat_r12 (hermes) | 05670a6272507a9488ea33d04d398e0e |
| gen_nw_hermes.texts.jsonl | same run, the actions | 33ac4cb21d70648f2545238cae56b5d2 |
| bnd_hermes.npz | `hs_boundary_r12.py` (hermes) | 05a1b8349cfa2e9f7c3b8dcf954d217c |
| gen_nw_qwen.labels.jsonl | `label_steps_r4.py` | fa7b7d4ead70190e053c7bc016ea8a7d |
| gen_nw_granite.labels_permissive.jsonl | `label_steps_r4.py` --permissive | 3c19bf60727479ea9385360b0d3d7786 |
| gen_nw_hermes.labels.jsonl | `label_steps_r4.py` | 9644c05cb5766c929a14f318780dc5f3 |

Labeler control: an empty action on every r12 step gives 0 strict and 0 follows.

## Test outcomes (correction)

The message of commit 51d84a0 lists the failed criteria incompletely: for Hermes, NEW-P-K also fails
R1 and C2, not only C1. The full outcomes from `eval_r12_<model>.json`, "tests":

| Model | CUR-K | NEW-P-K | D1 (NP / PS) |
|---|---|---|---|
| qwen | fails C1 | fails C1 | inconclusive / inconclusive |
| granite | fails R1, C1, C2 | fails R1, R2 | inconclusive / loses |
| hermes | fails R1, C1 | fails R1, C1, C2 | inconclusive / inconclusive |
