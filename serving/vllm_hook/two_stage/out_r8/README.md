# r8 outputs

- `eval_r8_<model>.json`: pre-registered (`eval_r8.py`, commit 078afbc): the training-free attention
  provenance stage (ATT) on AgentDojo v1 (splits A and D, tool-call steps and all steps), the v1.2
  steps, and InjecAgent for Qwen (descriptive).
- `posthoc_r8_length_<model>.json`: post hoc, descriptive: AUROC of a length-only share and of ATT
  divided by it. Not used by any pre-registered decision.

Attention captures (`attn_prov_r8.py`, capture_qk score mode, eager engine, no prefix caching; per
head the last-token attention mass on the user's request and on the tool outputs) are small and are
in `../data/attn_r8/`:

| File | md5 |
|---|---|
| attn_old_qwen.npz | 0c09f3f1bea8c258181a8fc5b476517f |
| attn_new_qwen.npz | 980bbb80db8ef0827a3eb1e60e68318d |
| attn_ia_qwen.npz | 955d3c826fe8b7567957dd89e88717bb |
| attn_old_granite.npz | fdcb7ca99c00761b0569eaf75d4d18c7 |
| attn_new_granite.npz | c753e1fef27c291c99e28b79bc0b3a64 |
| attn_old_hermes.npz | 0134b4ea6c2b369419c2dfe276d8b806 |
| attn_new_hermes.npz | aa3803f3614fc8a9963f22a7793175bb |
