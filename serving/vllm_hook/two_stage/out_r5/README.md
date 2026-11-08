# r5 outputs

`eval_r5_<model>.json` and `.log` come from `eval_r5.py` (protocol in its header, committed before
any r5 capture existed). `summarize_r5.py out_r5/eval_r5_<model>.json` prints the tables.

Inputs, not in git:

| Capture | Made by | md5 |
|---|---|---|
| tf_granite.npz | tf_capture_r5.py ibm-granite/granite-3.1-8b-instruct 40 data/granite2_full.texts.jsonl --adapter granite | 22e791801b717cf506d0220f593752a5 |
| tf_hermes.npz | tf_capture_r5.py NousResearch/Hermes-3-Llama-3.1-8B 32 data/hermes_full.texts.jsonl | b5a5a76d4244f5c72f1f3c294d44e12d |
| tf_qwen.npz | tf_capture_r5.py Qwen/Qwen2.5-7B-Instruct 28 data/qwen_full.texts.jsonl | ba901813764ef736c426b256a18dfdef |

Each eval also reads that model's r4 capture (md5s in ../README.md). GPU runs: one NVIDIA A40,
vLLM 0.32.0, torch 2.13.0+cu130, vLLM-Hook feature/two-stage-fpr. Served-path check
(`e2e_window_r5.py`, v2) and prefix-controlled delivery timings (`bench_delivery_r5.py`, capture
directory on container disk and on tmpfs) are in `../data/`.

Correction to the eval_r5.py header (kept as committed): it describes split D as holding out the
attacker's goals, "the same attacker addresses and accounts". D holds out injection tasks.
Attacker identifiers such as one bank account or one email address recur across a suite's
injection tasks, so D does not make them new.
