"""GPU (r5): does the served decode_window path give the teacher-forced call-span features?

Granite-3.1-8B (layer 40), the first --n steps whose r4 action holds a tool call (suite_full
order). Each is generated greedily in one pass, up to 512 tokens, with every decode pass captured
(hooks_on="both", merge_probes=False), in batches of --chunk. For each request
mia.utils.two_stage.tool_call_window finds the call in the generated token ids, and
request_features(..., "decode_window", window) averages those decode states. Where the decoded
generated ids equal the r4 action text, the feature is compared with tf_capture_r5's span
feature (cosine) and the window with tf_capture_r5's call/end tokens (those index a
re-tokenization of the text, which can differ from the generated ids by a token).
v2: the first run compared vLLM's output text, which drops the end-of-text token, with the r4
text, which keeps it, so only 1 of 64 matched; its timing pass ran capture-off first on cold
prompts and is dropped (see bench_delivery_r5.py for timing).
Usage (pod): e2e_window_r5.py <chat dir> <r4 texts jsonl> <tf_granite.npz> <out.json> [--n 64]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL, LAYER = "ibm-granite/granite-3.1-8b-instruct", 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", type=Path)
    ap.add_argument("texts")
    ap.add_argument("tf")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_capture_r4 import granite_messages
    from vllm import SamplingParams
    from mia import MiaLLM
    from mia.analyzers.two_stage_probe_analyzer import request_features
    from mia.utils.two_stage import tool_call_window

    order = [json.loads(l)["case_id"] for l in open(args.chat / "suite_full.jsonl")]
    steps = {json.loads(l)["case_id"]: json.loads(l) for l in open(args.chat / "suite_steps.jsonl")}
    texts = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(args.texts)}
    T = np.load(args.tf)
    row = {c: i for i, c in enumerate(T["case_id"])}
    pick = [c for c in order if T["call_tok"][row[c]] >= 0][: args.n]
    cfg = Path(args.out + ".infer.json")
    cfg.write_text(json.dumps({"model_info": {"name": MODEL},
                               "hidden_states": {"layers": [LAYER], "mode": "last_token"}}))
    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="two_stage_probe", config_file=str(cfg),
                 hook_dir="/root/pinyu/mia_hook_e2e", gpu_memory_utilization=0.85, max_model_len=16384)
    tok = llm.tokenizer
    prompts = [{"prompt_token_ids": tok.apply_chat_template(granite_messages(steps[c]["messages"]),
                tools=steps[c]["tools"], add_generation_prompt=True, tokenize=True, return_dict=False)} for c in pick]
    both = SamplingParams(temperature=0.0, max_tokens=512, skip_special_tokens=False, extra_args={"hooks_on": "both"})
    rows, times = [], []
    for k in range(0, len(pick), args.chunk):
        part, pr = pick[k:k + args.chunk], prompts[k:k + args.chunk]
        t0 = time.time()
        out = llm.generate(pr, both, merge_probes=False)
        win = [tool_call_window(tok, list(o.outputs[0].token_ids)) for o in out]
        feats = [request_features(o.probes, LAYER, "decode_window", 16, w) for o, w in zip(out, win)]
        t_on = time.time() - t0
        times.append({"chunk": k // args.chunk, "capture_and_features_s": round(t_on, 3),
                      "gen_tokens": int(sum(len(o.outputs[0].token_ids) for o in out))})
        for c, o, w, f in zip(part, out, win, feats):
            i = row[c]
            same = tok.decode(list(o.outputs[0].token_ids), skip_special_tokens=False) == texts[c]
            r = {"case_id": c, "same_text_as_r4": bool(same),
                 "window": list(w) if w else None, "tf_call_tok": int(T["call_tok"][i]), "tf_end_tok": int(T["end_tok"][i])}
            if same:
                ref = T[f"span_layer_{LAYER}"][i].astype(np.float64)
                r["cos_vs_tf_span"] = float(f @ ref / (np.linalg.norm(f) * np.linalg.norm(ref)))
            rows.append(r)
        print(f"chunk {k // args.chunk} {times[-1]}", flush=True)
    same = [r for r in rows if r["same_text_as_r4"]]
    cos = np.array([r["cos_vs_tf_span"] for r in same])
    starts = sum(r["window"] is not None and r["window"][0] == r["tf_call_tok"] for r in same)
    summary = {"n": len(rows), "same_text_as_r4": len(same), "window_start_equal_tf": int(starts),
               "cos_median": float(np.median(cos)) if cos.size else None,
               "cos_min": float(cos.min()) if cos.size else None,
               "window_none": int(sum(r["window"] is None for r in rows))}
    Path(args.out).write_text(json.dumps({"protocol": __doc__, "summary": summary, "times": times, "rows": rows}, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
