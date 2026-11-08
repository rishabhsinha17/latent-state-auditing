"""GPU (r6): per-token decode-time states over each action's first tool call, by teacher forcing.

r5 saved only window means. r6 keeps every token's state so that windows can be re-pooled on CPU
(for example without the tokens inside argument values). For each step with K = the end of its
call span (tf_capture_r5.windows: the call from its first token, at most 64 tokens) or 16 when it
has no call, clipped to the action:
  1. a prompt-only request warms vLLM's prefix cache (and re-captures the prompt state);
  2. one request for prompt + action[:K] with hs_mode="all_tokens" captures every token vLLM
     computes, which is the uncached tail: the last partial block of the prompt plus the K
     action tokens. Row r of that capture is position (P + K - n_rows + r), P = prompt length.
The mapping is checked per step: n_rows >= K and the implied cached length is a multiple of the
cache block size; otherwise the step is redone with r5's one-request-per-token method
(fallback count saved). The state of action token t is the state its decode pass saw, as in r5.
Saves <out>.npz: case_id, n_action, call_tok, name_tok, end_tok, K, fallback (bool),
prefill_layer_L (n,H), fixed_layer_L (n,16,H), span_idx (row of each call step in span_layer_L),
span_layer_L (n_calls,64,H) float16, NaN where absent.
Usage (pod): tf_capture_r6.py <chat dir> <model> <layer> <texts jsonl> <out prefix> [--adapter granite]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

J, SPAN_MAX = 16, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", type=Path)
    ap.add_argument("model")
    ap.add_argument("layer", type=int)
    ap.add_argument("texts")
    ap.add_argument("out")
    ap.add_argument("--adapter", default="none", choices=("none", "granite"))
    ap.add_argument("--steps-per-call", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_capture_r4 import granite_messages
    from tf_capture_r5 import windows
    from vllm import SamplingParams
    from mia import MiaLLM

    order = [json.loads(l)["case_id"] for l in open(args.chat / "suite_full.jsonl")][: args.limit]
    steps = {json.loads(l)["case_id"]: json.loads(l) for l in open(args.chat / "suite_steps.jsonl")}
    texts = {json.loads(l)["case_id"]: json.loads(l) for l in open(args.texts)}
    cfg = Path(args.out + ".infer.json")
    cfg.write_text(json.dumps({"model_info": {"name": args.model},
                               "hidden_states": {"layers": [args.layer], "mode": "last_token"}}))
    llm = MiaLLM(model=args.model, worker_name="capture_hs", analyzer_name="hidden_states",
                 config_file=str(cfg), hook_dir="/root/pinyu/mia_hook",
                 gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
                 enable_prefix_caching=True)
    tok = llm.tokenizer
    render = granite_messages if args.adapter == "granite" else (lambda m: m)
    n, L = len(order), args.layer
    toks = np.full((n, 3), -1, np.int32)
    n_action = np.zeros(n, np.int32)
    K = np.zeros(n, np.int32)
    plan = []
    for i, c in enumerate(order):
        t = texts.get(c)
        if t is None or t.get("skipped_too_long") or not t["text"]:
            plan.append(None)
            continue
        enc = tok(t["text"], add_special_tokens=False, return_offsets_mapping=True)
        a = list(enc["input_ids"])
        n_action[i] = len(a)
        w, toks[i] = windows(t["text"], enc["offset_mapping"], len(a))
        K[i] = max(J if toks[i, 0] < 0 else max(w["span"]) + 1, min(J, len(a)))
        K[i] = min(K[i], len(a))
        pids = tok.apply_chat_template(render(steps[c]["messages"]), tools=steps[c]["tools"],
                                       add_generation_prompt=True, tokenize=True, return_dict=False)
        plan.append((i, pids, a, w))
    calls = [i for i in range(n) if toks[i, 0] >= 0 and plan[i] is not None]
    span_row = {i: r for r, i in enumerate(calls)}
    print(f"{n} steps, {len(calls)} with a call, K median {int(np.median(K[K > 0]))}", flush=True)

    last = SamplingParams(temperature=0.0, max_tokens=1)
    allt = SamplingParams(temperature=0.0, max_tokens=1, extra_args={"hs_mode": "all_tokens"})
    pre = fixed = span = None
    fallback = np.zeros(n, bool)
    t0 = time.time()

    def entry_rows(o):
        e = next(e for e in o.probes["hs_cache"].values() if int(e["layer_num"]) == L)
        hs = [t if t.dim() == 2 else t.unsqueeze(0) for t in e["hidden_states"]]
        return __import__("torch").cat(hs, 0).float().cpu().numpy()

    def put(i, t, v):
        if t < J:
            fixed[i, t] = v
        w = plan[i][3]
        if i in span_row and w["span"] and w["span"][0] <= t <= w["span"][-1]:
            span[span_row[i], t - w["span"][0]] = v

    for b in range(0, n, args.steps_per_call):
        part = [p for p in plan[b:b + args.steps_per_call] if p]
        if not part:
            continue
        warm = llm.generate([{"prompt_token_ids": p[1]} for p in part], last, merge_probes=False)
        full = llm.generate([{"prompt_token_ids": p[1] + p[2][:K[p[0]]]} for p in part], allt, merge_probes=False)
        redo = []
        for (i, pids, a, w), ow, of in zip(part, warm, full):
            v0 = entry_rows(ow)[-1]
            if pre is None:
                H = v0.size
                pre = np.full((n, H), np.nan, np.float16)
                fixed = np.full((n, J, H), np.nan, np.float16)
                span = np.full((len(calls), SPAN_MAX, H), np.nan, np.float16)
            pre[i] = v0
            rows, k, P = entry_rows(of), int(K[i]), len(pids)
            cached = P + k - len(rows)
            if len(rows) < k or cached < 0 or cached % args.block_size:
                redo.append((i, pids, a, w))
                continue
            for t in range(k):
                put(i, t, rows[len(rows) - k + t])
        for i, pids, a, w in redo:                    # r5's method for steps whose mapping failed
            fallback[i] = True
            outs = llm.generate([{"prompt_token_ids": pids + a[:t + 1]} for t in range(int(K[i]))], last)
            e = next(e for e in outs[0].probes["hs_cache"].values() if int(e["layer_num"]) == L)
            for t, h in enumerate(e["hidden_states"]):
                put(i, t, (h if h.dim() == 1 else h[-1]).float().cpu().numpy())
        if (b // args.steps_per_call) % 10 == 0:
            print(f"{b + args.steps_per_call}/{n} {time.time() - t0:.0f}s fallback {int(fallback.sum())}", flush=True)
    np.savez(args.out + ".npz", case_id=np.array(order), n_action=n_action, call_tok=toks[:, 0],
             name_tok=toks[:, 1], end_tok=toks[:, 2], K=K, fallback=fallback,
             span_idx=np.array([span_row.get(i, -1) for i in range(n)], np.int32),
             **{f"prefill_layer_{L}": pre, f"fixed_layer_{L}": fixed, f"span_layer_{L}": span})
    print(f"done {time.time() - t0:.0f}s fallback {int(fallback.sum())}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(3 << 30))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
