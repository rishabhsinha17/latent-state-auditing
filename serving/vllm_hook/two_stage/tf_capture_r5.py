"""GPU (r5): decision features read at the tool call, by teacher forcing the r4 actions.

r4 read the decode states of generated tokens 1..16. r5 reads them where the call is. For each
AgentDojo step the action is its r4 generation (texts jsonl), re-tokenized. The state of action
token t (0-based) is the last-token state of prompt + action[:t+1], the state its decode pass saw;
vLLM's prefix cache makes the repeated prefills cheap (one prompt-only request per step first
warms the cache and also re-captures the prompt state).
Call location (call_spans): the first permissive JSON call (an object, or a list holding one, with
"name" and "arguments"/"parameters", the rule behind Granite's r4 labels); the call starts at a
"<tool_call>"/"<|tool_call|>" tag directly before that JSON, else at the JSON. Without such JSON
a lone tag still marks a call start (no name, call runs to the end of the action).
Windows of action tokens, each pooled by the mean (as r4 DEC):
  FIXED  tokens 0..15 (r4 DEC; kept per token for the parity check against r4)
  CALL   16 tokens from the token holding the call's first character
  NAME   16 tokens from the token holding the tool name's first character (falls back to CALL)
  SPAN   the call from its first to its last token, at most 64 tokens
Steps with no call use FIXED for every window. Windows are clipped to the action's length.
Saves <out>.npz: case_id, prefill_layer_L (n,H), fixed_layer_L (n,16,H), call_layer_L,
name_layer_L, span_layer_L (n,H) means, float16, NaN where absent; call_tok, name_tok, end_tok
(-1 if none), n_action.

Usage (pod): tf_capture_r5.py <chat dir> <model> <layer> <texts jsonl> <out prefix> [--adapter granite]
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

TAGS = ("<tool_call>", "<|tool_call|>")
J, SPAN_MAX = 16, 64


def call_spans(text):
    """(call start, tool-name start, call end) character offsets, -1 where absent; None if no call."""
    dec, i = json.JSONDecoder(), 0
    while i < len(text):
        j = min([k for k in (text.find("{", i), text.find("[", i)) if k >= 0], default=-1)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text[j:])
        except json.JSONDecodeError:
            i = j + 1
            continue
        calls = [d for d in (obj if isinstance(obj, list) else [obj])
                 if isinstance(d, dict) and "name" in d and ("arguments" in d or "parameters" in d)]
        if calls:
            seg = text[j:j + end]
            m = re.search(r'"name"\s*:\s*"' + re.escape(str(calls[0]["name"])) + '"', seg)
            name = j + m.start() if m else -1
            start = j
            for t in TAGS:
                k = text.rfind(t, 0, j)
                if k >= 0 and not text[k + len(t):j].strip():
                    start = min(start, k)
            return start, name, j + end
        i = j + max(end, 1)
    tags = [text.find(t) for t in TAGS if text.find(t) >= 0]
    return (min(tags), -1, len(text)) if tags else None


def char_to_tok(offs, ch):
    return next((t for t, (lo, hi) in enumerate(offs) if hi > ch), -1)


def windows(text, offs, n_action):
    """Token windows {name: list of 0-based action token indices} and (call, name, end) tokens."""
    fixed = list(range(min(J, n_action)))
    sp = call_spans(text)
    if sp is None:
        return {"fixed": fixed, "call": fixed, "name": fixed, "span": fixed}, (-1, -1, -1)
    c = char_to_tok(offs, sp[0])
    q = char_to_tok(offs, sp[1]) if sp[1] >= 0 else -1
    e = char_to_tok(offs, sp[2] - 1)
    if c < 0:
        return {"fixed": fixed, "call": fixed, "name": fixed, "span": fixed}, (-1, -1, -1)
    e = n_action - 1 if e < 0 else e
    call = list(range(c, min(c + J, n_action)))
    name = list(range(q, min(q + J, n_action))) if q >= 0 else call
    span = list(range(c, min(e + 1, c + SPAN_MAX, n_action)))
    return {"fixed": fixed, "call": call, "name": name, "span": span}, (c, q, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", type=Path)
    ap.add_argument("model")
    ap.add_argument("layer", type=int)
    ap.add_argument("texts")
    ap.add_argument("out")
    ap.add_argument("--adapter", default="none", choices=("none", "granite"))
    ap.add_argument("--steps-per-call", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--plan-only", action="store_true", help="tokenizer only: print window statistics")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_capture_r4 import granite_messages

    order = [json.loads(l)["case_id"] for l in open(args.chat / "suite_full.jsonl")][: args.limit]
    steps = {json.loads(l)["case_id"]: json.loads(l) for l in open(args.chat / "suite_steps.jsonl")}
    texts = {json.loads(l)["case_id"]: json.loads(l) for l in open(args.texts)}
    if args.plan_only:
        from transformers import AutoTokenizer
        tok, llm = AutoTokenizer.from_pretrained(args.model), None
    else:
        from vllm import SamplingParams
        from mia import MiaLLM
        cfg = Path(args.out + ".infer.json")
        cfg.write_text(json.dumps({"model_info": {"name": args.model},
                                   "hidden_states": {"layers": [args.layer], "mode": "last_token"}}))
        llm = MiaLLM(model=args.model, worker_name="capture_hs", analyzer_name="hidden_states",
                     config_file=str(cfg), hook_dir="/root/pinyu/mia_hook",
                     gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
                     enable_prefix_caching=True)
        tok = llm.tokenizer
    render = granite_messages if args.adapter == "granite" else (lambda m: m)
    n = len(order)
    toks = np.full((n, 3), -1, np.int32)
    n_action = np.zeros(n, np.int32)
    plan = []
    for i, c in enumerate(order):
        if texts[c].get("skipped_too_long") or not texts[c]["text"]:
            plan.append(None)
            continue
        enc = tok(texts[c]["text"], add_special_tokens=False, return_offsets_mapping=True)
        a = list(enc["input_ids"])
        n_action[i] = len(a)
        w, toks[i] = windows(texts[c]["text"], enc["offset_mapping"], len(a))
        pids = None if args.plan_only else tok.apply_chat_template(
            render(steps[c]["messages"]), tools=steps[c]["tools"], add_generation_prompt=True,
            tokenize=True, return_dict=False)
        plan.append((i, pids, a, w))
    need = [sorted(set().union(*p[3].values())) for p in plan if p]
    has_call = toks[:, 0] >= 0
    print(f"{n} steps, call {has_call.sum()}, name {(toks[:, 1] >= 0).sum()}, call token median "
          f"{int(np.median(toks[has_call, 0])) if has_call.any() else -1}, "
          f"requests {sum(len(x) + 1 for x in need)}", flush=True)
    if args.plan_only:
        np.savez(args.out + ".plan.npz", case_id=np.array(order), call_tok=toks[:, 0], name_tok=toks[:, 1],
                 end_tok=toks[:, 2], n_action=n_action)
        return

    sp = SamplingParams(temperature=0.0, max_tokens=1)
    L = args.layer
    pre = fixed = means = None
    t0 = time.time()

    def states(reqs):
        out = llm.generate(reqs, sp)
        entry = next(e for e in out[0].probes["hs_cache"].values() if int(e["layer_num"]) == L)
        hs = entry["hidden_states"]
        assert len(hs) == len(reqs), (len(hs), len(reqs))
        return [(t if t.dim() == 1 else t[-1]).float().cpu().numpy() for t in hs]

    for b in range(0, n, args.steps_per_call):
        part = [p for p in plan[b:b + args.steps_per_call] if p]
        if not part:
            continue
        warm = states([{"prompt_token_ids": p[1]} for p in part])
        reqs, keys = [], []
        for i, pids, a, w in part:
            for t in sorted(set().union(*w.values())):
                reqs.append({"prompt_token_ids": pids + a[:t + 1]})
                keys.append((i, t))
        got = dict(zip(keys, states(reqs)))
        for (i, pids, a, w), v in zip(part, warm):
            if pre is None:
                H = v.size
                pre = np.full((n, H), np.nan, np.float16)
                fixed = np.full((n, J, H), np.nan, np.float16)
                means = {k: np.full((n, H), np.nan, np.float16) for k in ("call", "name", "span")}
            pre[i] = v
            for t in w["fixed"]:
                fixed[i, t] = got[(i, t)]
            for k in means:
                if w[k]:
                    means[k][i] = np.mean([got[(i, t)] for t in w[k]], 0)
        if (b // args.steps_per_call) % 10 == 0:
            print(f"{b + args.steps_per_call}/{n} {time.time() - t0:.0f}s", flush=True)
    np.savez(args.out + ".npz", case_id=np.array(order), n_action=n_action, call_tok=toks[:, 0],
             name_tok=toks[:, 1], end_tok=toks[:, 2], **{f"prefill_layer_{L}": pre, f"fixed_layer_{L}": fixed},
             **{f"{k}_layer_{L}": v for k, v in means.items()})
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
