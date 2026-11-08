"""GPU (r8): where does the model look when it writes a tool call's arguments?

Training-free decision signal. For each step whose action holds a tool call (tf_capture_r5.call_spans),
the action is teacher-forced up to a query token and vLLM-Hook's attention capture (capture_qk,
qk_capture="score", the last token's post-softmax attention) records, per head, the attention mass
on two spans of the prompt:
  user   the tokens of the user's request (every user message)
  tool   the tokens of every tool output (the untrusted data, where an injection sits)
Queries (0-based action token positions; the request is prompt + action[:q + 1]):
  Q1  the token before the first token of the first call's first argument value (r6lib.token_roles)
  Q2  the token before the first token of its tool name (-1 = the last prompt token)
Heads: every head of four layers at fixed depths (1/4, 1/2, 3/4 and the last layer), plus, where
vLLM-Hook ships one, the Attention Tracker head list (Granite-3.1-8B; Llama-3-8B for Hermes).
Steps whose chat-template ids differ from the offset tokenization, or whose message contents are not
found verbatim in the rendered prompt, are skipped and listed. Saves <out>.npz: case_id (steps with a
call), q (n,2) action positions (-2 = absent), user/tool masses (n, 2, n_heads) float32, heads
(n_heads, 2) [layer, head], n_user_tok, n_tool_tok; <out>.skipped.json.
Usage (pod): attn_prov_r8.py <chat dir> <model> <texts jsonl> <out prefix> [--adapter granite]
             [--at-config <vLLM-Hook attention_tracker json>] [--plan-only]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

DEPTH = {"Qwen/Qwen2.5-7B-Instruct": (28, 28), "ibm-granite/granite-3.1-8b-instruct": (40, 32),
         "NousResearch/Hermes-3-Llama-3.1-8B": (32, 32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", type=Path)
    ap.add_argument("model")
    ap.add_argument("texts")
    ap.add_argument("out")
    ap.add_argument("--adapter", default="none", choices=("none", "granite"))
    ap.add_argument("--at-config", default=None)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tokenizer", default=None, help="plan-only: local tokenizer dir")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_capture_r4 import granite_messages
    from r6lib import token_roles
    from tf_capture_r5 import call_spans

    n_layers, n_heads = DEPTH[args.model]
    layers = sorted({n_layers // 4 - 1, n_layers // 2, 3 * n_layers // 4, n_layers - 1})
    heads = [(l, h) for l in layers for h in range(n_heads)]
    if args.at_config:
        heads += [tuple(x) for x in json.load(open(args.at_config))["params"]["important_heads"] if tuple(x) not in heads]
    ftc = lambda t: (lambda s: None if s is None else (s[0], s[2]))(call_spans(t))
    order = [json.loads(l)["case_id"] for l in open(args.chat / "suite_full.jsonl")][: args.limit]
    steps = {json.loads(l)["case_id"]: json.loads(l) for l in open(args.chat / "suite_steps.jsonl")}
    texts = {json.loads(l)["case_id"]: json.loads(l) for l in open(args.texts)}
    if args.plan_only:
        from transformers import AutoTokenizer
        tok, llm = AutoTokenizer.from_pretrained(args.tokenizer or args.model), None
    else:
        from vllm import SamplingParams
        from mia import MiaLLM
        cfg = Path(args.out + ".attn.json")
        cfg.write_text(json.dumps({"model_info": {"name": args.model}, "params": {"important_heads": [list(x) for x in heads]},
                                   "hookq": {"hookq_mode": "last_token"}}))
        llm = MiaLLM(model=args.model, worker_name="capture_qk", analyzer_name="attn_tracker", config_file=str(cfg),
                     hook_dir="/root/pinyu/mia_hook_qk", gpu_memory_utilization=0.85, max_model_len=16384,
                     enforce_eager=True, enable_prefix_caching=False)
        tok = llm.tokenizer
    render = granite_messages if args.adapter == "granite" else (lambda m: m)
    plan, skipped = [], {}
    for c in order:
        t = texts.get(c)
        if t is None or not t["text"] or t.get("skipped_too_long") or ftc(t["text"]) is None:
            continue
        enc_a = tok(t["text"], add_special_tokens=False, return_offsets_mapping=True)
        roles = token_roles(t["text"], enc_a["offset_mapping"], ftc)
        a = list(enc_a["input_ids"])
        v = next((i for i, r in enumerate(roles) if r == "value"), None)
        nm = next((i for i, r in enumerate(roles) if r == "name"), None)
        q = [v - 1 if v is not None and v >= 1 else -2, nm - 1 if nm is not None else -2]
        msgs = render(steps[c]["messages"])
        ids = tok.apply_chat_template(msgs, tools=steps[c]["tools"], add_generation_prompt=True, tokenize=True, return_dict=False)
        text = tok.apply_chat_template(msgs, tools=steps[c]["tools"], add_generation_prompt=True, tokenize=False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(enc["input_ids"]) != list(ids):
            skipped[c] = "tokenization"
            continue
        spans, cur, ok = {"user": [], "tool": []}, 0, True
        for m in msgs:
            body = m.get("content") or ""
            if m.get("role") not in ("user", "tool") or not body.strip():
                continue
            k = text.find(body, cur)
            if k < 0:
                ok = False
                break
            spans[m["role"]].append((k, k + len(body)))
            cur = k + len(body)
        if not ok or not spans["user"] or not spans["tool"]:
            skipped[c] = "span"
            continue
        sel = {r: np.array([i for i, (lo, hi) in enumerate(enc["offset_mapping"])
                            if any(lo < b and hi > x for x, b in spans[r])]) for r in spans}
        plan.append((c, list(ids), a, q, sel))
    print(f"{len(plan)} steps with a call, skipped {len(skipped)} ({json.dumps({k: list(skipped.values()).count(k) for k in set(skipped.values())})}), "
          f"heads {len(heads)} layers {layers}, Q1 present {sum(p[3][0] >= -1 for p in plan)}", flush=True)
    if args.plan_only:
        return
    sp = SamplingParams(temperature=0.0, max_tokens=1, extra_args={"qk_capture": "score"})
    n = len(plan)
    U = np.full((n, 2, len(heads)), np.nan, np.float32)
    T = np.full((n, 2, len(heads)), np.nan, np.float32)
    hidx = {h: i for i, h in enumerate(heads)}
    seen_layers = set()
    t0 = time.time()
    jobs = [(i, j) for i, p in enumerate(plan) for j in range(2) if p[3][j] >= -1]
    for b in range(0, len(jobs), args.chunk):
        part = jobs[b:b + args.chunk]
        reqs = [{"prompt_token_ids": plan[i][1] + plan[i][2][:plan[i][3][j] + 1]} for i, j in part]
        out = llm.generate(reqs, sp)
        qk = out[0].probes["qk_cache"]
        for entry in qk.values():
            ln = int(entry["layer_num"])
            seen_layers.add(ln)
            hs = [int(h) for h in (entry.get("heads") or [])]
            sc = entry["scores"]
            assert len(sc) == len(part), (len(sc), len(part))
            for (i, j), s in zip(part, sc):
                s = s.float()
                row = s[:, -1, :] if s.dim() == 3 else s.reshape(len(hs), -1)
                P = len(plan[i][1])
                row = row[:, :P].cpu().numpy()            # keys inside the prompt only
                for r, h in enumerate(hs):
                    if (ln, h) in hidx:
                        U[i, j, hidx[(ln, h)]] = row[r, plan[i][4]["user"]].sum()
                        T[i, j, hidx[(ln, h)]] = row[r, plan[i][4]["tool"]].sum()
        if (b // args.chunk) % 25 == 0:
            print(f"{b + len(part)}/{len(jobs)} {time.time() - t0:.0f}s layers seen {sorted(seen_layers)}", flush=True)
    np.savez(args.out + ".npz", case_id=np.array([p[0] for p in plan]), q=np.array([p[3] for p in plan]),
             user=U, tool=T, heads=np.array(heads), layers_seen=np.array(sorted(seen_layers)),
             n_user_tok=np.array([len(p[4]["user"]) for p in plan]), n_tool_tok=np.array([len(p[4]["tool"]) for p in plan]))
    Path(args.out + ".skipped.json").write_text(json.dumps(skipped))
    print(f"done {time.time() - t0:.0f}s, layers seen {sorted(seen_layers)}, missing head values {int(np.isnan(U).sum())}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
