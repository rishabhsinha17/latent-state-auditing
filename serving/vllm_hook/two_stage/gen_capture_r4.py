"""GPU (r4): one model-agnostic pass that yields features AND the action they describe.

For every AgentDojo step in suite_full order:
  1. render with the model's chat template (--adapter granite: see granite_messages; v2 after the
     first Granite run produced no <|tool_call|> at all because the step's system message had
     replaced Granite's tool-calling instruction)
  2. batched capture through vLLM-Hook (capture_hs, one layer, last_token, hooks_on="both",
     merge_probes=False): prompt state and decode states for generated tokens 1..16 (17 tokens)
  3. continue each unfinished generation from its own 17-token prefix without capture
     (max 512 tokens in total), so the labelled action is exactly the one the features saw.
Saves <out>.npz (case_id, prefill_layer_L, dec_layer_L float16, n_prompt, finish17) and
<out>.texts.jsonl ({case_id, text}) for label_steps_r4.py.

Usage (pod): gen_capture_r4.py <chat dir> <model> <layer> <out prefix> [--adapter granite]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


GRANITE_TOOL_INSTRUCTION = (
    "You are Granite, developed by IBM. You are a helpful AI assistant with access to the following tools. "
    "When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list "
    "of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not "
    "have the ability to fulfill the request.")


def granite_messages(messages):
    """Granite's template replaces its default system prompt, which carries the <|tool_call|> protocol,
    whenever a system message is given, and it drops tool_calls. So: prepend that protocol sentence
    (verbatim from the template, without its date line) to the step's system message, and write prior
    tool calls in Granite's own syntax. Qwen's template adds its tool protocol on its own."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "system":
            m["content"] = GRANITE_TOOL_INSTRUCTION + "\n\n" + (m.get("content") or "")
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in m["tool_calls"]]
            m["content"] = (m.get("content") or "") + "<|tool_call|>" + json.dumps(calls)
            m.pop("tool_calls")
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", type=Path)
    ap.add_argument("model")
    ap.add_argument("layer", type=int)
    ap.add_argument("out")
    ap.add_argument("--adapter", default="none", choices=("none", "granite"))
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from vllm import SamplingParams
    from mia import MiaLLM

    order = [json.loads(l)["case_id"] for l in open(args.chat / "suite_full.jsonl")][: args.limit]
    steps = {}
    for line in open(args.chat / "suite_steps.jsonl"):
        s = json.loads(line)
        steps[s["case_id"]] = s
    cfg = Path(args.out + ".infer.json")
    cfg.write_text(json.dumps({"model_info": {"name": args.model},
                               "hidden_states": {"layers": [args.layer], "mode": "last_token"}}))
    llm = MiaLLM(model=args.model, worker_name="capture_hs", analyzer_name="hidden_states",
                 config_file=str(cfg), hook_dir="/root/pinyu/mia_hook",
                 gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
    tok = llm.tokenizer
    render = granite_messages if args.adapter == "granite" else (lambda m: m)
    ids = {c: tok.apply_chat_template(render(steps[c]["messages"]), tools=steps[c]["tools"],
                                      add_generation_prompt=True, tokenize=True, return_dict=False) for c in order}
    too_long = [c for c in order if len(ids[c]) > args.max_model_len - 512]
    print(f"{len(order)} steps, prompt tokens median {int(np.median([len(v) for v in ids.values()]))}, "
          f"max {max(len(v) for v in ids.values())}, too long {len(too_long)}", flush=True)
    sp_cap = SamplingParams(temperature=0.0, max_tokens=17, skip_special_tokens=False,
                            extra_args={"hooks_on": "both"})
    n, J = len(order), 16
    pre = dec = None
    gen = {}
    finish17 = np.zeros(n, bool)
    t0 = time.time()
    for k in range(0, n, args.chunk):
        cids = [c for c in order[k:k + args.chunk] if c not in too_long]
        out = llm.generate([{"prompt_token_ids": ids[c]} for c in cids], sp_cap, merge_probes=False)
        for c, o in zip(cids, out):
            i = order.index(c)
            entry = next(e for e in o.probes["hs_cache"].values() if int(e["layer_num"]) == args.layer)
            passes = [t.float().cpu().numpy().reshape(-1) for t in entry["hidden_states"]]
            if pre is None:
                H = passes[0].size
                pre = np.full((n, H), np.nan, np.float16)
                dec = np.full((n, J, H), np.nan, np.float16)
            pre[i] = passes[0]
            d = passes[1:1 + J]
            if d:
                dec[i, : len(d)] = np.stack(d)
            gen[c] = list(o.outputs[0].token_ids)
            finish17[i] = o.outputs[0].finish_reason == "stop"
        if (k // args.chunk) % 10 == 0:
            print(f"capture {k + len(cids)}/{n} {time.time() - t0:.0f}s", flush=True)
    t_cap = time.time() - t0

    sp_cont = SamplingParams(temperature=0.0, max_tokens=512 - 17, skip_special_tokens=False)
    todo = [c for c in order if c in gen and not finish17[order.index(c)]]
    t0 = time.time()
    cont = {}
    for k in range(0, len(todo), 64):
        part = todo[k:k + 64]
        out = llm.generate([{"prompt_token_ids": ids[c] + gen[c]} for c in part], sp_cont, use_hook=False)
        for c, o in zip(part, out):
            cont[c] = list(o.outputs[0].token_ids)
    t_cont = time.time() - t0
    texts = [{"case_id": c, "text": tok.decode(gen.get(c, []) + cont.get(c, []), skip_special_tokens=False),
              "n_generated": len(gen.get(c, [])) + len(cont.get(c, [])), "skipped_too_long": c in too_long}
             for c in order]
    np.savez(args.out + ".npz", case_id=np.array(order), **{f"prefill_layer_{args.layer}": pre,
             f"dec_layer_{args.layer}": dec}, n_prompt=np.array([len(ids[c]) for c in order]), finish17=finish17)
    Path(args.out + ".texts.jsonl").write_text("".join(json.dumps(t) + "\n" for t in texts))
    print(f"done capture {t_cap:.0f}s continuation {t_cont:.0f}s ({len(todo)} continued), "
          f"gpu {torch.cuda.get_device_name(0)}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
