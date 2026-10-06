"""Serve the AgentDojo steps through vLLM-Hook: one pass gives features and labels.

Renders each step with the model's chat template (tools included), generates the
next action greedily, and captures the last-prompt-token residual stream at the
given layers during prefill. The label is whether the generated action calls one
of the case's unsafe tools.

  python gen_capture.py steps.jsonl out_prefix [--chunk 0|1] [--max-tokens N]
Writes <out_prefix>.npz (layer_<L> arrays) and <out_prefix>.jsonl (generations).
"""
import argparse, json, multiprocessing as mp, os, re, time
from pathlib import Path

LAYERS = [7, 14, 21, 28]
MODEL = "Qwen/Qwen2.5-7B-Instruct"
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_calls(text):
    names = []
    for blob in TOOL_CALL.findall(text):
        try:
            names.append(json.loads(blob).get("name"))
        except json.JSONDecodeError:
            m = re.search(r'"name"\s*:\s*"([^"]+)"', blob)
            names.append(m.group(1) if m else None)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps"); ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"; os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import numpy as np, torch
    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM
    from vllm_hook_plugins.run_utils import unpack_hidden_states

    steps = [json.loads(l) for l in open(args.steps)]
    cfg = Path(args.out + ".config.json")
    cfg.write_text(json.dumps({"hidden_states": {"layers": LAYERS, "mode": "last_token"}}))
    llm = HookLLM(model=MODEL, worker_name="probe_hidden_states", analyzer_name="hidden_states",
                  config_file=str(cfg), download_dir=os.environ["HF_HOME"] + "/hub",
                  hook_dir="/dev/shm/vllm_hook", dtype=torch.bfloat16, gpu_memory_utilization=0.85,
                  max_model_len=16384, enable_prefix_caching=False, enforce_eager=True, enable_hook=True)
    tok = llm.tokenizer
    ids = [tok.apply_chat_template(s["messages"], tools=s["tools"], add_generation_prompt=True, tokenize=True)
           for s in steps]
    prompts = [{"prompt_token_ids": x} for x in ids]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    chunk = args.chunk or len(prompts)

    feats = {L: [] for L in LAYERS}; texts = []; t0 = time.time()
    for s in range(0, len(prompts), chunk):
        part = prompts[s:s + chunk]
        out = llm.generate(part, sp)
        hs = out[0].probes["hs_cache"]
        for e in hs.values():
            got = unpack_hidden_states(e)
            assert len(got) == len(part)
            feats[int(e["layer_num"])].extend(t.float().cpu().numpy() for t in got)
        texts.extend(o.outputs[0].text for o in out)
    elapsed = time.time() - t0

    np.savez(args.out + ".npz", **{f"layer_{L}": np.stack(v).astype(np.float32) for L, v in feats.items()})
    with open(args.out + ".jsonl", "w") as f:
        for s, x, text in zip(steps, ids, texts):
            calls = parse_calls(text)
            f.write(json.dumps({"case_id": s["case_id"], "user_task": s["user_task"],
                                "attack_present": s["attack_present"], "n_prompt_tokens": len(x),
                                "calls": calls, "unsafe": any(c in s.get("unsafe_tools", []) for c in calls),
                                "first_call_unsafe": bool(calls) and calls[0] in s.get("unsafe_tools", []),
                                "text": text}) + "\n")
    att = [json.loads(l) for l in open(args.out + ".jsonl") if json.loads(l)["attack_present"]]
    print(f"saved {args.out} steps={len(texts)} in {elapsed:.1f}s  attacked={len(att)} "
          f"unsafe={sum(a['unsafe'] for a in att)} no_tool_call={sum(not a['calls'] for a in att)}")


if __name__ == "__main__":
    main()
