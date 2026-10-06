"""Serving cost of vLLM-Hook capture on real agent steps (one config per process).

  python bench.py steps.jsonl <config> [--max-tokens 64]
configs: vllm_eager, vllm_graphs, hook_off, hook_mem, hook_disk
Appends one JSON line to bench.jsonl.
"""
import argparse, json, multiprocessing as mp, os, time

LAYER = 21
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps"); ap.add_argument("config")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"; os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import torch
    from vllm import LLM, SamplingParams

    steps = [json.loads(l) for l in open(args.steps)]
    common = dict(dtype=torch.bfloat16, gpu_memory_utilization=0.85, max_model_len=16384,
                  enable_prefix_caching=False, download_dir=os.environ["HF_HOME"] + "/hub")
    if args.config.startswith("vllm"):
        llm = LLM(model=MODEL, enforce_eager=args.config == "vllm_eager", **common)
        tok = llm.get_tokenizer()
        gen = lambda p, sp: llm.generate(p, sp, use_tqdm=False)
    else:
        from vllm_hook_plugins import HookLLM
        cfg = f"/dev/shm/bench_{args.config}.json"
        json.dump({"hidden_states": {"layers": [LAYER], "mode": "last_token"}}, open(cfg, "w"))
        llm = HookLLM(model=MODEL, worker_name="probe_hidden_states", analyzer_name="hidden_states",
                      config_file=cfg, hook_dir="/dev/shm/vllm_hook", enforce_eager=True,
                      enable_hook=True, **common)
        tok = llm.tokenizer
        use = args.config != "hook_off"
        disk = args.config == "hook_disk"
        gen = lambda p, sp: llm.generate(p, sp, use_hook=use, save_to_disk=disk, use_tqdm=False)

    ids = [tok.apply_chat_template(s["messages"], tools=s["tools"], add_generation_prompt=True, tokenize=True)
           for s in steps]
    prompts = [{"prompt_token_ids": x} for x in ids]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    gen(prompts[:8], sp)  # warmup, also installs hooks
    torch.cuda.synchronize()
    t0 = time.time()
    out = gen(prompts, sp)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_in = sum(len(x) for x in ids); n_out = sum(len(o.outputs[0].token_ids) for o in out)
    row = {"config": args.config, "requests": len(prompts), "seconds": round(dt, 2),
           "req_per_s": round(len(prompts) / dt, 3), "prompt_tok_per_s": round(n_in / dt),
           "output_tok_per_s": round(n_out / dt), "max_tokens": args.max_tokens}
    print(json.dumps(row))
    with open("bench.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
