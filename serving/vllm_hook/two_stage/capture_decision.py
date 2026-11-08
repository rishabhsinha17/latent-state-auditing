"""GPU. Decision-time hidden states for a stage-2 probe (r2: in-memory capture, shared-GPU caps).

Stage 1 scores the prompt at prefill. A decision-time stage 2 scores the first generated tokens
of the action (the tool name and the start of its arguments), still before any tool runs. This
captures those states on the same AgentDojo steps as suite_full through vLLM-Hook's capture_hs
worker with hooks_on="both" and hs_mode="last_token", layer 28 only (the layer stage 1 reads).

One request per generate call, read from out[0].probes (no disk), so a call's per-pass list is
[prefill, decode_1, ..., decode_{K-1}] for K = --max-tokens generated tokens.
Saved (npz, float16): prefill_layer_28 (parity check against suite_full.npz), dec_layer_28
(n, K-1, hidden), NaN where generation stopped early; n_passes, n_generated; per-step wall time.
Texts go to <out>.texts.jsonl; same_action marks steps whose short greedy regeneration is a
prefix of the labelled generation (eval drops the rest).

Usage (pod): python capture_decision.py <chat dir> <out.npz> [--limit N] [--gpu-mem 0.42]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", type=Path)
    ap.add_argument("out")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--max-tokens", type=int, default=17)   # prefill + 16 decode passes
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.42)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()

    import torch
    from vllm import SamplingParams
    from mia import MiaLLM
    from mia.artifacts import unpack_hidden_states

    full = [json.loads(l) for l in open(args.chat / "suite_full.jsonl")]
    order = [g["case_id"] for g in full][: args.limit]
    want = set(order)
    steps = {}
    for line in open(args.chat / "suite_steps.jsonl"):
        s = json.loads(line)
        if s["case_id"] in want:
            steps[s["case_id"]] = s
    assert len(steps) == len(order), f"{len(order) - len(steps)} steps missing from suite_steps.jsonl"

    cfg = Path(args.out).with_suffix(".infer.json")
    cfg.write_text(json.dumps({"model_info": {"name": MODEL},
                               "hidden_states": {"layers": [args.layer], "mode": "last_token"}}))
    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="hidden_states",
                 config_file=str(cfg), hook_dir="/root/pinyu/mia_hook",
                 gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, extra_args={"hooks_on": "both"})

    J = args.max_tokens - 1
    n = len(order)
    pre = dec = None
    n_passes = np.zeros(n, np.int32)
    n_gen = np.zeros(n, np.int32)
    wall = np.zeros(n, np.float64)
    texts = []
    for i, cid in enumerate(order):
        s = steps[cid]
        ids = llm.tokenizer.apply_chat_template(s["messages"], tools=s["tools"], add_generation_prompt=True,
                                                tokenize=True, return_dict=False)
        t0 = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": ids}], sp)
        torch.cuda.synchronize()
        wall[i] = time.perf_counter() - t0
        texts.append(out[0].outputs[0].text)
        n_gen[i] = len(out[0].outputs[0].token_ids)
        cache = out[0].probes["hs_cache"]
        entries = [e for e in cache.values() if int(e["layer_num"]) == args.layer]
        assert len(entries) == 1, f"{cid}: expected one layer-{args.layer} entry, got {len(entries)}"
        passes = [t.float().cpu().numpy().reshape(-1) for t in unpack_hidden_states(entries[0])]
        n_passes[i] = len(passes)
        if pre is None:
            H = passes[0].size
            pre = np.full((n, H), np.nan, np.float16)
            dec = np.full((n, J, H), np.nan, np.float16)
        pre[i] = passes[0]
        d = passes[1:1 + J]
        if d:
            dec[i, : len(d)] = np.stack(d)
        if i < 3 or i % 200 == 0:
            print(f"[{i}/{n}] {cid}: {len(passes)} passes, {n_gen[i]} generated tokens, {wall[i]:.2f}s", flush=True)

    labelled = {g["case_id"]: g["text"] for g in full}
    same = np.array([labelled[c].lstrip().startswith(t.strip()) for c, t in zip(order, texts)])
    print(f"same action as the labelled generation: {same.sum()} of {n}")
    print(f"passes == generated tokens: {(n_passes == n_gen).sum()} of {n}")
    print(f"wall per step (s): median {np.median(wall):.3f}, p90 {np.percentile(wall, 90):.3f}")
    np.savez(args.out, case_id=np.array(order), same_action=same, n_passes=n_passes, n_generated=n_gen,
             wall_s=wall, **{f"prefill_layer_{args.layer}": pre, f"dec_layer_{args.layer}": dec})
    Path(args.out).with_suffix(".texts.jsonl").write_text(
        "".join(json.dumps({"case_id": c, "text": t}) + "\n" for c, t in zip(order, texts)))


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
