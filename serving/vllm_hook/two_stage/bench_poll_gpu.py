"""GPU: is decision-mode delivery latency the 100 ms per-request poll in aperture_gather.wait_delivered?

12 chunks of 32 steps (same as bench_collect_gpu.py), 17 greedy tokens, hooks_on=both,
merge_probes=False, probes read one by one. Rotated per chunk:
  off        capture off
  poll100    default _WAIT_POLL_S = 0.1
  poll5      _WAIT_POLL_S patched to 0.005 in this (reading) process only
Usage (pod): bench_poll_gpu.py <chat dir> <out.json>
"""
import json, sys, time
from pathlib import Path
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    import torch
    from vllm import SamplingParams
    from mia import MiaLLM
    import mia.core.aperture.aperture_gather as ag
    chat, out_path = Path(sys.argv[1]), sys.argv[2]
    order = [json.loads(l)["case_id"] for l in open(chat / "suite_full.jsonl")]
    steps = {json.loads(l)["case_id"]: json.loads(l) for l in open(chat / "suite_steps.jsonl")}
    cfg = Path(out_path).with_suffix(".infer.json")
    cfg.write_text(json.dumps({"model_info": {"name": MODEL}, "hidden_states": {"layers": [28], "mode": "last_token"}}))
    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="two_stage_probe", config_file=str(cfg),
                 hook_dir="/root/pinyu/mia_hook", gpu_memory_utilization=0.8, max_model_len=16384)
    both = SamplingParams(temperature=0.0, max_tokens=17, extra_args={"hooks_on": "both"})
    plain = SamplingParams(temperature=0.0, max_tokens=17)
    P = lambda ids: [{"prompt_token_ids": llm.tokenizer.apply_chat_template(steps[c]["messages"], tools=steps[c]["tools"],
                      add_generation_prompt=True, tokenize=True, return_dict=False)} for c in ids]

    def read(poll):
        def run(p):
            ag._WAIT_POLL_S = poll
            try:
                [o.probes for o in llm.generate(p, both, merge_probes=False)]
            finally:
                ag._WAIT_POLL_S = 0.1
        return run

    runs = {"off": lambda p: llm.generate(p, plain, use_hook=False), "poll100": read(0.1), "poll5": read(0.005)}

    def timed(fn, p):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(p); torch.cuda.synchronize()
        return time.perf_counter() - t0

    chunks = [order[i:i + 32] for i in range(32 * 20, 32 * 32, 32)]
    for nm in runs:
        timed(runs[nm], P(chunks[0][:4]))
    res = {k: [] for k in runs}
    names = list(runs)
    for k, ids in enumerate(chunks):
        p = P(ids)
        for nm in names[k % 3:] + names[: k % 3]:
            res[nm].append(timed(runs[nm], p))
    import statistics as st
    out = {"batched32_s": res, "summary_median_s": {k: round(st.median(v), 4) for k, v in res.items()}}
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary_median_s"]))


if __name__ == "__main__":
    import multiprocessing as mp, os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
