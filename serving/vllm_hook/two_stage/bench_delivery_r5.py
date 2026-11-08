"""GPU (r5): what does reading every request's decode rows cost, with the prefix cache controlled?

The r3 and r4 benches rotated modes over the same prompts with vLLM's prefix cache on, so the
first mode on each chunk paid the cold prefill and the rest did not (r4 split that 6/6 between
its two modes). Here every chunk is first run once untimed, so all timed runs are warm, then
timed twice per mode in the order off, prefill, decision, decision, prefill, off.
Modes (Qwen2.5-7B-Instruct, layer 28, 17 greedy tokens, the same 12 chunks of 32 steps as r3/r4):
  off       capture off
  prefill   capture at prefill, merged probes (what the prompt-only cascade needs)
  decision  capture at prefill and every decode pass, merge_probes=False, every request's
            decode_mean feature read (what the decision cascade needs)
Reports per-chunk medians and the median over chunks of each mode's paired difference from off.
Run once per capture directory (MIA_APERTURE_DIR) to see whether the spill to disk sets the cost.
Usage (pod): [MIA_APERTURE_DIR=...] bench_delivery_r5.py <chat dir> <out json>
"""
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    import torch
    from vllm import SamplingParams
    from mia import MiaLLM
    from mia.analyzers.two_stage_probe_analyzer import request_features
    chat, out_path = Path(sys.argv[1]), sys.argv[2]
    order = [json.loads(l)["case_id"] for l in open(chat / "suite_full.jsonl")]
    steps = {json.loads(l)["case_id"]: json.loads(l) for l in open(chat / "suite_steps.jsonl")}
    cfg = Path(out_path + ".infer.json")
    cfg.write_text(json.dumps({"model_info": {"name": MODEL}, "hidden_states": {"layers": [28], "mode": "last_token"}}))
    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="two_stage_probe", config_file=str(cfg),
                 hook_dir="/root/pinyu/mia_hook_b5", gpu_memory_utilization=0.85, max_model_len=16384)
    both = SamplingParams(temperature=0.0, max_tokens=17, extra_args={"hooks_on": "both"})
    pre = SamplingParams(temperature=0.0, max_tokens=17, extra_args={"hooks_on": "prefill"})
    plain = SamplingParams(temperature=0.0, max_tokens=17)
    P = lambda ids: [{"prompt_token_ids": llm.tokenizer.apply_chat_template(steps[c]["messages"], tools=steps[c]["tools"],
                      add_generation_prompt=True, tokenize=True, return_dict=False)} for c in ids]
    runs = {"off": lambda p: llm.generate(p, plain, use_hook=False),
            "prefill": lambda p: llm.generate(p, pre)[0].probes,
            "decision": lambda p: [request_features(o.probes, 28, "decode_mean", 16)
                                   for o in llm.generate(p, both, merge_probes=False)]}

    def timed(fn, p):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(p); torch.cuda.synchronize()
        return time.perf_counter() - t0

    chunks = [order[i:i + 32] for i in range(32 * 20, 32 * 32, 32)]
    for nm in runs:
        timed(runs[nm], P(chunks[0][:4]))
    res = {k: [] for k in runs}
    for ids in chunks:
        p = P(ids)
        timed(runs["off"], p)                              # warm the prefix cache, untimed
        got = {k: [] for k in runs}
        for nm in ("off", "prefill", "decision", "decision", "prefill", "off"):
            got[nm].append(timed(runs[nm], p))
        for k in runs:
            res[k].append(got[k])
    med = {k: [st.median(v) for v in res[k]] for k in runs}
    summary = {"median_s": {k: round(st.median(v), 4) for k, v in med.items()},
               "median_paired_diff_from_off_s": {k: round(st.median([a - b for a, b in zip(med[k], med["off"])]), 4)
                                                 for k in ("prefill", "decision")},
               "aperture_dir": os.environ.get("MIA_APERTURE_DIR", "./hs_aperture_dump (default)"),
               "gpu": torch.cuda.get_device_name(0)}
    Path(out_path).write_text(json.dumps({"protocol": __doc__, "chunk_runs_s": res, "summary": summary}, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
