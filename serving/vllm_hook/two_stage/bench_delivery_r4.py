"""GPU (r4): does MIA_APERTURE_DELIVERY=drain cut the per-request decode-delivery cost?

Run twice with the engine's delivery mode set by the environment (default HYBRID, then drain).
Qwen2.5-7B-Instruct, layer 28, the same 12 chunks of 32 steps as r3's bench_poll_gpu.py,
17 greedy tokens, rotated per chunk: off (capture off) and decision (hooks_on=both,
merge_probes=False, every request's capture read). Saves per-chunk times and, for the first
3 chunks, each request's decode-mean feature and generated token ids so the two modes'
captures can be compared on requests that generated the same tokens.
Usage (pod): MIA_APERTURE_DELIVERY=<mode> bench_delivery_r4.py <chat dir> <out prefix>
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    import torch
    from vllm import SamplingParams
    from mia import MiaLLM
    from mia.analyzers.two_stage_probe_analyzer import request_features
    chat, prefix = Path(sys.argv[1]), sys.argv[2]
    order = [json.loads(l)["case_id"] for l in open(chat / "suite_full.jsonl")]
    steps = {json.loads(l)["case_id"]: json.loads(l) for l in open(chat / "suite_steps.jsonl")}
    cfg = Path(prefix + ".infer.json")
    cfg.write_text(json.dumps({"model_info": {"name": MODEL}, "hidden_states": {"layers": [28], "mode": "last_token"}}))
    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="two_stage_probe", config_file=str(cfg),
                 hook_dir="/root/pinyu/mia_hook", gpu_memory_utilization=0.85, max_model_len=16384)
    both = SamplingParams(temperature=0.0, max_tokens=17, extra_args={"hooks_on": "both"})
    plain = SamplingParams(temperature=0.0, max_tokens=17)
    P = lambda ids: [{"prompt_token_ids": llm.tokenizer.apply_chat_template(steps[c]["messages"], tools=steps[c]["tools"],
                      add_generation_prompt=True, tokenize=True, return_dict=False)} for c in ids]
    keep = {}

    def decision(p, k=None):
        out = llm.generate(p, both, merge_probes=False)
        probes = [o.probes for o in out]
        if k is not None and k < 3:
            keep[k] = ([request_features(x, 28, "decode_mean", 16) for x in probes], [list(o.outputs[0].token_ids) for o in out])

    def timed(fn):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        return time.perf_counter() - t0

    chunks = [order[i:i + 32] for i in range(32 * 20, 32 * 32, 32)]
    timed(lambda: decision(P(chunks[0][:4])))
    res = {"off": [], "decision": []}
    for k, ids in enumerate(chunks):
        p = P(ids)
        seq = [("off", lambda: llm.generate(p, plain, use_hook=False)), ("decision", lambda: decision(p, k))]
        for nm, fn in (seq if k % 2 == 0 else seq[::-1]):
            res[nm].append(timed(fn))
    import statistics as st
    summ = {k: round(st.median(v), 4) for k, v in res.items()}
    mode = os.environ.get("MIA_APERTURE_DELIVERY", "default(hybrid)")
    Path(prefix + ".json").write_text(json.dumps({"mode": mode, "chunk_s": res, "median_s": summ}, indent=1))
    np.savez(prefix + "_feats.npz", **{f"f{k}": np.stack(v[0]) for k, v in keep.items()},
             **{f"t{k}": np.array([json.dumps(t) for t in v[1]]) for k, v in keep.items()})
    print(mode, json.dumps(summ))


if __name__ == "__main__":
    import multiprocessing as mp
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
