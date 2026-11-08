"""GPU: sequential vs parallel reading of per-request captures (merge_probes=False), r3.

12 chunks of 32 steps, 17 greedy tokens, hooks_on=both, rotated order:
  off               capture off
  seq               generate(merge_probes=False) then [o.probes for o in out]
  par16             generate(merge_probes=False) then MiaLLM.collect_probes(out, 16)
Correctness: on every chunk, the decode-mean features (layer 28, tokens 1..16) and prompt states
from seq and par16 are compared (max abs difference) on the same requests.
Usage (pod): bench_collect_gpu.py <chat dir> <out.json>
"""
import json, sys, time
from pathlib import Path
import numpy as np
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    import torch
    from vllm import SamplingParams
    from mia import MiaLLM
    from mia.analyzers.two_stage_probe_analyzer import request_features
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
    feats = {}

    def off(p):
        llm.generate(p, plain, use_hook=False)

    def seq(p):
        feats["seq"] = [o for o in [o.probes for o in llm.generate(p, both, merge_probes=False)]]

    def par(p):
        feats["par"] = MiaLLM.collect_probes(llm.generate(p, both, merge_probes=False), 16)

    runs = {"off": off, "seq": seq, "par16": par}

    def timed(fn, p):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(p); torch.cuda.synchronize()
        return time.perf_counter() - t0

    chunks = [order[i:i + 32] for i in range(32 * 20, 32 * 32, 32)]   # 12 chunks not used by the first bench
    for nm in runs:
        timed(runs[nm], P(chunks[0][:4]))
    res = {"batched32": {k: [] for k in runs}, "max_abs_diff_decode_mean": [], "max_abs_diff_prompt": [], "n_passes": []}
    names = list(runs)
    for k, ids in enumerate(chunks):
        p = P(ids)
        for nm in names[k % 3:] + names[: k % 3]:
            res["batched32"][nm].append(timed(runs[nm], p))
        a = np.stack([request_features(x, 28, "decode_mean", 16) for x in feats["seq"]])
        b = np.stack([request_features(x, 28, "decode_mean", 16) for x in feats["par"]])
        a0 = np.stack([request_features(x, 28) for x in feats["seq"]])
        b0 = np.stack([request_features(x, 28) for x in feats["par"]])
        res["max_abs_diff_decode_mean"].append(float(np.abs(a - b).max()))
        res["max_abs_diff_prompt"].append(float(np.abs(a0 - b0).max()))
        res["n_passes"] += [len(next(iter(x["hs_cache"].values()))["hidden_states"]) for x in feats["par"]]
    import statistics as st
    res["summary_median_s"] = {k: round(st.median(v), 4) for k, v in res["batched32"].items()}
    Path(out_path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary_median_s"]), "max diff decode", max(res["max_abs_diff_decode_mean"]),
          "prompt", max(res["max_abs_diff_prompt"]), "passes", sorted(set(res["n_passes"])))


if __name__ == "__main__":
    import multiprocessing as mp, os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
