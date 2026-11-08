"""GPU: where does decision-mode latency go? (r3 decomposition, A40, vLLM 0.32)

Same 12 chunks of 32 steps as e2e_decision_gpu.py's timing pass, 17 greedy tokens each, rotated:
  off            capture off
  both_merged    hooks_on=both, default merge (decode rows captured, never delivered)
  both_unmerged  hooks_on=both, merge_probes=False (every request's decode rows delivered)
Then 32 single requests (bs1), rotated, off vs both_unmerged.
Usage (pod): bench_decision_gpu.py <chat dir> <out.json>
"""
import json, sys, time
from pathlib import Path
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    import torch
    from vllm import SamplingParams
    from mia import MiaLLM
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
    runs = {"off": lambda p: llm.generate(p, plain, use_hook=False),
            "both_merged": lambda p: llm.generate(p, both),
            "both_unmerged": lambda p: [o.probes for o in llm.generate(p, both, merge_probes=False)]}

    def timed(fn, p):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(p); torch.cuda.synchronize()
        return time.perf_counter() - t0

    chunks = [order[i:i + 32] for i in range(0, 32 * 12, 32)]
    for nm in runs:
        timed(runs[nm], P(chunks[0][:4]))
    res = {"batched32": {k: [] for k in runs}, "bs1": {"off": [], "both_unmerged": []}}
    names = list(runs)
    for k, ids in enumerate(chunks):
        p = P(ids)
        for nm in names[k % 3:] + names[: k % 3]:
            res["batched32"][nm].append(timed(runs[nm], p))
    for i, c in enumerate(order[::57][:32]):
        p = P([c])
        for nm in (("off", "both_unmerged") if i % 2 == 0 else ("both_unmerged", "off")):
            res["bs1"][nm].append(timed(runs[nm], p))
    import statistics as st
    res["summary_median_s"] = {g: {k: round(st.median(v), 4) for k, v in d.items()} for g, d in res.items() if g != "summary_median_s"}
    Path(out_path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary_median_s"]))


if __name__ == "__main__":
    import multiprocessing as mp, os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
