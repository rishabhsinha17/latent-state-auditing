"""GPU capture overhead on real AgentDojo steps (Qwen2.5-7B-Instruct, layer 28).

64 fixed steps (every 28th of suite_full order), each generating exactly 32 tokens
(min_tokens = max_tokens = 32, ignore_eos), temperature 0.
  --mode mia    one MiaLLM(capture_hs) engine; per step, in rotated order:
                  off      generate(use_hook=False)
                  prefill  hooks on the prefill pass (what tool_call_risk / two_stage_probe use)
                  both     hooks on prefill and every decode pass (decision-time capture)
                then the same three as one batched call of all 64 steps, 3 repeats.
                Also times the two_stage_probe and tool_call_risk analyzers on the batched
                prefill capture (CPU side).
  --mode plain  plain vllm.LLM, same steps, bs1 and batched (baseline without vLLM-Hook).
The GPU is shared with another session; heavy runs are wrapped in flock by the caller.

Usage: bench_gpu.py <chat dir> <out.json> --mode mia|plain [--cascade cascade.json --probe probe.npz]
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
    ap.add_argument("--mode", choices=("mia", "plain"), required=True)
    ap.add_argument("--cascade", default=None)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.42)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    import torch
    from vllm import SamplingParams

    order = [json.loads(l)["case_id"] for l in open(args.chat / "suite_full.jsonl")]
    pick = order[::28][: args.n]
    want = set(pick)
    steps = {}
    for line in open(args.chat / "suite_steps.jsonl"):
        s = json.loads(line)
        if s["case_id"] in want:
            steps[s["case_id"]] = s

    if args.mode == "mia":
        from mia import MiaLLM
        cfg = Path(args.out).with_suffix(".infer.json")
        cfg.write_text(json.dumps({"model_info": {"name": MODEL},
                                   "hidden_states": {"layers": [28], "mode": "last_token"}}))
        llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="two_stage_probe",
                     config_file=str(cfg), hook_dir="/root/pinyu/mia_hook",
                     gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
        tok = llm.tokenizer
    else:
        from vllm import LLM
        llm = LLM(model=MODEL, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
        tok = llm.get_tokenizer()

    prompts = [{"prompt_token_ids": tok.apply_chat_template(steps[c]["messages"], tools=steps[c]["tools"],
                                                            add_generation_prompt=True, tokenize=True,
                                                            return_dict=False)} for c in pick]
    n_tok = [len(p["prompt_token_ids"]) for p in prompts]

    def sp(both=False):
        kw = dict(temperature=0.0, max_tokens=32, min_tokens=32, ignore_eos=True)
        if both:
            kw["extra_args"] = {"hooks_on": "both"}
        return SamplingParams(**kw)

    if args.mode == "mia":
        configs = {"off": (sp(), False), "prefill": (sp(), True), "both": (sp(True), True)}

        def run(name, batch):
            p, hook = configs[name]
            return llm.generate(batch, p, use_hook=hook)
    else:
        configs = {"plain": None}

        def run(name, batch):
            return llm.generate(batch, sp(), use_tqdm=False)

    names = list(configs)

    def timed(name, batch):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = run(name, batch)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, out

    for i in range(4):                       # warmup, not recorded
        for nm in names:
            timed(nm, [prompts[i]])
    bs1 = {nm: [] for nm in names}
    for i, p in enumerate(prompts):
        rot = names[i % len(names):] + names[: i % len(names)]
        for nm in rot:
            bs1[nm].append(timed(nm, [p])[0])
    batched = {nm: [] for nm in names}
    last = {}
    for r in range(3):
        rot = names[r % len(names):] + names[: r % len(names)]
        for nm in rot:
            t, out = timed(nm, prompts)
            batched[nm].append(t)
            last[nm] = out

    res = {"mode": args.mode, "n_steps": len(prompts), "prompt_tokens_median": float(np.median(n_tok)),
           "prompt_tokens_max": int(max(n_tok)), "generated_tokens_per_step": 32,
           "gpu": torch.cuda.get_device_name(0),
           "bs1_seconds": {nm: v for nm, v in bs1.items()},
           "bs1_summary": {nm: {"median_s": round(float(np.median(v)), 4), "p90_s": round(float(np.percentile(v, 90)), 4)}
                           for nm, v in bs1.items()},
           "batched_seconds": batched,
           "batched_summary": {nm: {"median_s": round(float(np.median(v)), 3)} for nm, v in batched.items()}}

    if args.mode == "mia" and args.cascade:
        from mia.analyzers.tool_call_risk_analyzer import ToolCallRiskAnalyzer
        from mia.analyzers.two_stage_probe_analyzer import TwoStageProbeAnalyzer
        probes = last["prefill"][0].probes
        an2, an1 = TwoStageProbeAnalyzer("/root/pinyu/mia_hook", {}), ToolCallRiskAnalyzer("/root/pinyu/mia_hook", {})
        t2, t1 = [], []
        for _ in range(20):
            t0 = time.perf_counter(); o2 = an2.analyze(analyzer_spec={"cascade_path": args.cascade}, probes=probes)
            t2.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); o1 = an1.analyze(analyzer_spec={"probe_path": args.probe}, probes=probes)
            t1.append(time.perf_counter() - t0)
        res["analyzer_ms_for_batch"] = {"two_stage_probe": round(1e3 * float(np.median(t2)), 3),
                                        "tool_call_risk": round(1e3 * float(np.median(t1)), 3)}
        res["analyzer_batch_verdicts"] = {"two_stage_probe_flags": sum(v == "flag" for v in o2["verdicts"]),
                                          "tool_call_risk_flags": sum(v == "flag" for v in o1["verdicts"]),
                                          "n": len(o2["verdicts"])}
        res["case_ids"] = pick
        res["two_stage_stage1_probs"] = o2["stage1"]["probabilities"]
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("bs1_summary", "batched_summary")}))
    if "analyzer_ms_for_batch" in res:
        print(json.dumps({"analyzer_ms_for_batch": res["analyzer_ms_for_batch"], **res["analyzer_batch_verdicts"]}))


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
