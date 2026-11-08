"""GPU end-to-end: two_stage_probe and tool_call_risk analyzers on every AgentDojo step, batched.

MiaLLM(capture_hs, layer 28), prefill capture, max_tokens=1, chunks of --chunk prompts in one
generate call each; both analyzers read out[0].probes for the chunk. Saves per-step stage
probabilities and verdicts so they can be compared on CPU with (a) the bs1 decode-run prefill
states and (b) the stored training features. This is integration evidence on vLLM 0.32, not a
held-out metric (the exported probes were fit on all steps).

Usage (pod): e2e_cascade_gpu.py <chat dir> <cascade.json> <stage1 probe.npz> <out.json> [--chunk 32]
"""
import argparse
import json
import time
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", type=Path)
    ap.add_argument("cascade")
    ap.add_argument("probe")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.42)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()

    import torch
    from vllm import SamplingParams
    from mia import MiaLLM
    from mia.analyzers.tool_call_risk_analyzer import ToolCallRiskAnalyzer
    from mia.analyzers.two_stage_probe_analyzer import TwoStageProbeAnalyzer

    order = [json.loads(l)["case_id"] for l in open(args.chat / "suite_full.jsonl")]
    steps = {}
    for line in open(args.chat / "suite_steps.jsonl"):
        s = json.loads(line)
        steps[s["case_id"]] = s
    cfg = Path(args.out).with_suffix(".infer.json")
    cfg.write_text(json.dumps({"model_info": {"name": MODEL},
                               "hidden_states": {"layers": [28], "mode": "last_token"}}))
    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="two_stage_probe",
                 config_file=str(cfg), hook_dir="/root/pinyu/mia_hook",
                 gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
    two, single = TwoStageProbeAnalyzer("/root/pinyu/mia_hook", {}), ToolCallRiskAnalyzer("/root/pinyu/mia_hook", {})
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    res = {"case_id": [], "stage1": [], "stage2": [], "verdict": [], "tcr_prob": [], "tcr_verdict": [], "chunk_s": []}
    for c0 in range(0, len(order), args.chunk):
        ids = order[c0:c0 + args.chunk]
        prompts = [{"prompt_token_ids": llm.tokenizer.apply_chat_template(
            steps[c]["messages"], tools=steps[c]["tools"], add_generation_prompt=True, tokenize=True,
            return_dict=False)} for c in ids]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = llm.generate(prompts, sp)
        torch.cuda.synchronize(); res["chunk_s"].append(time.perf_counter() - t0)
        probes = out[0].probes
        o2 = two.analyze(analyzer_spec={"cascade_path": args.cascade, "score_all": True}, probes=probes)
        o1 = single.analyze(analyzer_spec={"probe_path": args.probe}, probes=probes)
        assert len(o2["verdicts"]) == len(ids) == len(o1["verdicts"]), (len(o2["verdicts"]), len(ids))
        res["case_id"] += ids
        res["stage1"] += o2["stage1"]["probabilities"]
        res["stage2"] += o2["stage2"]["probabilities"]
        res["verdict"] += o2["verdicts"]
        res["tcr_prob"] += o1["probabilities"]
        res["tcr_verdict"] += o1["verdicts"]
        if c0 % (args.chunk * 10) == 0:
            print(f"{c0 + len(ids)}/{len(order)} chunk {res['chunk_s'][-1]:.2f}s flags {o2['verdicts'].count('flag')}", flush=True)
    res["thresholds"] = {"stage1": o2["stage1"]["threshold"], "stage2": o2["stage2"]["threshold"],
                         "tool_call_risk": o1["threshold"]}
    Path(args.out).write_text(json.dumps(res))
    print("done", len(res["case_id"]), "flags", res["verdict"].count("flag"))


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
