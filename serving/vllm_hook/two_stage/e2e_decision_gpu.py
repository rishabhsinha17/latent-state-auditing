"""GPU: serve the decision-time cascade in batches through MiaLLM + two_stage_probe (r3).

Chunks of --chunk prompts, SamplingParams(temperature=0, max_tokens=17,
extra_args={"hooks_on": "both"}), generate(..., merge_probes=False), then the analyzer on
probes=[o.probes for o in out] with the exported decision cascade (stage 2 = presence x decision,
decision probe position "decode_mean"). Records per-request pass counts, the generated text,
stage scores and verdicts for every step, and per-chunk wall times. Then, on the first
--timing-chunks chunks, times the same generation with capture off and with prefill capture
merged (the prompt-only cascade path), rotating the order.
Compared on CPU afterwards with the exported cascade's verdicts on the bs1 r2 capture.

Usage (pod): e2e_decision_gpu.py <chat dir> <decision cascade.json> <out.json> [--chunk 32]
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
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--timing-chunks", type=int, default=12)
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()

    import torch
    from vllm import SamplingParams
    from mia import MiaLLM

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
    sp_both = SamplingParams(temperature=0.0, max_tokens=17, extra_args={"hooks_on": "both"})
    sp_plain = SamplingParams(temperature=0.0, max_tokens=17)
    chunks = [order[i:i + args.chunk] for i in range(0, len(order), args.chunk)]

    def prompts_of(ids):
        return [{"prompt_token_ids": llm.tokenizer.apply_chat_template(
            steps[c]["messages"], tools=steps[c]["tools"], add_generation_prompt=True, tokenize=True,
            return_dict=False)} for c in ids]

    def timed(fn):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t0, r

    timed(lambda: llm.generate(prompts_of(chunks[0][:4]), sp_both, merge_probes=False))   # warmup
    res = {"case_id": [], "n_passes": [], "n_generated": [], "text": [], "stage1": [], "stage2": [],
           "verdict": [], "gen_s": [], "analyze_s": []}
    for k, ids in enumerate(chunks):
        p = prompts_of(ids)
        tg, out = timed(lambda: llm.generate(p, sp_both, merge_probes=False))
        t0 = time.perf_counter()
        reqs = [o.probes for o in out]
        r = llm.analyze(analyzer_spec={"cascade_path": args.cascade, "score_all": True}, probes=reqs)
        res["analyze_s"].append(time.perf_counter() - t0)
        res["gen_s"].append(tg)
        assert len(r["verdicts"]) == len(ids)
        for c, o, pr in zip(ids, out, reqs):
            entry = next(e for e in pr["hs_cache"].values() if int(e["layer_num"]) == 28)
            res["case_id"].append(c)
            res["n_passes"].append(len(entry["hidden_states"]))
            res["n_generated"].append(len(o.outputs[0].token_ids))
            res["text"].append(o.outputs[0].text)
        res["stage1"] += r["stage1"]["probabilities"]
        res["stage2"] += r["stage2"]["probabilities"]
        res["verdict"] += r["verdicts"]
        if k % 10 == 0:
            print(f"chunk {k}/{len(chunks)} gen {tg:.2f}s analyze {res['analyze_s'][-1] * 1e3:.1f}ms "
                  f"passes {sorted(set(res['n_passes'][-len(ids):]))}", flush=True)

    timing = {"decision_both_unmerged": [], "off": [], "prefill_merged": []}
    for k, ids in enumerate(chunks[: args.timing_chunks]):
        p = prompts_of(ids)
        runs = {"off": lambda: llm.generate(p, sp_plain, use_hook=False),
                "prefill_merged": lambda: llm.generate(p, sp_plain),
                "decision_both_unmerged": lambda: [o.probes for o in llm.generate(p, sp_both, merge_probes=False)]}
        names = list(runs)
        for nm in names[k % 3:] + names[: k % 3]:
            timing[nm].append(timed(runs[nm])[0])
    res["timing_chunk_s"] = timing
    res["chunk"] = args.chunk
    res["gpu"] = torch.cuda.get_device_name(0)
    Path(args.out).write_text(json.dumps(res))
    import statistics as st
    print("done", len(res["case_id"]), "flags", res["verdict"].count("flag"),
          "passes==17:", sum(n == 17 for n in res["n_passes"]),
          {k: round(st.median(v), 3) for k, v in timing.items()})


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
