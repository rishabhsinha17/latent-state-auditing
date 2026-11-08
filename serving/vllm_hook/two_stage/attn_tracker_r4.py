"""GPU (r4): vLLM-Hook's existing Attention Tracker analyzer as a stage-2 presence signal (Granite).

Granite-3.1-8B-Instruct with vLLM-Hook's shipped config (model_configs/attention_tracker/
granite-3.1-8b-instruct.json, capture_qk, important heads). Every step is rendered exactly as in
gen_capture_r4.py (--adapter granite). Instruction span = tokens of the user task message;
data span = tokens after the user turn up to the generation prompt. attn_tracker returns the
heads' normalized attention of the last prompt token on the instruction ("sum_normalize");
lower means more attention went to the data. Steps whose chat-template ids differ from the
offset-mapped tokenization are skipped and counted.
Usage (pod): attn_tracker_r4.py <chat dir> <vllm-hook dir> <out.json> [--chunk 16]
"""
import argparse
import json
import time
from pathlib import Path

MODEL = "ibm-granite/granite-3.1-8b-instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", type=Path)
    ap.add_argument("hook", type=Path)
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1,
                    help="score every stride-th step in suite_full order (r4 used 2: the full pass ran ~28 s per 16 steps)")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_capture_r4 import granite_messages
    from vllm import SamplingParams
    from mia import MiaLLM

    order = [json.loads(l)["case_id"] for l in open(args.chat / "suite_full.jsonl")][::args.stride]
    steps = {json.loads(l)["case_id"]: json.loads(l) for l in open(args.chat / "suite_steps.jsonl")}
    llm = MiaLLM(model=MODEL, worker_name="capture_qk", analyzer_name="attn_tracker",
                 config_file=str(args.hook / "model_configs/attention_tracker/granite-3.1-8b-instruct.json"),
                 hook_dir="/root/pinyu/mia_hook_qk", gpu_memory_utilization=0.85, max_model_len=16384,
                 enforce_eager=True)   # qk_capture="score" requires an eager engine
    tok = llm.tokenizer
    gen_suffix = len(tok("<|start_of_role|>assistant<|end_of_role|>", add_special_tokens=False)["input_ids"])
    prompts, ranges, kept, skipped = [], [], [], []
    for c in order:
        s = steps[c]
        msgs = granite_messages(s["messages"])
        ids = tok.apply_chat_template(msgs, tools=s["tools"], add_generation_prompt=True, tokenize=True, return_dict=False)
        text = tok.apply_chat_template(msgs, tools=s["tools"], add_generation_prompt=True, tokenize=False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        user = next(m["content"] for m in msgs if m["role"] == "user")
        u0 = text.find(user, text.find("<|start_of_role|>user<|end_of_role|>"))
        if list(enc["input_ids"]) != list(ids) or u0 < 0:
            skipped.append(c)
            continue
        u1 = u0 + len(user)
        offs = enc["offset_mapping"]
        inst = [i for i, (a, b) in enumerate(offs) if a < u1 and b > u0]
        after = next(i for i, (a, b) in enumerate(offs) if a >= u1)
        ranges.append(((inst[0], inst[-1] + 1), (after, len(ids) - gen_suffix)))
        prompts.append({"prompt_token_ids": ids})
        kept.append(c)
    print(f"kept {len(kept)} skipped {len(skipped)}", flush=True)
    # score mode: the listed heads' last-token attention scores, not full Q/K (the Q/K spill filled
    # an 80 GB pod disk after ~100 five-thousand-token steps). Same quantity Attention Tracker reads.
    sp = SamplingParams(temperature=0.0, max_tokens=1, extra_args={"qk_capture": "score"})
    scores = []
    t0 = time.time()
    for k in range(0, len(kept), args.chunk):
        out = llm.generate(prompts[k:k + args.chunk], sp)
        r = llm.analyze(analyzer_spec={"input_range": ranges[k:k + args.chunk], "attn_func": "sum_normalize"},
                        probes=out[0].probes)
        sc = r["score"] if isinstance(r["score"], list) else [r["score"]]
        assert len(sc) == len(prompts[k:k + args.chunk]), (len(sc), args.chunk)
        scores += [float(x) for x in sc]
        if (k // args.chunk) % 20 == 0:
            print(f"{k + args.chunk}/{len(kept)} {time.time() - t0:.0f}s", flush=True)
    Path(args.out).write_text(json.dumps({"case_id": kept, "score": scores, "skipped": skipped,
                                          "ranges": ranges, "attn_func": "sum_normalize"}))
    print("done", len(scores), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
