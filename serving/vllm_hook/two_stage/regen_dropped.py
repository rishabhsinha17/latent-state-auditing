"""GPU. Regenerate, in full, the steps whose short vLLM 0.32 regeneration diverged from the label.

Same engine and sampling settings as capture_decision.py (MiaLLM capture_hs, layer 28,
hooks_on="both", greedy, one request per call) except max_tokens=512 (gen_capture.py's default),
so the first 17 tokens should reproduce the captured prefix. The texts are then labelled with
AgentDojo (chat/label_steps.py) for the r2 all-steps robustness analysis.

Usage (pod): regen_dropped.py <chat dir> <dec npz> <dec texts jsonl> <out jsonl>
"""
import argparse
import json
from pathlib import Path

import numpy as np

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", type=Path)
    ap.add_argument("dec")
    ap.add_argument("texts")
    ap.add_argument("out")
    ap.add_argument("--gpu-mem", type=float, default=0.42)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()

    from vllm import SamplingParams
    from mia import MiaLLM

    E = np.load(args.dec)
    prefix = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(args.texts)}
    todo = [c for c, s in zip(E["case_id"], E["same_action"]) if not s]
    steps = {}
    for line in open(args.chat / "suite_steps.jsonl"):
        s = json.loads(line)
        if s["case_id"] in set(todo):
            steps[s["case_id"]] = s
    cfg = Path(args.out).with_suffix(".infer.json")
    cfg.write_text(json.dumps({"model_info": {"name": MODEL},
                               "hidden_states": {"layers": [28], "mode": "last_token"}}))
    llm = MiaLLM(model=MODEL, worker_name="capture_hs", analyzer_name="hidden_states",
                 config_file=str(cfg), hook_dir="/root/pinyu/mia_hook",
                 gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
    sp = SamplingParams(temperature=0.0, max_tokens=512, extra_args={"hooks_on": "both"})
    n_match = 0
    with open(args.out, "w") as f:
        for i, c in enumerate(todo):
            s = steps[c]
            ids = llm.tokenizer.apply_chat_template(s["messages"], tools=s["tools"], add_generation_prompt=True,
                                                    tokenize=True, return_dict=False)
            text = llm.generate([{"prompt_token_ids": ids}], sp)[0].outputs[0].text
            match = text.lstrip().startswith(prefix[c].strip())
            n_match += match
            f.write(json.dumps({"case_id": c, "text": text, "prefix_matches_capture": bool(match)}) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(todo)} prefix matches so far {n_match}", flush=True)
    print(f"done {len(todo)} prefix matches capture: {n_match}")


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    os.environ.setdefault("MIA_APERTURE_GPU_BYTES", str(512 << 20))
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
