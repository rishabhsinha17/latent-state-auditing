"""GPU (r12): read the presence probe's input at every earlier step from ONE forward pass.

For a sample of r12 after-arrival steps (40 per model, fixed seed, prompts of at most 12,000 tokens),
capture the hidden state of every prompt token at the probe layer (capture_hs, hs_mode all_tokens,
prefix caching off so every row is computed, 2,048-token prefill chunks) and keep the rows that end
each assistant-turn header, the position where the model would start writing. Header t follows t
tool outputs, so its row is the one-pass stand-in for the prompt state of the same trajectory's
step at depth t. eval-side (boundary_r12.py) compares these rows with the separately captured
steps. Saves <out>.npz (case_id, depth, rows float16, n_prompt) and <out>.json (per step: positions,
how many headers were found).
Usage (pod): hs_boundary_r12.py <r12 chat dir> <model> <layer> <out prefix> [--adapter granite]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gen_capture_r4 import granite_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", type=Path)
    ap.add_argument("model")
    ap.add_argument("layer", type=int)
    ap.add_argument("out")
    ap.add_argument("--adapter", default="none", choices=("none", "granite"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=12000)
    args = ap.parse_args()

    from vllm import SamplingParams
    from mia import MiaLLM

    steps = [json.loads(l) for l in open(args.chat / "suite_steps.jsonl")]
    cfg = Path(args.out + ".infer.json")
    cfg.write_text(json.dumps({"model_info": {"name": args.model},
                               "hidden_states": {"layers": [args.layer], "mode": "all_tokens"}}))
    llm = MiaLLM(model=args.model, worker_name="capture_hs", analyzer_name="hidden_states", config_file=str(cfg),
                 hook_dir="/root/pinyu/mia_hook_b", gpu_memory_utilization=0.85, max_model_len=16384,
                 enable_prefix_caching=False, max_num_batched_tokens=2048)
    tok = llm.tokenizer
    render = granite_messages if args.adapter == "granite" else (lambda m: m)
    enc = lambda s, gen: tok.apply_chat_template(render(s["messages"]), tools=s["tools"], add_generation_prompt=gen,
                                                tokenize=True, return_dict=False)
    after = [s for s in steps if not s["arrival"]]
    ids = {s["case_id"]: enc(s, True) for s in after}
    ok = [s for s in after if len(ids[s["case_id"]]) <= args.max_tokens]
    pick = [ok[i] for i in sorted(np.random.default_rng(0).choice(len(ok), size=min(args.n, len(ok)), replace=False))]
    s0 = dict(pick[0], messages=pick[0]["messages"][:2])
    head = enc(s0, True)[len(enc(s0, False)):]          # the generation-prompt tokens = an assistant-turn header
    sp = SamplingParams(temperature=0.0, max_tokens=1, extra_args={"hooks_on": "prefill"})
    out_rows, meta = [], []
    for k in range(0, len(pick), 4):
        part = pick[k:k + 4]
        res = llm.generate([{"prompt_token_ids": ids[s["case_id"]]} for s in part], sp, merge_probes=False)
        for s, o in zip(part, res):
            full = ids[s["case_id"]]
            entry = next(e for e in o.probes["hs_cache"].values() if int(e["layer_num"]) == args.layer)
            H = np.concatenate([t.float().cpu().numpy().reshape(-1, t.shape[-1]) for t in entry["hidden_states"]], 0)
            ends = [p for p in range(len(head) - 1, len(full)) if full[p - len(head) + 1:p + 1] == head]
            rec = {"case_id": s["case_id"], "depth": s["depth"], "n_prompt": len(full), "rows_captured": int(H.shape[0]),
                   "headers": len(ends), "positions": ends}
            meta.append(rec)
            if H.shape[0] != len(full):
                continue
            for t, p in enumerate(ends):
                out_rows.append((s["case_id"], t, H[p]))
        print(f"{k + len(part)}/{len(pick)}", flush=True)
    np.savez(args.out + ".npz", case_id=np.array([r[0] for r in out_rows]), depth=np.array([r[1] for r in out_rows]),
             rows=np.stack([r[2] for r in out_rows]).astype(np.float16) if out_rows else np.zeros((0, 1), np.float16))
    Path(args.out + ".json").write_text(json.dumps(meta, indent=1))
    print("done", len(out_rows), "rows from", sum(m["rows_captured"] == m["n_prompt"] for m in meta), "of", len(meta), "steps",
          flush=True)


if __name__ == "__main__":
    main()
