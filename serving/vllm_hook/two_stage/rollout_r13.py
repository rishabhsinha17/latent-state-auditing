"""GPU (r13): model-driven AgentDojo rollouts, captured at every step.

Each v1 user task runs once with no injection, and once per injection task with AgentDojo's
<attack> template (env_r13.py init). In round t every unfinished trajectory is one step: its messages
are rendered and captured exactly as gen_capture_r4.py does it (prompt state and decode states 1..16 at
<layer>, then the same generation continued to 512 tokens). env_r13.py then parses the action, runs its
calls in AgentDojo and appends them, in the step builders' message format. So each step's history is
the model's own earlier calls and their real outputs, not the ground truth. Up to <max rounds> actions.
Saves <out>.npz (case_id, prefill_layer_L, dec_layer_L, n_prompt, finish17), <out>.steps.jsonl (per
step: case_id, traj, round, suite, user_task, injection_task, attack, injections, markers, messages),
<out>.texts.jsonl (case_id, text, skipped_too_long) and the final state.
Usage (pod): rollout_r13.py <env python> <model> <layer> <out prefix> <attack> [--adapter granite]
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from gen_capture_r4 import granite_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env_python")
    ap.add_argument("model")
    ap.add_argument("layer", type=int)
    ap.add_argument("out")
    ap.add_argument("attack")
    ap.add_argument("--adapter", default="none", choices=("none", "granite"))
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: keep every k-th trajectory, about this many")
    args = ap.parse_args()

    import torch
    from vllm import SamplingParams
    from mia import MiaLLM

    work = Path(args.out + "_state")
    work.mkdir(parents=True, exist_ok=True)
    # -s -P instead of -I so that PYTHONHASHSEED=0 applies (-I ignores it) and AgentDojo's set-valued lists
    # come out in the same order on every run; the script directory and user site stay off sys.path.
    env = lambda *a: subprocess.run([args.env_python, "-s", "-P", "env_r13.py", *map(str, a)], check=True,
                                    env={**os.environ, "PYTHONHASHSEED": "0"})
    env("init", work / "state_0.jsonl", args.attack)
    if args.limit:
        rows0 = (work / "state_0.jsonl").read_text().splitlines(True)
        (work / "state_0.jsonl").write_text("".join(rows0[:: max(1, len(rows0) // args.limit)]))
    cfg = Path(args.out + ".infer.json")
    cfg.write_text(json.dumps({"model_info": {"name": args.model},
                               "hidden_states": {"layers": [args.layer], "mode": "last_token"}}))
    llm = MiaLLM(model=args.model, worker_name="capture_hs", analyzer_name="hidden_states", config_file=str(cfg),
                 hook_dir="/root/pinyu/mia_hook_r", gpu_memory_utilization=0.85, max_model_len=args.max_model_len)
    tok = llm.tokenizer
    render = granite_messages if args.adapter == "granite" else (lambda m: m)
    sp_cap = SamplingParams(temperature=0.0, max_tokens=17, skip_special_tokens=False, extra_args={"hooks_on": "both"})
    sp_cont = SamplingParams(temperature=0.0, max_tokens=512 - 17, skip_special_tokens=False)
    J = 16
    pre_all, dec_all, steps, texts, n_prompt, fin = [], [], [], [], [], []
    t0 = time.time()
    for rnd in range(args.max_rounds):
        rows = [json.loads(l) for l in open(work / f"state_{rnd}.jsonl")]
        active = [r for r in rows if not r["done"]]
        if not active:
            break
        ids = {r["traj"]: tok.apply_chat_template(render(r["messages"]), tools=r["tools"], add_generation_prompt=True,
                                                  tokenize=True, return_dict=False) for r in active}
        too_long = {t for t, v in ids.items() if len(v) > args.max_model_len - 512}
        run = [r for r in active if r["traj"] not in too_long]
        gen, pre, dec, f17 = {}, {}, {}, {}
        for k in range(0, len(run), args.chunk):
            part = run[k:k + args.chunk]
            out = llm.generate([{"prompt_token_ids": ids[r["traj"]]} for r in part], sp_cap, merge_probes=False)
            for r, o in zip(part, out):
                entry = next(e for e in o.probes["hs_cache"].values() if int(e["layer_num"]) == args.layer)
                passes = [t.float().cpu().numpy().reshape(-1) for t in entry["hidden_states"]]
                pre[r["traj"]] = passes[0]
                d = np.full((J, passes[0].size), np.nan, np.float32)
                if passes[1:1 + J]:
                    d[: len(passes[1:1 + J])] = np.stack(passes[1:1 + J])
                dec[r["traj"]] = d
                gen[r["traj"]] = list(o.outputs[0].token_ids)
                f17[r["traj"]] = o.outputs[0].finish_reason == "stop"
        todo = [r["traj"] for r in run if not f17[r["traj"]]]
        cont = {}
        for k in range(0, len(todo), 64):
            part = todo[k:k + 64]
            out = llm.generate([{"prompt_token_ids": ids[t] + gen[t]} for t in part], sp_cont, use_hook=False)
            for t, o in zip(part, out):
                cont[t] = list(o.outputs[0].token_ids)
        rtexts = []
        for r in active:
            t = r["traj"]
            text = tok.decode(gen.get(t, []) + cont.get(t, []), skip_special_tokens=False) if t in gen else ""
            cid = f"{t}_roll{rnd}"
            rtexts.append({"traj": t, "case_id": cid, "text": text, "skipped_too_long": t in too_long})
            if t in too_long:
                continue
            H = pre[t].size
            pre_all.append(pre[t].astype(np.float16))
            dec_all.append(dec[t].astype(np.float16))
            n_prompt.append(len(ids[t]))
            fin.append(f17[t])
            steps.append({"case_id": cid, "traj": t, "round": rnd, **{k: r[k] for k in ("suite", "user_task", "injection_task",
                                                                                      "attack", "injections", "markers", "messages")}})
            texts.append({"case_id": cid, "text": text, "skipped_too_long": False})
        (work / f"texts_{rnd}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rtexts))
        env("step", work / f"state_{rnd}.jsonl", work / f"texts_{rnd}.jsonl", work / f"state_{rnd + 1}.jsonl", rnd,
            args.max_rounds, "granite" if args.adapter == "granite" else "qwen")
        print(f"round {rnd}: {len(run)} steps, {len(too_long)} too long, {time.time() - t0:.0f}s", flush=True)
    np.savez(args.out + ".npz", case_id=np.array([s["case_id"] for s in steps]),
             **{f"prefill_layer_{args.layer}": np.stack(pre_all), f"dec_layer_{args.layer}": np.stack(dec_all)},
             n_prompt=np.array(n_prompt), finish17=np.array(fin))
    Path(args.out + ".steps.jsonl").write_text("".join(json.dumps(s, default=str) + "\n" for s in steps))
    Path(args.out + ".texts.jsonl").write_text("".join(json.dumps(t) + "\n" for t in texts))
    print(f"done {len(steps)} steps in {time.time() - t0:.0f}s, gpu {torch.cuda.get_device_name(0)}", flush=True)


if __name__ == "__main__":
    main()
