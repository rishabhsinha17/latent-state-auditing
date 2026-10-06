"""GPU capture for the vLLM-Hook vs HF parity check.

Both paths see identical token ids: the prompts are tokenized once with the
HF tokenizer exactly as the paper's TransformersActivationBackend did
(truncation=True, max_length from meta.json), and vLLM receives the ids
directly, so any difference is in the forward pass, not tokenization.

  python capture.py vllm <data> --chunk 0 --out vllm_full.npy
  python capture.py vllm <data> --chunk 1 --out vllm_bs1.npy
  python capture.py hf   <data> --out hf_fresh.npy

--chunk 0 sends every prompt in one generate() call (vLLM batches them as it
likes); --chunk 1 sends one prompt per call. Each run writes <out>.json with
versions, GPU, and timing.
"""
import argparse
import json
import os
import time
from pathlib import Path


def load(data, args=None):
    data = Path(data)
    meta = json.loads((data / "meta.json").read_text())
    # Overrides for the exactness control (same prompts, other model/dtype/layer).
    for key in ("model_name", "dtype", "hidden_state_index"):
        if args is not None and getattr(args, key, None) is not None:
            meta[key] = getattr(args, key)
    rows = [json.loads(l) for l in open(data / "prompts.jsonl")]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta["model_name"])
    ids = [tok(r["prompt"], truncation=True, max_length=meta["max_length"])["input_ids"] for r in rows]
    bad = [i for i, (r, x) in enumerate(zip(rows, ids)) if len(x) != r["n_tokens_hf"]]
    assert not bad, f"token count differs from the stored HF run at rows {bad[:10]}"
    return meta, ids


def env_info():
    import torch
    info = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    for mod in ("vllm", "transformers"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


def run_vllm(args):
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import numpy as np
    import torch
    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM
    from vllm_hook_plugins.run_utils import unpack_hidden_states

    meta, ids = load(args.data, args)
    layer = meta["hidden_state_index"]  # HF hidden_states[i] == vLLM-Hook layer i
    cfg = Path(args.out).with_suffix(".config.json")
    cfg.write_text(json.dumps({"hidden_states": {"layers": [layer], "mode": "last_token"}}))

    llm = HookLLM(
        model=meta["model_name"],
        worker_name="probe_hidden_states",
        analyzer_name="hidden_states",
        config_file=str(cfg),
        download_dir=args.cache,
        hook_dir=args.hook_dir,
        dtype=getattr(torch, meta["dtype"]),
        gpu_memory_utilization=0.85,
        max_model_len=meta["max_length"] + 16,
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_hook=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    prompts = [{"prompt_token_ids": x} for x in ids]
    chunk = args.chunk or len(prompts)

    vecs, t0 = [], time.time()
    for s in range(0, len(prompts), chunk):
        part = prompts[s:s + chunk]
        out = llm.generate(part, sp)
        hs = out[0].probes["hs_cache"]
        (entry,) = [e for e in hs.values() if int(e["layer_num"]) == layer]
        got = unpack_hidden_states(entry)
        assert len(got) == len(part), (len(got), len(part))
        vecs.extend(t.float().cpu().numpy() for t in got)
    elapsed = time.time() - t0

    np.save(args.out, np.stack(vecs).astype(np.float32))
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"path": "vllm", "chunk": chunk, "layer": layer, "rows": len(vecs),
         "seconds": round(elapsed, 1), **env_info()}, indent=2))
    print(f"saved {args.out} rows={len(vecs)} in {elapsed:.1f}s")


def run_hf(args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    meta, ids = load(args.data, args)
    idx = meta["hidden_state_index"]
    model = AutoModelForCausalLM.from_pretrained(
        meta["model_name"], dtype=getattr(torch, meta["dtype"]), cache_dir=args.cache,
    ).to("cuda").eval()

    vecs, t0 = [], time.time()
    with torch.no_grad():
        for x in ids:
            out = model(input_ids=torch.tensor([x], device="cuda"), output_hidden_states=True, use_cache=False)
            vecs.append(out.hidden_states[idx][0, -1].float().cpu().numpy())
    elapsed = time.time() - t0

    np.save(args.out, np.stack(vecs).astype(np.float32))
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"path": "hf", "hidden_state_index": idx, "rows": len(vecs),
         "seconds": round(elapsed, 1), **env_info()}, indent=2))
    print(f"saved {args.out} rows={len(vecs)} in {elapsed:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vllm")
    v.add_argument("data")
    v.add_argument("--chunk", type=int, default=0)
    v.add_argument("--out", required=True)
    v.add_argument("--cache", default=os.path.expanduser("~/.cache/huggingface/hub"))
    v.add_argument("--hook-dir", default="/dev/shm/vllm_hook")
    h = sub.add_parser("hf")
    h.add_argument("data")
    h.add_argument("--out", required=True)
    h.add_argument("--cache", default=None)
    for p in (v, h):
        p.add_argument("--model", dest="model_name")
        p.add_argument("--dtype")
        p.add_argument("--layer", dest="hidden_state_index", type=int)
    args = ap.parse_args()
    run_vllm(args) if args.cmd == "vllm" else run_hf(args)
