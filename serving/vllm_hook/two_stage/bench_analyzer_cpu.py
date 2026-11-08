"""CPU cost of the analyzer step only: tool_call_risk vs two_stage_probe on an in-memory hs_cache.

Measures probe scoring after capture (hidden size 3584, random probes, float32). It does not
measure GPU capture of the second layer, which needs a GPU run (see PLAN in the milestone report).

Usage: bench_analyzer_cpu.py <vllm-hook repo>   (run from a venv that can import mia)
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from mia.analyzers.tool_call_risk_analyzer import ToolCallRiskAnalyzer  # noqa: E402
from mia.analyzers.two_stage_probe_analyzer import TwoStageProbeAnalyzer  # noqa: E402

H, REPS = 3584, 50
tmp = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)
paths = {}
for name, layer in (("s1", 28), ("s2", 21)):
    d = tmp / name
    d.mkdir()
    np.savez(d / "probe.npz", mean=rng.normal(size=H).astype(np.float32), scale=np.ones(H, np.float32),
             weights=(rng.normal(size=H) / 60).astype(np.float32), bias=np.float32(0), layer=np.int64(layer))
    (d / "probe.json").write_text(json.dumps({"model_name": "bench", "layer": layer, "threshold": 0.5}))
    paths[name] = str(d / "probe.npz")

rows = []
for batch in (1, 16, 256):
    cache = {f"model.layers.{l - 1}": {"layer_num": l, "hidden_states": [torch.randn(H, dtype=torch.bfloat16)
                                                                        for _ in range(batch)]}
             for l in (21, 28)}
    single, two = ToolCallRiskAnalyzer(str(tmp), {}), TwoStageProbeAnalyzer(str(tmp), {})
    specs = {"tool_call_risk": (single, {"probe_path": paths["s1"]}),
             "two_stage_probe": (two, {"stage1": {"probe_path": paths["s1"]},
                                       "stage2": {"probe_path": paths["s2"]}})}
    for name, (an, spec) in specs.items():
        an.analyze(analyzer_spec=spec, probes={"hs_cache": cache})   # load and warm the probes
        t = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            an.analyze(analyzer_spec=spec, probes={"hs_cache": cache})
            t.append(time.perf_counter() - t0)
        rows.append({"analyzer": name, "batch": batch, "median_ms": round(1e3 * float(np.median(t)), 3),
                     "p90_ms": round(1e3 * float(np.percentile(t, 90)), 3)})
        print(json.dumps(rows[-1]), flush=True)
