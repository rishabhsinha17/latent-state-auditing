#!/usr/bin/env bash
# Run on a fresh CUDA pod (one GPU with >= 24 GB) from the directory holding
# capture.py and data/. Pulls the PR branch, installs vLLM 0.21.0 (main's
# supported ceiling), then writes three captures into out/.
# Layout: venv on local disk, model cache in RAM (/dev/shm), no pip cache.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
export HF_HOME=/dev/shm/hf PIP_NO_CACHE_DIR=1 HF_HUB_ENABLE_HF_TRANSFER=0

[ -d /root/venv ] || python3 -m venv /root/venv
. /root/venv/bin/activate
pip install -q --upgrade pip
pip install -q "vllm==0.21.0" transformers numpy zstandard blake3

[ -d vLLM-Hook ] || git clone -q -b feature/tool-call-risk https://github.com/rishabhsinha17/vLLM-Hook.git
pip install -q -e vLLM-Hook/vllm_hook_plugins
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python capture.py vllm data/exp_ws_full --chunk 0 --out out/vllm_full.npy --cache "$HF_HOME/hub"
python capture.py vllm data/exp_ws_full --chunk 1 --out out/vllm_bs1.npy  --cache "$HF_HOME/hub"
python capture.py hf   data/exp_ws_full --out out/hf_fresh.npy
ls -la out/
echo PARITY_DONE
