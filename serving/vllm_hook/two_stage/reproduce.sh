#!/bin/bash
# One-command CPU reproduction of the r0 and r1 two-stage FPR numbers.
#
#   PYTHON=/path/to/python3.12 [PROBE_DIR=dir with probe.npz] ./reproduce.sh [chat_dir] [out_dir]
#
# 1. checks the input checksums, 2. builds .venv with pinned numpy (requirements.txt) if absent,
# 3. reruns r0 and r1 into a fresh out_dir with the vendored cascade module, 4. compares every
# output JSON with expected_md5.txt. Takes about 15 minutes on an Apple M1 Pro. Numbers were
# produced with CPython 3.12.14 and numpy 2.3.5 on macOS arm64; other BLAS builds can change
# last-digit rounding and therefore the checksums, not the conclusions.
set -euo pipefail
cd "$(dirname "$0")"
CHAT="${1:-../chat}"
OUT="${2:-repro_$(date +%Y%m%d_%H%M%S)}"
PY="${PYTHON:-python3.12}"
PROBE_DIR="${PROBE_DIR:-../export}"
CAS=vendor/mia/utils/two_stage/cascade.py

md5of() { if command -v md5 >/dev/null; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }

echo "== inputs"
while read -r sum file; do
  got=$(md5of "$CHAT/$file")
  [ "$got" = "$sum" ] && echo "ok   $file" || { echo "BAD  $file ($got, want $sum)"; exit 1; }
done <<'EOF'
4819017b9713ca9bdc5c82922269bc83 suite_full.npz
701f1e830b4c4bbc5af483385ade9584 suite_full.jsonl
200736e3198da25e9e952ac8eb257078 suite_full.labels.jsonl
EOF
[ -f "$PROBE_DIR/probe.npz" ] || { echo "missing $PROBE_DIR/probe.npz (released stage-1 probe)"; exit 1; }

echo "== environment"
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
V=.venv/bin/python
$V -c "import sys, numpy; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"

echo "== r0 (out: $OUT/r0)"
$V -I eval_two_stage.py "$CHAT" vendor "$OUT/r0" --seeds 13,0,1,2 --boot 2000 > "$OUT.r0.log" 2>&1 || { mkdir -p "$OUT"; cat "$OUT.r0.log"; exit 1; }
mv "$OUT.r0.log" "$OUT/r0/eval.log"
$V -I eval_context_thresholds.py "$CHAT" vendor "$OUT/r0" > /dev/null
$V -I within_suite_auroc.py "$CHAT" vendor "$OUT/r0/within_suite_auroc.json" > /dev/null
$V -I per_suite.py "$CHAT" "$OUT/r0" > /dev/null

echo "== r1 (out: $OUT/r1)"
mkdir -p "$OUT/r1"
$V -I diagnose.py "$CHAT" "$CAS" "$OUT/r1/diagnose.json" > "$OUT/r1/diagnose.log"
$V -I eval_r1.py "$CHAT" "$CAS" "$OUT/r1" > "$OUT/r1/eval_r1.log"
$V -I eval_r1.py "$CHAT" "$CAS" "$OUT/r1" --posthoc > "$OUT/r1/eval_r1_posthoc.log"
$V -I control_presence.py "$CHAT" "$CAS" "$OUT/r1" > "$OUT/r1/control_presence.log"
$V -I per_suite_r1.py "$CHAT" "$OUT/r1" > "$OUT/r1/per_suite_r1.log"
$V -I export_r1.py "$CHAT" "$CAS" "$PROBE_DIR" "$OUT/r1/export" > "$OUT/r1/export.log"

echo "== compare with expected_md5.txt"
fail=0
while read -r sum file; do
  got=$(md5of "$OUT/$file")
  if [ "$got" = "$sum" ]; then echo "MATCH $file"; else echo "DIFF  $file"; fail=1; fi
done < expected_md5.txt
[ $fail = 0 ] && echo "all outputs reproduced" || { echo "some outputs differ (see above)"; exit 2; }
