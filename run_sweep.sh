#!/usr/bin/env bash
# run_sweep.sh -- run decompose_v7_2.py (unchanged) once per setting, then score_sweep.py.
# Run from the repository root.
# Last modified: 2026-11-23
#
#   ./run_sweep.sh samples/sources/14.jpeg      one image
#   ./run_sweep.sh samples/sources              every image in the folder
#
# Settings (environment variables, space separated):
#   OUT=output/decompose/sweep  TOPK="32 64 128"  CLEAN="0.3 0.5 0.7"  SEEDS="0 1 2"
#   DECOMPOSE_ARGS="--steps 25"   further decompose_v7_2.py options
#   e.g.  TOPK=64 CLEAN="0.3 0.7" SEEDS=1 ./run_sweep.sh samples/sources/14.jpeg   (quick test)
#
# Output: $OUT/topk<K>_c<clean>_s<seed>/<decompose_v7_2.py folders>, logs in $OUT/_logs/,
#         review in $OUT/_review/ (see score_sweep.py). Existing folders are skipped by
#         decompose_v7_2.py, so an interrupted sweep continues where it stopped.
# Score again with other thresholds without decomposing: python score_sweep.py --dup-clip 0.75
set -u

TARGET=${1:?usage: ./run_sweep.sh <image or folder of images>}
OUT=${OUT:-output/decompose/sweep}
TOPK=${TOPK:-"32 64"}
CLEAN=${CLEAN:-"0.3 0.5 0.7"}
SEEDS=${SEEDS:-"0 1"}
DECOMPOSE_ARGS=${DECOMPOSE_ARGS:-}

if [[ -d "$TARGET" ]]; then SRC=(--image-dir "$TARGET"); else SRC=(--image "$TARGET"); fi
mkdir -p "$OUT/_logs"

for k in $TOPK; do
  for c in $CLEAN; do
    for s in $SEEDS; do
      run="topk${k}_c${c}_s${s}"
      echo "== $run"
      python decompose_v7_2.py "${SRC[@]}" --output-dir "$OUT/$run" \
          --top-k "$k" --clean-frac "$c" --seed "$s" $DECOMPOSE_ARGS > "$OUT/_logs/$run.log" 2>&1 \
        || echo "   FAILED, see $OUT/_logs/$run.log"
    done
  done
done

python score_sweep.py --sweep-root "$OUT" "${SRC[@]}"