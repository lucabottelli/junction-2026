#!/bin/bash
# Tomorrow's ONE command per brick + camera: train the main + fast anomaly model -> OpenVINO export -> evaluate through
# the run-time path (scorer.py / score_worker.py) -> thresholds.json where the sorter reads them -> summary.
#
#   ~/junction26/anomaly_capture/anomaly_pipeline.sh red                  # front camera (brick held at WP2)
#   anomaly_capture/anomaly_pipeline.sh red wrist                        # top photo of the released brick (tape spot)
#   FAST_MODEL=none anomaly_capture/anomaly_pipeline.sh green            # main model only, no cascade
#   MAIN_ARGS="--image-size 384 --tag s384" FAST_MODEL=none anomaly_capture/anomaly_pipeline.sh red   # a variant, then:
#   $ANOMALIB_PY anomaly_capture/eval_anomaly.py --brick red --compare  # does the variant earn its place?
#
# Writes models/<cam>/<brick>/ (PatchCore, the sorter's main model) and models/<cam>/<brick>_efficientad/ (the
# cascade's fast stage, sort_settings fast_variant "_efficientad"), each with weights/openvino/model.xml,
# holdout.json, split.json, model_card.json, thresholds.json and eval/{report.md,eval.json,*.png}.
# Log: runs/anomaly_pipeline_<brick>_<cam>_<time>.log.  Then on /sort: judge = model, brick = <brick> (cascade on).
#
# Environment (all optional):
#   ANOMALIB_PY      python of the Anomalib venv (default ~/anomalib/.venv/bin/python); runs training, eval and the worker
#   DEVICE           training device auto|xpu|cpu (default auto = XPU if torch sees it)
#   EVAL_DEVICE      OpenVINO device for the metrics (default GPU; falls back to CPU if the model can't run there)
#   FAST_DEVICE      device of the fast stage (default NPU; falls back to CPU, e.g. while the NPU driver is missing)
#   LATENCY_DEVICES  devices to time (default CPU,GPU,NPU; a device that fails is reported, not fatal)
#   MAIN_MODEL / FAST_MODEL   default patchcore / efficientad (FAST_MODEL=none: skip the cascade)
#   MAIN_ARGS / FAST_ARGS / EVAL_ARGS   extra arguments (e.g. FAST_ARGS="--compress int8_ptq" for the NPU)
#   DATA / MODELS    default <repo>/training_images and <repo>/models
set -uo pipefail
BRICK=${1:?usage: anomaly_pipeline.sh <brick> [cam]}
CAM=${2:-front}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
export ANOMALIB_PY=${ANOMALIB_PY:-$HOME/anomalib/.venv/bin/python}
PY=$ANOMALIB_PY
DEVICE=${DEVICE:-auto}; EVAL_DEVICE=${EVAL_DEVICE:-GPU}; FAST_DEVICE=${FAST_DEVICE:-NPU}
LATENCY_DEVICES=${LATENCY_DEVICES:-CPU,GPU,NPU}
MAIN_MODEL=${MAIN_MODEL:-patchcore}; FAST_MODEL=${FAST_MODEL:-efficientad}
DATA=${DATA:-$ROOT/training_images}; MODELS=${MODELS:-$ROOT/models}
mkdir -p "$ROOT/runs"
LOG="$ROOT/runs/anomaly_pipeline_${BRICK}_${CAM}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
[ -x "$PY" ] || { echo "Anomalib python not found: $PY (set ANOMALIB_PY)"; exit 1; }
[ -f "$DATA/synth_info.json" ] && echo "*** SYNTHETIC DATA in $DATA: pipeline test only, not station results ***"
COMMON=(--brick "$BRICK" --cam "$CAM" --data "$DATA" --models "$MODELS")
t0=$(date +%s)

echo "== 1/3 main model ($MAIN_MODEL) =="
# shellcheck disable=SC2086
"$PY" "$HERE/train_anomaly.py" "${COMMON[@]}" --model "$MAIN_MODEL" --device "$DEVICE" ${MAIN_ARGS:-} || { echo "main model training FAILED"; exit 1; }
MAIN_DIR=$("$PY" "$HERE/train_anomaly.py" "${COMMON[@]}" --model "$MAIN_MODEL" ${MAIN_ARGS:-} --dry-run | sed -n 's/.*"out": "\(.*\)".*/\1/p')

FAST_DIR=""
if [ "$FAST_MODEL" != "none" ]; then
  echo "== 2/3 fast model ($FAST_MODEL) =="
  # shellcheck disable=SC2086
  if "$PY" "$HERE/train_anomaly.py" "${COMMON[@]}" --model "$FAST_MODEL" --device "$DEVICE" ${FAST_ARGS:-}; then
    FAST_DIR=$("$PY" "$HERE/train_anomaly.py" "${COMMON[@]}" --model "$FAST_MODEL" ${FAST_ARGS:-} --dry-run | sed -n 's/.*"out": "\(.*\)".*/\1/p')
  else
    echo "WARNING: fast model training failed: continuing without the cascade"
  fi
fi

echo "== 3/3 evaluate through scorer.py/score_worker.py, write thresholds =="
# shellcheck disable=SC2086
"$PY" "$HERE/eval_anomaly.py" --model-dir "$MAIN_DIR" --fast "${FAST_DIR:-none}" --data "$DATA" --device "$EVAL_DEVICE" \
  --fast-device "$FAST_DEVICE" --latency-devices "$LATENCY_DEVICES" --write-thresholds ${EVAL_ARGS:-} || { echo "evaluation FAILED"; exit 1; }
if [ -n "$FAST_DIR" ]; then
  # shellcheck disable=SC2086
  "$PY" "$HERE/eval_anomaly.py" --model-dir "$FAST_DIR" --fast none --data "$DATA" --device "$FAST_DEVICE" \
    --latency-devices "$LATENCY_DEVICES" ${EVAL_ARGS:-} || echo "WARNING: fast model evaluation failed"
fi

echo
echo "== summary ($(( $(date +%s) - t0 )) s) =="
sed -n '1,/^## Per view/p' "$MAIN_DIR/eval/report.md" | grep -v '^## Per view'
grep -A6 '^## Cascade' "$MAIN_DIR/eval/report.md" || true
echo
echo "main : $MAIN_DIR  (thresholds: $(cat "$MAIN_DIR/thresholds.json" 2>/dev/null | tr -d '\n ' | cut -c1-120))"
[ -n "$FAST_DIR" ] && echo "fast : $FAST_DIR  (thresholds: $(cat "$FAST_DIR/thresholds.json" 2>/dev/null | tr -d '\n ' | cut -c1-120))"
echo "report: $MAIN_DIR/eval/report.md   log: $LOG"
echo "next : /sort -> judge = model, brick = $BRICK$([ -n "$FAST_DIR" ] && echo ', cascade on')"
