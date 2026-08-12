#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

MODEL_ID="${MODEL_ID:-Gigax/NPC-LLM-3_8B}"

TRAIN_MANIFEST="${1:-}"
VALIDATION_MANIFEST="${2:-}"

CACHE_DIR="${CACHE_DIR:-$ROOT/artifacts/hidden_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/phi3/lowfpr_sparse}"

LAYERS="${LAYERS:-24,28,30}"
TOP_K_GRID="${TOP_K_GRID:-32,64,128,256,512}"
CV_FOLDS="${CV_FOLDS:-5}"
HEAD_TARGET_FPR="${HEAD_TARGET_FPR:-0.01}"
FUSION_TARGET_FPRS="${FUSION_TARGET_FPRS:-0,0.01,0.025,0.05}"
SEED="${SEED:-42}"

DTYPE="${DTYPE:-bfloat16}"
CACHE_DTYPE="${CACHE_DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-phi3_manual}"

if [[ -z "$TRAIN_MANIFEST" || -z "$VALIDATION_MANIFEST" ]]; then
  echo "Usage:"
  echo "  bash scripts/build_lowfpr_sparse_multiraac.sh \\"
  echo "    <train.jsonl> <validation.jsonl>"
  exit 2
fi

TRAIN_MANIFEST="$(realpath "$TRAIN_MANIFEST")"
VALIDATION_MANIFEST="$(realpath "$VALIDATION_MANIFEST")"

mkdir -p \
  "$CACHE_DIR" \
  "$OUTPUT_ROOT"

echo "============================================================"
echo "MATE — Low-FPR Sparse Multi-RAAC Builder"
echo "============================================================"
echo "Model:              $MODEL_ID"
echo "Train:              $TRAIN_MANIFEST"
echo "Validation:         $VALIDATION_MANIFEST"
echo "Layers:             $LAYERS"
echo "Top-K grid:         $TOP_K_GRID"
echo "CV folds:           $CV_FOLDS"
echo "Head target FPR:    $HEAD_TARGET_FPR"
echo "Fusion target FPRs: $FUSION_TARGET_FPRS"
echo "Cache dir:          $CACHE_DIR"
echo "Output root:        $OUTPUT_ROOT"
echo

echo "[1/2] Extract hidden representations"

python -m training.extract_hidden_states \
  --model-id "$MODEL_ID" \
  --manifests \
    "$TRAIN_MANIFEST" \
    "$VALIDATION_MANIFEST" \
  --output-dir "$CACHE_DIR" \
  --dtype "$DTYPE" \
  --cache-dtype "$CACHE_DTYPE" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --chat-template-mode "$CHAT_TEMPLATE_MODE"

TRAIN_CACHE="$CACHE_DIR/$(basename "${TRAIN_MANIFEST%.*}").pt"
VALIDATION_CACHE="$CACHE_DIR/$(basename "${VALIDATION_MANIFEST%.*}").pt"

if [[ ! -f "$TRAIN_CACHE" ]]; then
  echo "[ERROR] Missing train cache: $TRAIN_CACHE"
  exit 1
fi

if [[ ! -f "$VALIDATION_CACHE" ]]; then
  echo "[ERROR] Missing validation cache: $VALIDATION_CACHE"
  exit 1
fi

echo
echo "[2/2] Build Low-FPR Sparse Multi-RAAC"

python -m training.build_lowfpr_sparse_multiraac \
  --train-cache "$TRAIN_CACHE" \
  --validation-cache "$VALIDATION_CACHE" \
  --output-root "$OUTPUT_ROOT" \
  --layers "$LAYERS" \
  --top-k-grid "$TOP_K_GRID" \
  --cv-folds "$CV_FOLDS" \
  --head-target-fpr "$HEAD_TARGET_FPR" \
  --fusion-target-fprs "$FUSION_TARGET_FPRS" \
  --seed "$SEED"

echo
echo "============================================================"
echo "[DONE] Low-FPR Sparse Multi-RAAC"
echo "============================================================"
echo "Base heads:"
echo "  $OUTPUT_ROOT/base_heads/"
echo
echo "Deployment variants:"
find "$OUTPUT_ROOT" \
  -maxdepth 1 \
  -type d \
  -name 'fusion_fpr_*' \
  -print \
  | sort

echo
echo "Training summary:"
echo "  $OUTPUT_ROOT/lowfpr_training_summary.json"
