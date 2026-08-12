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
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/phi3/original}"

LAYERS="${LAYERS:-24,28,30}"
DTYPE="${DTYPE:-bfloat16}"
CACHE_DTYPE="${CACHE_DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-phi3_manual}"

if [[ -z "$TRAIN_MANIFEST" || -z "$VALIDATION_MANIFEST" ]]; then
  echo "Usage:"
  echo "  bash scripts/build_original_multiraac.sh \\"
  echo "    <train.jsonl> <validation.jsonl>"
  exit 2
fi

TRAIN_MANIFEST="$(realpath "$TRAIN_MANIFEST")"
VALIDATION_MANIFEST="$(realpath "$VALIDATION_MANIFEST")"

mkdir -p \
  "$CACHE_DIR" \
  "$OUTPUT_DIR"

echo "============================================================"
echo "MATE — Original Multi-RAAC Builder"
echo "============================================================"
echo "Model:       $MODEL_ID"
echo "Train:       $TRAIN_MANIFEST"
echo "Validation:  $VALIDATION_MANIFEST"
echo "Layers:      $LAYERS"
echo "Cache dir:   $CACHE_DIR"
echo "Output dir:  $OUTPUT_DIR"
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
echo "[2/2] Build Original Multi-RAAC"

python -m training.build_original_multiraac \
  --train-cache "$TRAIN_CACHE" \
  --validation-cache "$VALIDATION_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --layers "$LAYERS"

echo
echo "============================================================"
echo "[DONE] Original Multi-RAAC"
echo "============================================================"
echo "$OUTPUT_DIR/harm_detector.pt"
echo "$OUTPUT_DIR/injection_detector.pt"
echo "$OUTPUT_DIR/system.json"
