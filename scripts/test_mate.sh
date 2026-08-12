#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

cd "$ROOT"

pytest -q \
  tests/test_mate_policy.py \
  tests/test_execution_gate.py \
  tests/test_mate_runtime.py \
  tests/test_mate_config.py \
  tests/test_action_extensibility.py \
  tests/test_multiraac_fusion.py \
  tests/test_sparse_detector.py \
  tests/test_shared_sparse_multiraac.py \
  tests/test_layer1_config.py \
  tests/test_original_builder.py \
  tests/test_lowfpr_sparse_builder.py \
  tests/test_hidden_state_extractor.py \
  tests/test_prepare_data.py \
  tests/test_hf_dataset_adapter.py \
  tests/test_evaluate_layer1.py \
  tests/test_demo_cli.py
