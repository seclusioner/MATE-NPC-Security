# Detector Artifacts

Pretrained detector artifacts are optional.

MATE supports two Layer-1 detector families:

## Original Multi-RAAC

Expected output:

```text
artifacts/phi3/original/
├── harm_detector.pt
├── injection_detector.pt
└── system.json
```

Build with:
```bash
bash scripts/build_original_multiraac.sh \
  data/train.jsonl \
  data/validation.jsonl
```

## Low-FPR Sparse Multi-RAAC
Expected output:
```text
artifacts/phi3/lowfpr_sparse/
├── base_heads/
├── fusion_fpr_0p0/
├── fusion_fpr_0p01/
├── fusion_fpr_0p025/
├── fusion_fpr_0p05/
└── lowfpr_training_summary.json
```

Each deployment variant contains:
```text
harm_detector.pt
injection_detector.pt
system.json
```

Build with:
```bash
bash scripts/build_lowfpr_sparse_multiraac.sh \
  data/train.jsonl \
  data/validation.jsonl
```

Do not mix detector heads and system.json files from
different fusion operating-point directories.
