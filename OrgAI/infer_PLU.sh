#!/usr/bin/env bash
set -euo pipefail

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$(pwd)/OrgAI}"
WEIGHTS="${1:-OrgAI/outputs/PLU_supervised/model_final.pth}"

python OrgAI/infer_PLU.py \
  --config OrgAI/configs/mask_rcnn_PLU_supervised.yaml \
  --weights "$WEIGHTS" \
  --split test \
  --output OrgAI/outputs/PLU_inference

