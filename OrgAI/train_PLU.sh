#!/usr/bin/env bash
set -euo pipefail

# The custom trainer builds Mask R-CNN + PLU_Overlap_Judge +
# PLU_Decomposition_Branch.  LG_Organoids is registered by Point-Teaching
# from DETECTRON2_DATASETS (default: ./OrgAI/LG_Organoids).
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$(pwd)/OrgAI}"

python Point-Teaching-main/tools/train_net.py \
  --config-file OrgAI/configs/mask_rcnn_PLU_supervised.yaml \
  OUTPUT_DIR OrgAI/outputs/PLU_supervised \
  MODEL.WEIGHTS detectron2://ImageNetPretrained/MSRA/R-50.pkl

