# OrgAI Supervised Training and Inference

This directory provides scripts for fully supervised training and inference with the complete OrgAI model. Training uses the `MaskRCNNpteacherTrainer`, which implements PLU label generation and the corresponding loss functions. Rather than a standard Mask R-CNN, the model consists of:

```text
Mask R-CNN backbone/ROI mask head
              ├── PLU_Overlap_Judge        # Determines whether an ROI contains overlapping instances or a reliable single instance
              └── PLU_Decomposition_Branch # Separates an overlapping ROI into up to K=5 instances
```

The two PLU branches correspond to `PLU_Overlap_Judge` and `PLU_Decomposition_Branch` under `OrgAI/` at the repository root. Both branches use ROI features extracted by Mask R-CNN. This setup assumes a dataset named `LG_Organoids` and does not load data from the repository's existing `datasets` directory.

## Dataset Structure

The dataset must follow the COCO instance segmentation format:

```text
OrgAI/LG_Organoids/
├── train/images/                 # Training images
├── train/annotations/instances_train.json
├── val/images/
├── val/annotations/instances_val.json
├── test/images/
└── test/annotations/instances_test.json
```

## Training

Run the following commands from the repository root after installing the required Detectron2 and PyTorch dependencies.

In the configuration, `SUP_PERCENT: 100.0` indicates that all training images have ground-truth (GT) instance masks. The trainer automatically generates PLU supervision labels from these masks. Note that `SUP_PERCENT: 1.0` means **1%**, not 100%. Use `train_PLU.sh` below to train the complete model.

### Fully Supervised PLU Training

Use the project's `Point-Teaching` trainer, which integrates both PLU branches and their supervision labels:

```bash
bash OrgAI/train_PLU.sh
```

The equivalent direct command is:

```bash
python Point-Teaching-main/tools/train_net.py \
  --config-file OrgAI/configs/mask_rcnn_PLU_supervised.yaml \
  OUTPUT_DIR OrgAI/outputs/PLU_supervised
```

The corresponding configuration file is [`configs/mask_rcnn_PLU_supervised.yaml`](configs/mask_rcnn_PLU_supervised.yaml). Setting `SEMISUPNET.Trainer: "mask_rcnn_pteacher"` invokes the custom trainer, which generates `gt_overlap_labels`, `gt_decomp_counts`, and `gt_decomp_masks` from the GT masks; computes `loss_O_cls`, `loss_i_count`, and `loss_i_IoU`; and saves the weights of both PLU heads.

## Inference

Inference uses the Mask R-CNN configuration and checkpoint from training and applies both PLU heads to each ROI:

```bash
bash OrgAI/infer_PLU.sh OrgAI/outputs/PLU_supervised/model_final.pth
```

Predictions are saved to `OrgAI/outputs/PLU_inference/plu_predictions.json`. Each predicted instance includes `overlap_probability`, `predicted_instance_count`, `decomposition_count_logits`, and `decomposition_mask_logits`.
