# OrgAI 监督训练与推理

本目录提供项目完整模型的全标注监督训练和推理脚本。训练使用已经实现 PLU 标签构造和 loss 的 `MaskRCNNpteacherTrainer`。模型不是普通 Mask R-CNN，而是：

```text
Mask R-CNN backbone/ROI mask head
              ├── PLU_Overlap_Judge       # 判断 ROI 是否为重叠/可靠实例
              └── PLU_Decomposition_Branch # 将重叠 ROI 解耦为最多 K=5 个实例
```

两个 PLU 分支对应仓库根目录 `OrgAI/` 中的 `PLU_Overlap_Judge` 和 `PLU_Decomposition_Branch`，并使用 Mask R-CNN 的 ROI 特征。这里使用假定的数据集 `LG_Organoids`，不读取仓库中已有的 `datasets` 数据。

## 数据集目录

数据集需为 COCO 实例分割格式：

```text
OrgAI/LG_Organoids/
├── train/images/                 # 训练图片
├── train/annotations/instances_train.json
├── val/images/
├── val/annotations/instances_val.json
├── test/images/
└── test/annotations/instances_test.json
```

## 训练

在仓库根目录执行（需先安装本仓库的 Detectron2 和 PyTorch 依赖）：

配置中的 `SUP_PERCENT: 100.0` 表示所有训练图片都有 GT 实例 mask；trainer 会根据这些 mask 自动生成 PLU 标签。注意 `SUP_PERCENT: 1.0` 表示 1%，不是 100%。完整模型训练统一使用下面的 `train_PLU.sh`。

### 全标注 PLU 训练命令行入口

使用项目中集成了 PLU 两个分支和监督标签的 `Point-Teaching` 训练器：

```bash
bash OrgAI/train_PLU.sh
```

等价的直接命令为：

```bash
python Point-Teaching-main/tools/train_net.py \
  --config-file OrgAI/configs/mask_rcnn_PLU_supervised.yaml \
  OUTPUT_DIR OrgAI/outputs/PLU_supervised
```

对应配置文件为 [`configs/mask_rcnn_PLU_supervised.yaml`](configs/mask_rcnn_PLU_supervised.yaml)。其中 `SEMISUPNET.Trainer: "mask_rcnn_pteacher"` 会调用自定义 trainer，使用 GT mask 生成 `gt_overlap_labels`、`gt_decomp_counts`、`gt_decomp_masks`，计算 `loss_O_cls`、`loss_i_count`、`loss_i_IoU`，并保存两个 PLU head 的权重。

## 推理

推理入口使用训练时的 Mask R-CNN 配置和 checkpoint，并逐个 ROI 执行两个 PLU head：

```bash
bash OrgAI/infer_PLU.sh OrgAI/outputs/PLU_supervised/model_final.pth
```

输出文件为 `OrgAI/outputs/PLU_inference/plu_predictions.json`，每个实例包含 `overlap_probability`、`predicted_instance_count`、`decomposition_count_logits` 和 `decomposition_mask_logits`。


