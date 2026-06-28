# Organoid_Overlap_Instance_Segmentation
Overlapping Organoid Instance Segmentation Using Pseudo-Label Unmixing and Synthesis-Assisted Learning
# **Preprint/Associated Paper:**
This code repository accompanies our preprint:  
 **"Boosting Overlapping Organoid Instance Segmentation Using Pseudo-Label Unmixing and Synthesis-Assisted Learning"**   
Available on arXiv: https://arxiv.org/abs/2601.06642

This project proposes a semi-supervised learning (SA-SSL) method for overlapping instance segmentation of organoids. By combining the Point-Teaching mechanism with pix2pixHD contour synthesis technology, it effectively addresses the challenges of instance overlap and high annotation costs in organoid microscopic images.

## 📑 Table of Contents
- [Environment Setup](#-environment-setup)
- [Dataset Preparation](#-dataset-preparation)
- [Model and Training Configuration](#%EF%B8%8F-model-and-training-configuration)
- [Model Training](#-model-training)
- [Acknowledgements and Citation](#-acknowledgements-and-citation)

---

## 🛠️ Environment Setup

This project is built on Python 3.8, PyTorch, and Detectron2. Please follow the steps below to configure your running environment.

### 1. Create Conda Virtual Environment
```bash
conda create -n SA-SSL python=3.8.5
conda activate SA-SSL
```

### 2. Install PyTorch
Please visit the [PyTorch Official Website](https://pytorch.org/get-started/previous-versions/) to get the installation command that matches your CUDA version. For example (CUDA 11.1):
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Install Detectron2 and Related Dependencies
Navigate to the project root directory and execute the following commands to install the locally modified Detectron2:
```bash
pip install -e ./detectron2-main/
pip install pandas
```

### 4. Install Point-Teaching Module
```bash
pip install -e ./Point-Teaching-main/
pip install opencv-python
# Downgrade setuptools to avoid compilation errors with some older dependencies
pip install setuptools==57.5.0 
```

### 5. Install pix2pixHD Dependencies
Required for the Contour Synthesis module:
```bash
pip install dominate scipy scikit-image
```

---

## 📂 Dataset Preparation

This project utilizes datasets in COCO format. Please ensure your dataset includes `train`, `val`, and `test` image folders along with their corresponding `annotations` JSON files.

### 1. Set Dataset Environment Variable
Set the Detectron2 dataset root directory in your terminal (it is recommended to add this to your `~/.bashrc` or `~/.zshrc`):
```bash
export DETECTRON2_DATASETS="/opt/data/private/SA-SSL/datasets/"
```

### 2. Register Custom Dataset
Modify `detectron2-main/detectron2/data/datasets/builtin.py` to add your dataset path mapping in the COCO dataset registration logic:
```python
# Add the following mappings in builtin.py
"coco_organoids_train": ("train", "annotations/instances_train.json"),
"coco_organoids_val": ("val", "annotations/instances_val.json"),
"coco_organoids_test": ("test", "annotations/instances_test.json"),
```

### 3. Modify Category Metadata
Modify `detectron2-main/detectron2/data/datasets/builtin_meta.py` to update the category information:
```python
# Modify the COCO_CATEGORIES list to keep only your categories
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "organoids"},
]

# In the _get_coco_instances_meta() function, modify the assertion for the number of thing_ids
# If you have multiple classes, change 1 to the actual number of classes
assert len(thing_ids) == 1, len(thing_ids)
```

---

## ⚙️ Model and Training Configuration

Before starting training, you need to modify the relevant configuration files according to your specific experimental setup.

### 1. Dataset and Number of Classes Configuration
- **Modify Dataset Names**:
  Open `Point-Teaching-main/configs/Mask-RCNN/Base-RCNN-FPN.yaml` and the yaml files under the `coco_supervision/` directory. Change `DATASETS` to the dataset names registered earlier:
  ```yaml
  DATASETS:
    TRAIN: ("coco_organoids_train",)
    TEST: ("coco_organoids_val",)
  ```
- **Modify Model Number of Classes**:
  Open `detectron2-main/detectron2/config/defaults.py` and modify the number of classes for ROI Heads:
  ```python
  _C.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Set to the actual number of classes
  ```

### 2. Semi-Supervised and Generative Model Configuration
- **Semi-Supervised Data Split**:
  Open `Point-Teaching-main/pteacher/config.py` and modify the reading path for the semi-supervised data split file (e.g., `COCO_organoids_supervision_1540.txt`).
- **Generative Model (pix2pixHD) Paths**:
  Open `Point-Teaching-main/pteacher/engine/options/base_options.py` and configure the checkpoint parameters for the generative model:
  ```python
  parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
  parser.add_argument('--name', type=str, default='organoid_pix2pix', help='name of the experiment')
  parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
  ```

### 3. Pre-trained Weights Configuration
To accelerate convergence, it is recommended to use semi-supervised pre-trained weights. Open `Point-Teaching-main/configs/Mask-RCNN/coco_supervision/mask_rcnn_R_50_FPN_sup1_run1.yaml` and configure the `WEIGHTS` parameter:
```yaml
MODEL:
  WEIGHTS: "/path/to/your/semi_supervised_pretrained_weights.pth"
```

---

## 🚀 Model Training

### 1. Start Training
Use the following command to start model training. Ensure you replace the path after `--config` with the configuration file you are actually using, and specify `OUTPUT_DIR` as the save path for logs and weights:

```bash
python Point-Teaching-main/tools/train_net.py \
    --config Point-Teaching-main/configs/Mask-RCNN/coco_supervision/mask_rcnn_R_50_FPN_sup1_run1.yaml \
    OUTPUT_DIR ./SA-semi-maskrcnn/250317_1
```

### 2. Training Monitoring and Stopping Criteria
- **Monitoring Tool**: It is highly recommended to use **TensorBoard** to monitor the training process in real-time. 
