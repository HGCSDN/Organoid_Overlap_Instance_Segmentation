#!/usr/bin/env python3
"""Run Mask R-CNN and PLU heads, saving one record per predicted ROI."""
import argparse, json, sys
from pathlib import Path
import cv2
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "Point-Teaching-main" / "tools"))
sys.path.insert(0, str(REPO))
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Boxes
from pteacher.engine.ins_seg_trainer import MaskRCNNpteacherTrainer
from pteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from OrgAI import PLU_Overlap_Judge, PLU_Decomposition_Branch
import train_net

def load_plu_weights(path, judge, decomp):
    checkpoint = torch.load(path, map_location="cpu")
    model_state = checkpoint.get("model", {})
    judge_state = checkpoint.get("plu_judge", {}) or {k[10:]: v for k, v in model_state.items() if k.startswith("plu_judge.")}
    decomp_state = checkpoint.get("plu_decomp", {}) or {k[11:]: v for k, v in model_state.items() if k.startswith("plu_decomp.")}
    if not judge_state or not decomp_state:
        raise RuntimeError("Checkpoint does not contain plu_judge and plu_decomp weights.")
    judge.load_state_dict(judge_state, strict=True)
    decomp.load_state_dict(decomp_state, strict=True)

def main(args):
    dataset_root = Path(args.dataset_root).resolve()
    # Support both the documented LG_Organoids layout and the repository's
    # existing flat datasets/{train,val,test,annotations} layout.
    split_image_root = dataset_root / args.split / "images"
    split_json_file = dataset_root / args.split / "annotations" / f"instances_{args.split}.json"
    image_root = split_image_root if split_image_root.is_dir() else dataset_root / args.split
    json_file = split_json_file if split_json_file.is_file() else dataset_root / "annotations" / f"instances_{args.split}.json"
    register_coco_instances("lg_organoids_plu_infer", {}, str(json_file), str(image_root))
    class SetupArgs:
        config_file = str(Path(args.config).resolve())
        opts = []
        eval_only = True
        resume = False
    cfg = train_net.setup(SetupArgs())
    cfg.defrost(); cfg.MODEL.WEIGHTS = str(Path(args.weights).resolve()); cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold; cfg.freeze()
    student = MaskRCNNpteacherTrainer.build_model(cfg)
    teacher = MaskRCNNpteacherTrainer.build_model(cfg)
    ensemble = EnsembleTSModel(teacher, student)
    DetectionCheckpointer(ensemble).resume_or_load(str(Path(args.weights).resolve()), resume=False)
    model = teacher
    device = next(model.parameters()).device
    judge = PLU_Overlap_Judge(256, 14).to(device).eval()
    decomp = PLU_Decomposition_Branch(256, 14, 5).to(device).eval()
    load_plu_weights(args.weights, judge, decomp)
    model.eval()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    records = []
    for image_path in sorted(image_root.iterdir()):
        image = cv2.imread(str(image_path))
        if image is None: continue
        height, width = image.shape[:2]
        tensor = torch.as_tensor(image.transpose(2, 0, 1).copy(), device=device)
        with torch.no_grad():
            instances = model([{"image": tensor}])[0]["instances"].to(device)
            image_list = model.preprocess_image([{"image": tensor}])
            features = model.backbone(image_list.tensor)
            boxes = instances.pred_boxes.tensor.clone()
            boxes[:, [0, 2]] *= image_list.image_sizes[0][1] / width
            boxes[:, [1, 3]] *= image_list.image_sizes[0][0] / height
            roi_features = model.roi_heads.mask_pooler([features[f] for f in model.roi_heads.in_features], [Boxes(boxes)])
            overlap = torch.sigmoid(judge(roi_features)).cpu()
            mask_logits, count_logits = decomp(roi_features)
        image_records = []
        for i in range(len(instances)):
            image_records.append({"bbox_xyxy": instances.pred_boxes.tensor[i].cpu().tolist(), "category_id": int(instances.pred_classes[i].item()), "score": float(instances.scores[i].item()), "overlap_probability": float(overlap[i].item()), "predicted_instance_count": int((count_logits[i].sigmoid() > 0.5).sum().item()), "decomposition_count_logits": count_logits[i].cpu().tolist(), "decomposition_mask_logits": mask_logits[i].cpu().tolist()})
        records.append({"file_name": image_path.name, "instances": image_records})
    (output / "plu_predictions.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {sum(len(x['instances']) for x in records)} PLU ROI records to {output / 'plu_predictions.json'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--dataset-root", default=str(ROOT / "LG_Organoids")); p.add_argument("--split", choices=["train", "val", "test"], default="test"); p.add_argument("--config", default=str(ROOT / "configs" / "mask_rcnn_PLU_supervised.yaml")); p.add_argument("--weights", required=True); p.add_argument("--output", default=str(ROOT / "outputs" / "PLU_inference")); p.add_argument("--threshold", type=float, default=0.5); main(p.parse_args())
