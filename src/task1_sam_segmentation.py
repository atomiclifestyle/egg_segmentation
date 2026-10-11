import os
import glob
import json
import argparse
import zipfile
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.ops import nms
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          SamModel, SamProcessor)

parser = argparse.ArgumentParser()
parser.add_argument("--zip", default=None)
parser.add_argument("--data", default="data/dataset")
parser.add_argument("--out", default="outputs/task1_ground_truth")
parser.add_argument("--sam", default="facebook/sam-vit-huge")
args = parser.parse_args()

DATA_DIR = args.data
if args.zip:
    os.makedirs(DATA_DIR, exist_ok=True)
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(DATA_DIR)
    entries = [e for e in os.listdir(DATA_DIR) if not e.startswith(("__MACOSX", "."))]
    if len(entries) == 1 and os.path.isdir(os.path.join(DATA_DIR, entries[0])):
        DATA_DIR = os.path.join(DATA_DIR, entries[0])

OUT_DIR = args.out
SPLITS = ["train", "test", "valid"]
PROMPT = "an egg."
BOX_THRESH = 0.30
TEXT_THRESH = 0.25
MAX_BOX_AREA = 0.90
NMS_IOU = 0.50
SAM_ID = args.sam
DINO_ID = "IDEA-Research/grounding-dino-base"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
RED = (255, 30, 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

dino_proc = AutoProcessor.from_pretrained(DINO_ID)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(device).eval()
sam_proc = SamProcessor.from_pretrained(SAM_ID)
sam_model = SamModel.from_pretrained(SAM_ID).to(device).eval()

@torch.no_grad()
def detect_eggs(img: Image.Image):
    """Text-prompted detection -> boxes (N,4 xyxy, pixels), scores (N,)"""
    inputs = dino_proc(images=img, text=PROMPT, return_tensors="pt").to(device)
    out = dino_model(**inputs)
    kw = dict(target_sizes=[img.size[::-1]])
    try:
        res = dino_proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=BOX_THRESH, text_threshold=TEXT_THRESH, **kw)[0]
    except TypeError:
        res = dino_proc.post_process_grounded_object_detection(
            out, inputs.input_ids, box_threshold=BOX_THRESH, text_threshold=TEXT_THRESH, **kw)[0]
    boxes, scores = res["boxes"].cpu(), res["scores"].cpu()
    if len(boxes) == 0:
        return boxes, scores
    W, H = img.size
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = area < MAX_BOX_AREA * W * H
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes):
        keep = nms(boxes, scores, NMS_IOU)
        boxes, scores = boxes[keep], scores[keep]
    return boxes, scores

@torch.no_grad()
def segment_boxes(img: Image.Image, boxes: torch.Tensor):
    """SAM with box prompts -> bool masks (N,H,W)"""
    inputs = sam_proc(img, input_boxes=[boxes.tolist()], return_tensors="pt").to(device)
    out = sam_model(**inputs, multimask_output=False)
    masks = sam_proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())[0]
    return masks[:, 0].numpy().astype(bool)

def clean_mask(m: np.ndarray) -> np.ndarray:
    """Keep largest connected component and fill holes."""
    m = m.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)

def side_by_side(img_np, masks):
    over = img_np.copy()
    for m in masks:
        over[m] = RED
    return np.concatenate([img_np, over], axis=1)

def mask_to_polygons(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).astype(float).tolist() for c in cnts if len(c) >= 3]

os.makedirs(OUT_DIR, exist_ok=True)
no_detection = []

for split in SPLITS:

    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_dir):
        print(f"[skip] {split} not found"); continue
    files_ = sorted(p for p in glob.glob(os.path.join(split_dir, "**", "*"), recursive=True)
                    if p.lower().endswith(IMG_EXT))

    for sub in ("masks", "instance_masks", "viz"):
        os.makedirs(os.path.join(OUT_DIR, split, sub), exist_ok=True)

    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "egg"}]}
    ann_id = 1

    for img_id, path in enumerate(tqdm(files_, desc=split), start=1):
        stem = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        boxes, scores = detect_eggs(img)
        if len(boxes) == 0:
            no_detection.append(path)
            masks = np.zeros((0, H, W), bool)
        else:
            masks = np.stack([clean_mask(m) for m in segment_boxes(img, boxes)])

        binary = (masks.any(0) if len(masks) else np.zeros((H, W), bool)).astype(np.uint8) * 255
        inst = np.zeros((H, W), np.uint16)
        for i, m in enumerate(masks, start=1):
            inst[m] = i
        cv2.imwrite(os.path.join(OUT_DIR, split, "masks", stem + ".png"), binary)
        cv2.imwrite(os.path.join(OUT_DIR, split, "instance_masks", stem + ".png"), inst)
        Image.fromarray(side_by_side(img_np, masks)).save(
            os.path.join(OUT_DIR, split, "viz", stem + ".jpg"), quality=92)

        coco["images"].append({"id": img_id, "file_name": os.path.relpath(path, split_dir),
                               "width": W, "height": H})
        for m, b, s in zip(masks, boxes.tolist(), scores.tolist()):
            ys, xs = np.where(m)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
            coco["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                        "segmentation": mask_to_polygons(m), "bbox": bbox,
                                        "area": int(m.sum()), "iscrowd": 0, "score": float(s)})
            ann_id += 1

    with open(os.path.join(OUT_DIR, split, "annotations_coco.json"), "w") as f:
        json.dump(coco, f)

print(f"\nDone. Images with NO egg detected: {len(no_detection)}")
with open(os.path.join(OUT_DIR, "no_detection.txt"), "w") as f:
    f.write("\n".join(no_detection))
