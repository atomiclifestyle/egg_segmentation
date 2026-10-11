# Egg Segmentation and Tracking

1. **Task 1**: ground-truth egg masks using Grounding DINO (text prompt "an egg") + SAM.
2. **Task 2**: classical image-processing segmentation, with a leakage-aware 4:1 dev/eval split and a logged experiment series.
3. **Task 3**: quantitative and qualitative evaluation of Task 2 against Task 1.
4. **Task 4**: video egg tracking (partial submission, with limitations).

The full write-up is in [`report/report.pdf`](report/report.pdf).

Tracking output video: [`Video Tracking Output`](https://drive.google.com/file/d/16-N6Ag4vIMdx15ZJXwc_v4KcCdPXqyXk/view?usp=sharing)

## Setup

```bash
git clone https://github.com/atomiclifestyle/egg_segmentation.git
cd egg-segmentation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the experiments

Place the dataset zip somewhere, e.g. `data/egg_dataset.zip`.

**Task 1 - ground truth**

```bash
python src/task1_sam_segmentation.py --zip data/egg_dataset.zip --data data/dataset --out outputs/task1_ground_truth
```

**Tasks 2 and 3 - classical segmentation and evaluation**

```bash
python src/task2_3_classical_eval.py --data data/dataset --gt outputs/task1_ground_truth --work outputs/task2_3
```

**Task 4 - video tracking**

```bash
python src/task4_video_tracking.py --cfg outputs/task2_3/best_cfg.json --out outputs/task4
```

## Method summary

- **Split**: images from all three folders are grouped (same base filename with augmentation suffix removed, or perceptual-hash match under flips/rotations) and split by group, so no group appears in both dev and eval.
- **Classical algorithm**: smoothing, adaptive threshold (Otsu on gray / HSV / Lab or Canny edge fill), morphology, area and shape filters, optional watershed to split touching eggs and GrabCut boundary refinement. Chosen by a trial series scored only on the dev set; see the experiment log in the report.
- **Metrics**: IoU, Dice, precision, recall, pixel accuracy, instance F1 (IoU >= 0.5, Hungarian matching), exact-count accuracy.
- **Tracker**: per-frame segmentation, Hungarian assignment of detections to tracks on predicted centroids with a distance gate, constant-velocity coasting for missed frames.

## Limitations

Video tracking is a partial submission: the classical segmentation does not generalise well to the video, which produces fragmented tracks and identity switches. See the report for details and proposed fixes.
