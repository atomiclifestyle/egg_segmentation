# Egg Segmentation and Tracking

1. **Task 1**: ground-truth egg masks using Grounding DINO (text prompt "an egg") + SAM.
2. **Task 2**: classical image-processing segmentation, with a leakage-aware 4:1 dev/eval split and a logged experiment series.
3. **Task 3**: quantitative and qualitative evaluation of Task 2 against Task 1.
4. **Task 4**: video egg tracking (partial submission, see limitations).

The full write-up is in [`report/report.pdf`](report/report.pdf).

Tracking output video: **PASTE_PUBLIC_GOOGLE_DRIVE_LINK_HERE**

## Repository layout

```
src/
  common.py                      classical segmentation pipeline (shared by tasks 2-4)
  task1_sam_segmentation.py      Grounding DINO + SAM ground truth
  task2_3_classical_eval.py      split, experiment log, evaluation, plots, report.md
  task4_video_tracking.py        segmentation + centroid tracker on a video
results/                         outputs produced by the runs used in the report
report/                          PDF report and its markdown source
requirements.txt
```

The dataset itself is not committed. It must contain `train/`, `valid/` and `test/` folders of images.

## Setup

Python 3.10+ and `ffmpeg` are required. A GPU is strongly recommended for Task 1.

```bash
git clone <REPO_URL>
cd <REPO_NAME>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Google Colab, `pip install -r requirements.txt` is enough (torch is preinstalled); select a GPU runtime for Task 1.

## Running the experiments

Place the dataset zip somewhere, e.g. `data/egg_dataset.zip`.

**Task 1 - ground truth**

```bash
python src/task1_sam_segmentation.py --zip data/egg_dataset.zip --data data/dataset --out outputs/task1_ground_truth
```

Writes, per split, `masks/` (binary), `instance_masks/` (one id per egg), `viz/` (original next to red overlay) and `annotations_coco.json`, plus `no_detection.txt`.
Use `--sam facebook/sam-vit-base` if GPU memory is limited.

**Tasks 2 and 3 - classical segmentation and evaluation**

```bash
python src/task2_3_classical_eval.py --data data/dataset --gt outputs/task1_ground_truth --work outputs/task2_3
```

If the zip extracted into a single top-level folder, point `--data` at that folder.
Outputs in `outputs/task2_3/`: `split.csv` (dev/eval assignment), `best_cfg.json` (final algorithm parameters), `predictions/`, and `report/` containing `experiment_log.md/.csv`, `eval_per_image.csv`, plots and the auto-generated `report.md`.

**Task 4 - video tracking**

```bash
python src/task4_video_tracking.py --cfg outputs/task2_3/best_cfg.json --out outputs/task4
```

Downloads the video with `yt-dlp` into `data/egg_video.mp4` if it is not already there (if YouTube blocks the download, download the video manually and pass `--video path/to/video.mp4`).
Useful options: `--max-seconds`, `--stride`, `--override '{"method": "hsv", "watershed": true}'` to change segmentation parameters for the video.
Outputs in `outputs/task4/`: `tracked_eggs.mp4`, `egg_tracks.csv`, `trajectories.png`, `preview.png`.

## Method summary

- **Split**: images from all three folders are grouped (same base filename with augmentation suffix removed, or perceptual-hash match under flips/rotations) and split by group, so no group appears in both dev and eval.
- **Classical algorithm**: smoothing, adaptive threshold (Otsu on gray / HSV / Lab or Canny edge fill), morphology, area and shape filters, optional watershed to split touching eggs and GrabCut boundary refinement. Chosen by a trial series scored only on the dev set; see the experiment log in the report.
- **Metrics**: IoU, Dice, precision, recall, pixel accuracy, instance F1 (IoU >= 0.5, Hungarian matching), exact-count accuracy.
- **Tracker**: per-frame segmentation, Hungarian assignment of detections to tracks on predicted centroids with a distance gate, constant-velocity coasting for missed frames.

## Limitations

Video tracking is a partial submission: the classical segmentation does not generalise well to the video, which produces fragmented tracks and identity switches. See the report for details and proposed fixes.
