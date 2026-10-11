# Egg Segmentation and Tracking - Report

Author, date, repository link, video link (public Google Drive).

## 1. Overview
Two or three sentences per task: what was built and the headline number. State clearly that Task 4 is a partial submission.

## 2. Task 1 - Ground truth with SAM
- Pipeline: Grounding DINO prompt "an egg." -> boxes (threshold, NMS, oversized-box filter) -> SAM box prompts -> largest component + hole fill.
- Parameters used (box/text thresholds, SAM variant).
- Figure: 3-4 of your `viz/` images (original | red overlay).
- Number of images per split, number with no detection, and how those were handled.

## 3. Task 2 - Classical segmentation
### 3.1 Dev/eval split
- 4:1 ratio, group-based, hash threshold and seed from the run.
- Table: images and groups in dev/eval, nearest dev-eval hash distance (from the console output / report.md).
### 3.2 Final algorithm
- Pipeline diagram or numbered steps, plus the final parameters (`best_cfg.json`).
### 3.3 Experiment log
- Table copied from `experiment_log.md` (trial, what changed, why, parameters, dev mIoU / instance F1, keep/discard, reason).
- `trial_progress.png`.
- A short paragraph per important trial in your own words: what failed and why, what finally worked.

## 4. Task 3 - Evaluation and analysis
### 4.1 Quantitative
- Dev and eval metric table (mIoU, mDice, precision, recall, pixel accuracy, instance F1, count accuracy).
- `quantitative.png` with 3-4 sentences of reading.
### 4.2 Qualitative
- `qualitative_worst.png` and `qualitative_best.png`.
- For each failure category (from `eval_per_image.csv`): count, example image, cause, fix.
### 4.3 Gaps and improvements
- Bullet list tied to the evidence above.

## 5. Task 4 - Video tracking (partial)
- Method: segmentation + centroid Hungarian tracker (gate, velocity coasting, min hits).
- Link to the Drive video, `trajectories.png`, a few frames.
- What did not work (fragmented tracks, ID switches, segmentation failures on video) with the evidence, and what you would do next (e.g. stronger per-frame detector, appearance features, Kalman filter).

## 6. Reproducibility
Commands from the README, library versions, seeds, runtime.
