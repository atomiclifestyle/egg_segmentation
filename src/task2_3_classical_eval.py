import os
import re
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupShuffleSplit
from common import segment, overlay, BASE_CFG

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--gt", required=True)
parser.add_argument("--work", default="outputs/task2_3")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-side", type=int, default=640)
parser.add_argument("--eval-frac", type=float, default=0.2)
args = parser.parse_args()

DATA_DIR = args.data
GT_DIR = args.gt
WORK_DIR = args.work
SPLIT_NAMES = ["train", "test", "valid"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_SIDE = args.max_side
SEED = args.seed
EVAL_FRAC = args.eval_frac
for d in ("", "report", "predictions"):
    os.makedirs(os.path.join(WORK_DIR, d), exist_ok=True)

rows = []
for split in SPLIT_NAMES:
    for p in sorted(glob.glob(os.path.join(DATA_DIR, split, "**", "*"), recursive=True)):
        if not p.lower().endswith(IMG_EXT):
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        gm = os.path.join(GT_DIR, split, "masks", stem + ".png")
        gi = os.path.join(GT_DIR, split, "instance_masks", stem + ".png")
        if os.path.exists(gm) and os.path.exists(gi):
            rows.append(dict(path=p, orig_split=split, stem=stem, gt_mask=gm, gt_inst=gi))
df = pd.DataFrame(rows)
print(len(df), "images with Task-1 ground truth")

def load_resized(path, interp, flag=cv2.IMREAD_COLOR):
    im = cv2.imread(path, flag)
    h, w = im.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1:
        im = cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=interp)
    return im

IMG, GTB, GTI = {}, {}, {}
for r in tqdm(df.itertuples(), total=len(df), desc="loading"):
    IMG[r.path] = load_resized(r.path, cv2.INTER_AREA)
    GTB[r.path] = load_resized(r.gt_mask, cv2.INTER_NEAREST, cv2.IMREAD_GRAYSCALE) > 127
    GTI[r.path] = load_resized(r.gt_inst, cv2.INTER_NEAREST, cv2.IMREAD_UNCHANGED).astype(np.int32)

def dihedral(a):
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, np.fliplr(r)]
    return out

def dhash(g):
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.packbits(bits).view(np.uint64)[0]

paths = df.path.tolist()
N = len(paths)
H = np.zeros((N, 8), np.uint64)
for i, p in enumerate(paths):
    thumb = cv2.resize(cv2.cvtColor(IMG[p], cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA)
    for k, v in enumerate(dihedral(thumb)):
        H[i, k] = dhash(cv2.resize(np.ascontiguousarray(v), (9, 8), interpolation=cv2.INTER_AREA))

POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
D = np.zeros((N, N), np.uint8)
for i in range(N):
    x = np.bitwise_xor(H, H[i, 0])
    D[i] = POP[x.view(np.uint8)].reshape(N, 8, 8).sum(-1).min(1)

def base_name(s):
    s = re.sub(r"\.rf\.[0-9a-fA-F]+$", "", s)
    return re.sub(r"_(jpg|jpeg|png)$", "", s)

def build_groups(T):
    parent = list(range(N))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b): parent[find(a)] = find(b)
    seen = {}
    for i, s in enumerate(df.stem):
        b = base_name(s)
        if b in seen: union(i, seen[b])
        else: seen[b] = i
    ii, jj = np.where(np.triu(D <= T, 1))
    for a, b in zip(ii, jj): union(a, b)
    return np.array([find(i) for i in range(N)])

for T in (10, 8, 6, 4, 2, 0):
    groups = build_groups(T)
    biggest = pd.Series(groups).value_counts().iloc[0] / N
    print(f"hash T={T}: {len(set(groups))} groups, largest group = {biggest:.1%} of data")
    if biggest <= 0.15:
        break
HASH_T = T

dev_idx, eval_idx = next(GroupShuffleSplit(1, test_size=EVAL_FRAC, random_state=SEED).split(df, groups=groups))
df["group"] = groups
df["subset"] = "dev"; df.loc[eval_idx, "subset"] = "eval"
DEV, EVAL = df[df.subset == "dev"].path.tolist(), df[df.subset == "eval"].path.tolist()
cross = D[np.ix_(dev_idx, eval_idx)]
print(f"\ndev={len(DEV)}  eval={len(EVAL)}  ratio={len(DEV)/max(1,len(EVAL)):.2f}:1")
print(f"leakage check: nearest dev<->eval pair has Hamming distance {cross.min()} (merge threshold was {HASH_T});",
      f"groups shared across subsets: {len(set(groups[dev_idx]) & set(groups[eval_idx]))}")
df.drop(columns=[]).to_csv(os.path.join(WORK_DIR, "split.csv"), index=False)

def iou_bin(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else np.logical_and(a, b).sum() / u

def eval_pair(pred_lab, gt_bin, gt_lab):
    p = pred_lab > 0
    tp = np.logical_and(p, gt_bin).sum(); ps, gs = p.sum(), gt_bin.sum()
    iou = 1.0 if ps + gs == 0 else tp / (ps + gs - tp)
    dice = 1.0 if ps + gs == 0 else 2 * tp / (ps + gs)
    prec = tp / ps if ps else (1.0 if gs == 0 else 0.0)
    rec = tp / gs if gs else (1.0 if ps == 0 else 0.0)
    pids = [i for i in np.unique(pred_lab) if i > 0]; gids = [i for i in np.unique(gt_lab) if i > 0]
    tpi = 0
    if pids and gids:
        M = np.array([[iou_bin(gt_lab == g, pred_lab == q) for q in pids] for g in gids])
        r, cidx = linear_sum_assignment(-M)
        tpi = int((M[r, cidx] >= 0.5).sum())
    return dict(iou=iou, dice=dice, precision=prec, recall=rec, pixel_acc=(p == gt_bin).mean(),
                n_gt=len(gids), n_pred=len(pids), inst_tp=tpi, inst_fp=len(pids) - tpi, inst_fn=len(gids) - tpi,
                pred_area=int(ps), gt_area=int(gs))

def evaluate_cfg(path_list, cfg, keep_preds=False):
    rows, preds = [], {}
    for p in path_list:
        lab = segment(IMG[p], cfg)
        r = eval_pair(lab, GTB[p], GTI[p]); r["path"] = p
        rows.append(r)
        if keep_preds: preds[p] = lab
    return pd.DataFrame(rows), preds

def summarize(res):
    tp, fp, fn = res.inst_tp.sum(), res.inst_fp.sum(), res.inst_fn.sum()
    return dict(mIoU=res.iou.mean(), mDice=res.dice.mean(), mPrec=res.precision.mean(), mRec=res.recall.mean(),
                inst_F1=2 * tp / max(1, 2 * tp + fp + fn), count_acc=(res.n_gt == res.n_pred).mean())

MIN_GAIN = 0.002
score = lambda s: 0.5 * (s["mIoU"] + s["inst_F1"])
TRIALS = [
 dict(id="T0", name="Baseline: gray + Otsu", change={},
      why="Simplest thing possible: eggs are usually brighter than their surroundings."),
 dict(id="T1", name="Edge based: Canny + close + fill contours", change=dict(method="canny"),
      why="Assignment hint: try edge detection first. Egg outline is a clean closed ellipse, so filling closed edge loops should work."),
 dict(id="T2", name="HSV: low saturation & high value", change=dict(method="hsv"),
      why="White eggs are bright and colourless, whereas skin / wood / cloth backgrounds are coloured; gray-only Otsu ignores that."),
 dict(id="T3", name="Lab: low chroma & high lightness", change=dict(method="lab"),
      why="Lab separates lightness from colour more perceptually uniformly than HSV; may be more robust to lighting."),
 dict(id="T4", name="Morphology: open 0.01 + close 0.02 + fill holes", change=dict(open=0.01, close=0.02, fill=True),
      why="Raw thresholds leave speckle and holes from specular highlights / egg-shell spots."),
 dict(id="T5", name="Aggressive morphology: open 0.03 + close 0.06", change=dict(open=0.03, close=0.06),
      why="Check whether more smoothing keeps helping or starts to distort shapes / merge objects."),
 dict(id="T6", name="Min component area 0.01", change=dict(min_area=0.01),
      why="Small false blobs (bright spots in background) hurt precision and instance counts."),
 dict(id="T7", name="Sweep saturation threshold scale", sweep=dict(s_scale=[0.6, 0.8, 1.0, 1.2, 1.5]),
      why="Otsu is only a heuristic split; scaling it up/down adjusts how much slightly-coloured shell/shadow is included."),
 dict(id="T8", name="Sweep value/lightness threshold scale", sweep=dict(v_scale=[0.5, 0.7, 0.85, 1.0, 1.15]),
      why="Same idea for brightness; shadowed side of the egg is darker."),
 dict(id="T9", name="Convex hull per egg", change=dict(convex=True),
      why="Eggs are convex; hull repairs dents caused by shadows and highlights."),
 dict(id="T10", name="Solidity filter >= 0.85", change=dict(solidity=0.85),
      why="Non-egg blobs (fingers, cloth folds) are typically non-convex."),
 dict(id="T11", name="Larger blur (0.03)", change=dict(blur=0.03),
      why="Stronger smoothing could suppress texture before thresholding."),
 dict(id="T12", name="Watershed split of touching eggs (ws_frac 0.5)", change=dict(watershed=True, ws_frac=0.5),
      why="Touching eggs form one component, which hurts the instance count / instance F1."),
 dict(id="T13", name="Watershed split, stricter seeds (ws_frac 0.7)", change=dict(watershed=True, ws_frac=0.7),
      why="ws_frac 0.5 may over-split single eggs; stricter seeds are more conservative."),
 dict(id="T14", name="GrabCut boundary refinement", change=dict(grabcut=True),
      why="Colour-model based refinement of the boundary should recover the true egg edge."),
]

best_cfg, best = dict(BASE_CFG), None
log, history = [], []
for T in TRIALS:
    if "sweep" in T:
        (k, vals), = T["sweep"].items()
        cands = [({**best_cfg, k: v}, {k: v}) for v in vals]
    else:
        cands = [({**best_cfg, **T["change"]}, T["change"] or {"(none)": "baseline"})]
    scored = []
    for cfg, delta in cands:
        t0 = time.time()
        res, _ = evaluate_cfg(DEV, cfg)
        scored.append((summarize(res), cfg, delta, time.time() - t0))
    j = int(np.argmax([score(s[0]) for s in scored]))
    for idx, (s, cfg, delta, secs) in enumerate(scored):
        prev = score(best) if best else float("nan")
        if best is None:
            decision, reason = "KEEP", "baseline"
        elif idx == j and score(s) >= prev + MIN_GAIN:
            decision, reason = "KEEP", f"dev score {score(s):.3f} vs {prev:.3f} (+{score(s)-prev:.3f}); mIoU {s['mIoU']:.3f}, inst_F1 {s['inst_F1']:.3f}"
        elif len(scored) > 1 and idx != j:
            decision, reason = "DISCARD", "worse sweep point than the best one in this sweep"
        else:
            decision, reason = "DISCARD", f"dev score {score(s):.3f} vs {prev:.3f} ({score(s)-prev:+.3f}); gain < {MIN_GAIN}; mIoU {s['mIoU']:.3f}, inst_F1 {s['inst_F1']:.3f}"
        log.append(dict(trial=T["id"], name=T["name"], what_changed=json.dumps(delta), why_tried=T["why"],
                        mIoU=s["mIoU"], mDice=s["mDice"], inst_F1=s["inst_F1"], count_acc=s["count_acc"],
                        secs=round(secs, 1), decision=decision, reason=reason))
    if score(scored[j][0]) >= (score(best) + MIN_GAIN if best else -1):
        best, best_cfg = scored[j][0], scored[j][1]
    history.append(dict(trial=T["id"], best_mIoU=best["mIoU"], best_score=score(best)))
    print(f"{T['id']:>3} {T['name'][:55]:55s} best-so-far dev score={score(best):.3f} (mIoU {best['mIoU']:.3f})  [{log[-len(scored)+j]['decision']}]")

FINAL_CFG = best_cfg
json.dump(FINAL_CFG, open(os.path.join(WORK_DIR, "best_cfg.json"), "w"), indent=1)
logdf = pd.DataFrame(log); logdf.to_csv(os.path.join(WORK_DIR, "report", "experiment_log.csv"), index=False)
print("\nFINAL CONFIG:", json.dumps(FINAL_CFG))

hd = pd.DataFrame(history)
plt.figure(figsize=(8, 3.5)); plt.plot(hd.trial, hd.best_score, "o-", label="score"); plt.plot(hd.trial, hd.best_mIoU, "s--", label="mIoU"); plt.legend(); plt.ylabel("best dev value so far")
plt.grid(alpha=.3); plt.title("Dev-set progress over trials"); plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "report", "trial_progress.png"), dpi=130); plt.close()

def md_table(d, cols=None):
    cols = cols or list(d.columns)
    f = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v).replace("|", "/")
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(f(v) for v in r) + " |" for r in d[cols].itertuples(index=False)]
    return "\n".join(out)

exp_md = "## Experiment log (dev set only)\n\n"
exp_md += (f"Selection rule: a change is kept only if it raises the dev score = 0.5*(mIoU + instance-F1) by >= {MIN_GAIN}; every trial starts from the "
           "best configuration so far. Parameters like kernel sizes are fractions of the image's shorter side.\n\n")
exp_md += md_table(logdf, ["trial", "name", "what_changed", "why_tried", "mIoU", "mDice", "inst_F1", "decision", "reason"])
exp_md += f"\n\n**Final configuration:** `{json.dumps(FINAL_CFG)}`\n\n![progress](trial_progress.png)\n"
open(os.path.join(WORK_DIR, "report", "experiment_log.md"), "w").write(exp_md)

res_dev, pred_dev = evaluate_cfg(DEV, FINAL_CFG, keep_preds=True)
res_eval, pred_eval = evaluate_cfg(EVAL, FINAL_CFG, keep_preds=True)
PRED = {**pred_dev, **pred_eval}
for p, lab in PRED.items():
    stem = df.loc[df.path == p, "stem"].iloc[0]
    cv2.imwrite(os.path.join(WORK_DIR, "predictions", stem + ".png"), ((lab > 0) * 255).astype(np.uint8))

summary = pd.DataFrame({"dev": summarize(res_dev), "eval": summarize(res_eval)}).T
summary["pixel_acc"] = [res_dev.pixel_acc.mean(), res_eval.pixel_acc.mean()]
summary["IoU_median"] = [res_dev.iou.median(), res_eval.iou.median()]
summary["IoU_std"] = [res_dev.iou.std(), res_eval.iou.std()]
summary.insert(0, "subset", summary.index); summary.insert(1, "n_images", [len(DEV), len(EVAL)])
print(summary.round(3).to_string(index=False))

GOOD_IOU = 0.85
def failure_type(r):
    if r.iou >= GOOD_IOU: return "good"
    if r.n_gt > 0 and r.pred_area == 0: return "missed egg"
    if r.n_gt == 0 and r.pred_area > 0: return "false positive (no egg in GT)"
    if r.n_pred != r.n_gt: return "wrong egg count (merge/split/extra blob)"
    if r.precision < r.recall - 0.05: return "over-segmentation (leaks into hand/background)"
    if r.recall < r.precision - 0.05: return "under-segmentation (parts of egg lost)"
    return "boundary mismatch"

E = res_eval.copy()
E["failure"] = [failure_type(r) for r in E.itertuples()]
E["gt_area_frac"] = [GTB[p].mean() for p in E.path]
E["brightness"] = [cv2.cvtColor(IMG[p], cv2.COLOR_BGR2GRAY).mean() for p in E.path]
E["saturation"] = [cv2.cvtColor(IMG[p], cv2.COLOR_BGR2HSV)[..., 1].mean() for p in E.path]
E["contrast_egg_bg"] = [abs(cv2.cvtColor(IMG[p], cv2.COLOR_BGR2GRAY)[GTB[p]].mean() -
                            cv2.cvtColor(IMG[p], cv2.COLOR_BGR2GRAY)[~GTB[p]].mean()) if GTB[p].any() and (~GTB[p]).any() else np.nan
                        for p in E.path]
E.to_csv(os.path.join(WORK_DIR, "report", "eval_per_image.csv"), index=False)

fig, ax = plt.subplots(2, 3, figsize=(17, 9))
ax[0, 0].hist(E.iou, bins=20, color="steelblue"); ax[0, 0].axvline(E.iou.mean(), color="r", ls="--", label=f"mean {E.iou.mean():.2f}")
ax[0, 0].set(title="IoU distribution (eval)", xlabel="IoU", ylabel="#images"); ax[0, 0].legend()
ax[0, 1].scatter(E.recall, E.precision, c=E.iou, cmap="viridis", s=18); ax[0, 1].plot([0, 1], [0, 1], "k:")
ax[0, 1].set(title="Precision vs recall (below diagonal = leaks, above = misses)", xlabel="recall", ylabel="precision")
ax[0, 2].scatter(E.gt_area_frac, E.iou, s=14); ax[0, 2].set(title="IoU vs egg size", xlabel="egg area / image area", ylabel="IoU")
ax[1, 0].scatter(E.contrast_egg_bg, E.iou, s=14); ax[1, 0].set(title="IoU vs egg/background brightness contrast", xlabel="|mean gray egg - mean gray bg|", ylabel="IoU")
ax[1, 1].scatter(E.saturation, E.iou, s=14); ax[1, 1].set(title="IoU vs mean image saturation", xlabel="mean saturation", ylabel="IoU")
fc = E.failure.value_counts()
ax[1, 2].barh(fc.index[::-1], fc.values[::-1], color="tomato"); ax[1, 2].set(title=f"Failure categories (IoU < {GOOD_IOU})", xlabel="#images")
plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, "report", "quantitative.png"), dpi=130); plt.close()

def overlay(img, mask, color=(255, 30, 60), a=0.55):
    out = img.copy(); out[mask] = (np.array(color)[::-1] * a + out[mask] * (1 - a)).astype(np.uint8); return out

def show_cases(sub, title, fname):
    fig, ax = plt.subplots(len(sub), 4, figsize=(14, 3.1 * len(sub)))
    ax = np.atleast_2d(ax)
    for i, r in enumerate(sub.itertuples()):
        img, gt, pr = IMG[r.path], GTB[r.path], PRED[r.path] > 0
        err = np.zeros(img.shape, np.uint8)
        err[gt & pr] = (0, 200, 0); err[~gt & pr] = (0, 0, 255); err[gt & ~pr] = (255, 0, 0)
        panels = [img, overlay(img, gt), overlay(img, pr), err]
        for j, (im, t) in enumerate(zip(panels, ["image", "GT (SAM)", "prediction", "error map"])):
            ax[i, j].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax[i, j].axis("off")
            ax[i, j].set_title(t if j else f"{t}  IoU={r.iou:.2f}\n{r.failure}", fontsize=8)
    plt.suptitle(title); plt.tight_layout(); plt.savefig(os.path.join(WORK_DIR, "report", fname), dpi=110); plt.close()

show_cases(E.sort_values("iou").head(8), "Worst eval samples (green=TP, red=FP, blue=FN)", "qualitative_worst.png")
show_cases(E.sort_values("iou").tail(3), "Best eval samples", "qualitative_best.png")

def pct(x): return f"{100*x:.1f}%"
corr = lambda c: E[[c, "iou"]].corr(method="spearman").iloc[0, 1]
fails = E[E.failure != "good"].failure.value_counts()
rep = f"""# Egg segmentation: classical image processing vs SAM ground truth

## 1. Setup
- {len(df)} images, split by **group** into dev ({len(DEV)}) / eval ({len(EVAL)}) = 4:1 (seed {SEED}).
  Groups = same base filename (augmentation suffix removed) OR perceptual-hash distance <= {HASH_T} under any flip/rotation;
  the original train/valid/test folders were merged first because they can contain augmented copies of the same photo.
  Nearest dev<->eval Hamming distance: {cross.min()}.
- Ground truth = Task 1 masks (Grounding-DINO + SAM). Predictions = final classical pipeline.
- Final configuration: `{json.dumps(FINAL_CFG)}`
- All tuning used the dev set only; eval numbers below were computed once at the end.

## 2. Quantitative results
{md_table(summary.round(3))}

Metrics: IoU / Dice / precision / recall / pixel accuracy are per image (averaged); `inst_F1` = instance-level F1 with
Hungarian matching at IoU>=0.5; `count_acc` = fraction of images whose number of eggs is exactly right.

Key numbers on the eval set:
- mean IoU {E.iou.mean():.3f}, median {E.iou.median():.3f}; {pct((E.iou>=0.9).mean())} of images have IoU >= 0.9 and {pct((E.iou<0.5).mean())} have IoU < 0.5.
- mean precision {E.precision.mean():.3f} vs mean recall {E.recall.mean():.3f} -> the method tends to {"over-segment (include extra pixels)" if E.precision.mean() < E.recall.mean() else "under-segment (lose egg pixels)"}.
- Spearman correlation of IoU with: egg size {corr('gt_area_frac'):+.2f}, egg/background contrast {corr('contrast_egg_bg'):+.2f}, image brightness {corr('brightness'):+.2f}, saturation {corr('saturation'):+.2f}.

## 3. Failure analysis (eval images with IoU < {GOOD_IOU}: {int(len(E)-(E.failure=='good').sum())} of {len(E)})
{md_table(fails.rename_axis('failure type').reset_index(name='#images')) if len(fails) else 'No failures.'}

Reading the error maps: red = pixels the classical method added that SAM did not (leakage into hand/background/shadow),
blue = egg pixels the classical method missed (shadowed side, specular highlight, shell spots).

**What goes wrong (check against your plots and edit):**
1. Global colour/brightness thresholds cannot separate the egg from bright or low-saturation neighbours (skin highlights, white cloth, glare) -> leakage (red).
2. Shadowed or glossy parts of the egg fall outside the brightness/saturation range -> missing pixels (blue); hull/closing only partially fixes this.
3. Touching or overlapping eggs form one component -> wrong counts and low instance F1 unless watershed splits them correctly.
4. Per-image Otsu fails when the egg occupies a very small or very large share of the image (bimodal assumption breaks).
5. Brown / coloured eggs violate the "egg = bright & colourless" assumption.

## 4. Ideas for improvement
- Local / adaptive thresholds or a per-image 2-3 cluster k-means in Lab, picking the cluster that is most egg-like (convex, ellipse-like).
- Ellipse fitting (RANSAC / Hough) on edges to impose the egg shape prior; reject candidates with poor elliptical fit.
- Colour-model refinement (GrabCut / GMM) seeded from the confident core of the mask.
- Illumination normalisation (CLAHE, shadow removal) before thresholding.
- Better touching-egg splitting (distance-transform peaks + ellipse fit); trained pixel classifier (random forest on colour/texture features) as a stronger-but-still-classical alternative.

{exp_md}
"""
open(os.path.join(WORK_DIR, "report", "report.md"), "w").write(rep)
print(rep[:1500], "\n...\nSaved to", os.path.join(WORK_DIR, "report"))
