import os
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from common import segment, overlay

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="outputs/task2_3/best_cfg.json")
parser.add_argument("--video", default="data/egg_video.mp4")
parser.add_argument("--url", default="https://www.youtube.com/watch?v=Y4JyV94y2_A")
parser.add_argument("--out", default="outputs/task4")
parser.add_argument("--max-seconds", type=float, default=60)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--max-side", type=int, default=640)
parser.add_argument("--override", default="{}")
args = parser.parse_args()

VIDEO_URL = args.url
VIDEO_PATH = args.video
OUT = args.out
VID_MAX_SIDE = args.max_side
MAX_SECONDS = args.max_seconds
STRIDE = args.stride
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.dirname(VIDEO_PATH) or ".", exist_ok=True)

if not os.path.exists(VIDEO_PATH):
    subprocess.run(["yt-dlp", "-f", "best[height<=720][ext=mp4]/best[height<=720]/best",
                    "-o", VIDEO_PATH, VIDEO_URL], check=True)
cap = cv2.VideoCapture(VIDEO_PATH)
FPS, NFRAMES = cap.get(cv2.CAP_PROP_FPS) or 30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video: {int(cap.get(3))}x{int(cap.get(4))} @ {FPS:.1f} fps, {NFRAMES} frames ({NFRAMES/FPS:.0f}s)")
cap.release()

FINAL_CFG = json.load(open(args.cfg))
VIDEO_CFG = {**FINAL_CFG, **json.loads(args.override)}

def read_frame(cap, idx=None):
    if idx is not None: cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    if not ok: return None
    s = VID_MAX_SIDE / max(f.shape[:2])
    return cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else f

cap = cv2.VideoCapture(VIDEO_PATH)
sel = np.linspace(0, max(0, NFRAMES - 2), 4).astype(int)
fig, ax = plt.subplots(2, 4, figsize=(16, 6))
for j, fi in enumerate(sel):
    f = read_frame(cap, fi); lab = segment(f, VIDEO_CFG)
    ax[0, j].imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); ax[0, j].axis("off"); ax[0, j].set_title(f"frame {fi}")
    ax[1, j].imshow(cv2.cvtColor(overlay(f, lab > 0), cv2.COLOR_BGR2RGB)); ax[1, j].axis("off"); ax[1, j].set_title(f"{lab.max()} eggs found")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "preview.png"), dpi=110); plt.close(); cap.release()

class Track:
    def __init__(self, tid, c):
        self.id, self.c, self.v = tid, np.array(c, float), np.zeros(2)
        self.missed, self.hits, self.trail = 0, 1, [tuple(c)]

class EggTracker:
    def __init__(self, max_dist_frac=0.12, max_missed=15, min_hits=3):
        self.tracks, self.next_id = [], 1
        self.max_dist_frac, self.max_missed, self.min_hits = max_dist_frac, max_missed, min_hits

    def update(self, centroids, diag):
        gate = self.max_dist_frac * diag
        pred = [t.c + t.v for t in self.tracks]
        assigned = {}
        if self.tracks and len(centroids):
            cost = np.linalg.norm(np.array(pred)[:, None] - np.array(centroids)[None], axis=2)
            cost[cost > gate] = 1e6
            for ti, di in zip(*linear_sum_assignment(cost)):
                if cost[ti, di] < 1e6:
                    assigned[di] = self.tracks[ti]
        for di, t in assigned.items():
            c = np.array(centroids[di], float)
            t.v = 0.6 * t.v + 0.4 * (c - t.c); t.c = c; t.missed = 0; t.hits += 1; t.trail.append(tuple(c))
        matched = {id(t) for t in assigned.values()}
        for t in self.tracks:
            if id(t) not in matched:
                t.missed += 1; t.c = t.c + t.v
        for di, c in enumerate(centroids):
            if di not in assigned:
                t = Track(self.next_id, c); self.next_id += 1
                self.tracks.append(t); assigned[di] = t
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return {di: t for di, t in assigned.items() if t.hits >= self.min_hits}

def color_for(i):
    rng = np.random.RandomState(i * 7919 % 2**31); return tuple(int(x) for x in rng.randint(60, 255, 3))

cap = cv2.VideoCapture(VIDEO_PATH)
first = read_frame(cap, 0); Hh, Ww = first.shape[:2]; diag = float(np.hypot(Hh, Ww))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
OUT_RAW, OUT_MP4 = os.path.join(OUT, "tracked_raw.mp4"), os.path.join(OUT, "tracked_eggs.mp4")
writer = cv2.VideoWriter(OUT_RAW, cv2.VideoWriter_fourcc(*"mp4v"), FPS / STRIDE, (Ww, Hh))
tracker, records, fi = EggTracker(), [], 0
max_frames = int(MAX_SECONDS * FPS)
pbar = tqdm(total=min(NFRAMES, max_frames), desc="tracking")
while fi < max_frames:
    f = read_frame(cap)
    if f is None: break
    if fi % STRIDE == 0:
        lab = segment(f, VIDEO_CFG)
        ids = [i for i in np.unique(lab) if i > 0]
        cents, masks = [], []
        for i in ids:
            m = lab == i; M = cv2.moments(m.astype(np.uint8), True)
            cents.append((M["m10"] / M["m00"], M["m01"] / M["m00"])); masks.append(m)
        tr = tracker.update(cents, diag)
        vis = f.copy()
        for di, t in tr.items():
            col = color_for(t.id); m = masks[di]
            vis[m] = (0.5 * np.array(col) + 0.5 * vis[m]).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, col, 2)
            if len(t.trail) > 1:
                cv2.polylines(vis, [np.array(t.trail[-40:], np.int32)], False, col, 2)
            cx, cy = map(int, cents[di])
            cv2.putText(vis, f"egg {t.id}", (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(vis, f"egg {t.id}", (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            x, y, w_, h_ = cv2.boundingRect(m.astype(np.uint8))
            records.append(dict(frame=fi, id=t.id, cx=cents[di][0], cy=cents[di][1], x=x, y=y, w=w_, h=h_, area=int(m.sum())))
        writer.write(vis)
    fi += 1; pbar.update(1)
pbar.close(); writer.release(); cap.release()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_RAW, "-vcodec", "libx264", "-pix_fmt", "yuv420p", OUT_MP4])
trk = pd.DataFrame(records); trk.to_csv(os.path.join(OUT, "egg_tracks.csv"), index=False)
if len(trk):
    L = trk.groupby("id").size()
    print(f"{trk.id.nunique()} tracks | mean length {L.mean():.0f} frames | longest {L.max()} | "
          f"tracks shorter than 15 frames (likely noise / ID switches): {(L < 15).sum()}")

if len(trk):
    plt.figure(figsize=(8, 6)); plt.imshow(cv2.cvtColor(first, cv2.COLOR_BGR2RGB), alpha=0.6)
    for tid, g in trk.groupby("id"):
        plt.plot(g.cx, g.cy, "-", lw=2, label=f"egg {tid}"); plt.text(g.cx.iloc[0], g.cy.iloc[0], str(tid), color="k")
    plt.title("Egg trajectories"); plt.axis("off"); plt.savefig(os.path.join(OUT, "trajectories.png"), dpi=130); plt.close()
