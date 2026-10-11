import os
import json
import numpy as np
import cv2

def overlay(img, mask, color=(255, 30, 60), a=0.55):
    out = img.copy()
    out[mask] = (np.array(color)[::-1] * a + out[mask] * (1 - a)).astype(np.uint8)
    return out

def odd(k): k = int(round(k)); return max(1, k | 1)
def ksz(frac, shape): return odd(frac * min(shape[:2]))
def ell(k): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

BASE_CFG = dict(method="otsu_gray",
                blur=0.01,
                s_scale=1.0, v_scale=1.0,
                canny_sigma=0.33,
                open=0.0, close=0.0,
                fill=False,
                min_area=0.002, max_area=0.9,
                solidity=0.0,
                convex=False,
                watershed=False, ws_frac=0.5,
                grabcut=False)

def foreground(img, c):
    k = ksz(c["blur"], img.shape) if c["blur"] > 0 else 1
    im = cv2.GaussianBlur(img, (k, k), 0) if k > 1 else img
    otsu = lambda ch: cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if c["method"] == "otsu_gray":
        return gray > otsu(gray) * c["v_scale"]
    if c["method"] == "hsv":
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        return (hsv[..., 1] < otsu(hsv[..., 1]) * c["s_scale"]) & (hsv[..., 2] > otsu(hsv[..., 2]) * c["v_scale"])
    if c["method"] == "lab":
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128).astype(np.uint8)
        L = lab[..., 0].astype(np.uint8)
        return (chroma < otsu(chroma) * c["s_scale"]) & (L > otsu(L) * c["v_scale"])
    if c["method"] == "canny":
        med = np.median(gray)
        e = cv2.Canny(gray, int(max(0, (1 - c["canny_sigma"]) * med)), int(min(255, (1 + c["canny_sigma"]) * med)))
        e = cv2.morphologyEx(cv2.dilate(e, ell(3)), cv2.MORPH_CLOSE, ell(ksz(0.02, img.shape)))
        cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        m = np.zeros(gray.shape, np.uint8)
        A = gray.size
        for cnt in cnts:
            if c["min_area"] * A < cv2.contourArea(cnt) < 0.5 * A:
                cv2.drawContours(m, [cnt], -1, 1, cv2.FILLED)
        return m > 0
    raise ValueError(c["method"])

def split_watershed(img, m, frac):
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    n, cc = cv2.connectedComponents(m)
    markers = np.zeros(m.shape, np.int32); k = 1
    for i in range(1, n):
        comp = cc == i
        seeds = ((dist > frac * dist[comp].max()) & comp).astype(np.uint8)
        n2, l2 = cv2.connectedComponents(seeds)
        for j in range(1, n2):
            markers[l2 == j] = k; k += 1
    bg = k
    markers[cv2.dilate(m, ell(5)) == 0] = bg
    ws = cv2.watershed(img, markers)
    ws[(ws == bg) | (ws < 0)] = 0
    out = np.zeros_like(ws)
    for new, old in enumerate([v for v in np.unique(ws) if v > 0], start=1):
        out[ws == old] = new
    return out

def refine_grabcut(img, lab):
    m = (lab > 0).astype(np.uint8)
    band = ksz(0.03, img.shape)
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(m, ell(band)) > 0] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(m, ell(band)) > 0] = cv2.GC_FGD
    if not (gc == cv2.GC_FGD).any():
        return lab
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return lab
    new = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    grown = cv2.dilate(lab.astype(np.float32), ell(band))
    return np.where(new, np.where(lab > 0, lab, grown.astype(np.int32)), 0).astype(np.int32)

def segment(img, cfg):
    """BGR image -> int32 instance label map (0 = background, 1..N = eggs)."""
    c = {**BASE_CFG, **cfg}
    A = img.shape[0] * img.shape[1]
    m = foreground(img, c).astype(np.uint8)
    if c["open"] > 0:  m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  ell(ksz(c["open"],  img.shape)))
    if c["close"] > 0: m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell(ksz(c["close"], img.shape)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(m)
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if not (c["min_area"] * A <= a <= c["max_area"] * A):
            continue
        hull = cv2.convexHull(cnt); ha = cv2.contourArea(hull)
        if ha > 0 and a / ha < c["solidity"]:
            continue
        comp = np.zeros_like(m)
        cv2.drawContours(comp, [hull if c["convex"] else cnt], -1, 1, cv2.FILLED)
        if not (c["fill"] or c["convex"]):
            comp &= m
        keep |= comp
    if not keep.any():
        return np.zeros(m.shape, np.int32)
    lab = split_watershed(img, keep, c["ws_frac"]) if c["watershed"] else cv2.connectedComponents(keep)[1]
    if c["grabcut"]:
        lab = refine_grabcut(img, lab.astype(np.int32))
    return lab.astype(np.int32)
