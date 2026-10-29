"""在 logit 空间里搜索成员权重（<3 成员，网格搜索），并用直方图法快速扫阈值。

    python tools/tune_blend2.py --members ST_A L4_unetd_100 --min-size 128
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from select_model import gt_lookup, hist_of, load_gt, load_run, metrics_from_hist  # noqa


def logit(p):
    p = np.clip(p.astype(np.float32), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def q(z):
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(np.rint(p * 255), 0, 255).astype(np.uint8)


def postprocess(masks, min_size):
    if not min_size:
        return masks
    out = np.empty_like(masks)
    for i, m in enumerate(masks):
        lab, n = ndi.label(m)
        if n:
            sizes = ndi.sum(m, lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sizes >= min_size
            m = keep[lab]
        out[i] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--min-size", type=int, default=128)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()
    z = {}
    for kind in ("v1", "v2"):
        arr, names, y = {}, None, None
        for r in args.members:
            n, pr = load_run(r, kind)
            if names is None:
                names, y = n, load_gt(n, gt_lookup())
            elif n != names:
                pr = pr[[n.index(x) for x in names]]
            arr[r] = logit(pr)
        z[kind] = (arr, y)

    k = len(args.members)
    grid = np.arange(0.0, 1.0001, args.step)
    best = []
    for w in itertools.product(grid, repeat=k - 1):
        s = sum(w)
        if s > 1.0:
            continue
        ws = list(w) + [1.0 - s]
        if any(x < 0.0 for x in ws):
            continue
        rows = {}
        for kind, (arr, y) in z.items():
            zz = sum(wi * arr[r] for wi, r in zip(ws, args.members))
            m = metrics_from_hist(hist_of(q(zz), y))
            b = int(np.nanargmax(m["iou"]))
            rows[kind] = (float(m["iou"][b]), b, m)
        avg = (rows["v1"][0] + rows["v2"][0]) / 2
        best.append((avg, ws, rows))
    best.sort(key=lambda t: -t[0])
    print(f"{'weights':40s} {'v1':>7s} {'v2':>7s} {'avg':>7s} {'thr1':>5s} {'thr2':>5s}")
    for avg, ws, rows in best[:10]:
        print(f"{str([round(x, 2) for x in ws]):40s} {rows['v1'][0]:7.4f} {rows['v2'][0]:7.4f} "
              f"{avg:7.4f} {rows['v1'][1]/255:5.2f} {rows['v2'][1]/255:5.2f}")

    # 用最佳权重检查后处理增益（在 v1 与 v2 上分别取各自最优阈值）
    avg, ws, rows = best[0]
    out = {"members": args.members, "weights": ws,
           "v1": {"iou": rows["v1"][0], "thr": rows["v1"][1] / 255},
           "v2": {"iou": rows["v2"][0], "thr": rows["v2"][1] / 255}}
    for kind, (arr, y) in z.items():
        zz = sum(wi * arr[r] for wi, r in zip(ws, args.members))
        b = rows[kind][1]
        pred = q(zz) >= b
        for ms in (0, 64, 128, 192, 256):
            pp = postprocess(pred, ms)
            m = metrics_from_hist(hist_of((pp.astype(np.uint8) * 255), y))
            out.setdefault(f"{kind}_post", {})[str(ms)] = round(float(m["iou"][255]), 4)
        print(f"[{kind}] min_size -> IoU:", out[f"{kind}_post"])
    dest = ROOT / "docs" / "blend_best.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {dest}")


if __name__ == "__main__":
    main()
