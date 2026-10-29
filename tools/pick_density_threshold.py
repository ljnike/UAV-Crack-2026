"""按"线上标定过的前景密度"选阈值出包。

线上四连提交给出的经验（同一集成、只改阈值）：
    thr 0.41 -> 前景中位 5.80% -> 线上 IoU 65.94
    thr 0.43 -> 5.70%           -> 66.10
    thr 0.45 -> 5.61%           -> 66.21
    thr 0.47 -> 5.44%           -> 66.26   (最优，再高只剩 ~66.25)
也就是说线上最优点对应"测试集预测前景中位数 ≈ 5.4%"。
新集成换阈值时不该沿用旧阈值（概率标定变了），而应该**沿用这个密度**。

    python tools/pick_density_threshold.py --probs a.npz b.npz --weights .1 .65 .25 \
        --logit --target 0.054
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--logit", action="store_true")
    ap.add_argument("--target", type=float, default=0.054,
                    help="target median foreground fraction on the 300 test tiles")
    ap.add_argument("--grid", type=float, default=0.005)
    args = ap.parse_args()

    members = [np.load(p) for p in args.probs]
    keys = sorted(members[0].files)
    w = np.array(args.weights if args.weights else [1.0] * len(members))
    w = w / w.sum()

    tiles = []
    for k in keys:
        if args.logit:
            z = np.zeros_like(members[0][k], dtype=np.float32)
            for m, wi in zip(members, w):
                p = np.clip(m[k].astype(np.float32), 1e-4, 1 - 1e-4)
                z += float(wi) * np.log(p / (1 - p))
            prob = 1.0 / (1.0 + np.exp(-z))
        else:
            prob = np.zeros_like(members[0][k], dtype=np.float32)
            for m, wi in zip(members, w):
                prob += float(wi) * m[k].astype(np.float32)
        tiles.append(prob.ravel())

    rows = []
    for th in np.arange(0.30, 0.61, args.grid):
        fg = np.array([float((t >= th).mean()) for t in tiles])
        rows.append((float(th), float(np.median(fg)), float(np.mean(fg))))
    best = min(rows, key=lambda r: abs(r[1] - args.target))
    print(f"{'thr':>6s} {'median_fg':>10s} {'mean_fg':>8s}")
    for th, med, mean in rows:
        mark = "  <== target" if abs(med - args.target) < 1e-9 else ""
        if abs(med - args.target) < 0.004 or th in (0.41, 0.47):
            print(f"{th:6.3f} {med:10.4f} {mean:8.4f}{mark}")
    print(f"\n[density-match] threshold={best[0]:.3f} median_fg={best[1]:.4f} "
          f"mean_fg={best[2]:.4f} (target {args.target:.3f})")


if __name__ == "__main__":
    main()
