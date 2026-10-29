"""诊断：测试集与本地验证集的"裂缝宽度/前景密度"分布是否不同（解释线上 P/R 偏移）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

MEM = ["ST_A", "E_swins_prec", "L4_unetd_100"]
W = {"ST_A": 0.10, "E_swins_prec": 0.65, "L4_unetd_100": 0.25}


def lg(p):
    p = np.clip(p.astype(np.float32), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def tile_width(mask: np.ndarray) -> float:
    """用距离变换估计平均裂缝半宽（只统计骨架附近）。"""
    if mask.sum() == 0:
        return float("nan")
    dt = ndi.distance_transform_edt(mask)
    return float(dt[mask].mean())


def stats(probs: np.ndarray, tag: str, band: float = 0.47) -> None:
    fg, width = [], []
    for p in probs:
        m = p >= band
        fg.append(float(m.mean()))
        width.append(tile_width(m))
    fg = np.array(fg)
    wd = np.array(width)
    print(f"{tag:10s} n={len(fg):3d}  fg: mean={fg.mean():.4f} median={np.median(fg):.4f} "
          f"p90={np.percentile(fg, 90):.4f} | half-width: median={np.nanmedian(wd):.3f} "
          f"p90={np.nanpercentile(wd, 90):.3f} | 宽裂缝占比(fg>8%)={np.mean(fg > 0.08):.3f}")


def main() -> None:
    # 本地
    for kind in ("v1", "v2"):
        names = None
        z = None
        for r in MEM:
            sys.path.insert(0, str(ROOT / "tools"))
            from select_model import load_run  # noqa
            n, pr = load_run(r, kind)
            if names is None:
                names = n
            elif n != names:
                pr = pr[[n.index(x) for x in names]]
            z = W[r] * lg(pr) if z is None else z + W[r] * lg(pr)
        stats(1 / (1 + np.exp(-z)), kind)

    # 测试集（各成员的多尺度/单尺度概率）
    files = {"ST_A": "sub/test_probs_STA_ms.npz",
             "E_swins_prec": "sub/test_probs_E_ms.npz",
             "L4_unetd_100": "sub/test_probs_L4.npz"}
    names = None
    z = None
    for r, f in files.items():
        d = np.load(ROOT / f)
        nn = sorted(d.files)
        if names is None:
            names = nn
        elif nn != names:
            raise SystemExit("tile mismatch")
        z = W[r] * lg(np.stack([d[k] for k in nn]).astype(np.float32)) if z is None else \
            z + W[r] * lg(np.stack([d[k] for k in nn]).astype(np.float32))
    stats(1 / (1 + np.exp(-z)), "test(300)")


if __name__ == "__main__":
    main()
