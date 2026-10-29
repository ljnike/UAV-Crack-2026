"""最终候选配置的对比（logit 加权 + 可选 64px 过滤），输出 v1/v2 双榜与 P/R。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from select_model import gt_lookup, hist_of, load_gt, load_run, metrics_from_hist  # noqa

CANDS = [
    (["ST_A", "L4_unetd_100", "E_swins_prec"], [0.10, 0.35, 0.55]),
    (["ST_E", "L4_unetd_100", "E_swins_prec"], [0.15, 0.35, 0.50]),
    (["E_swins_prec", "L4_unetd_100", "F_swinv2s"], [0.45, 0.30, 0.25]),
    (["ST_A", "L4_unetd_100", "F_swinv2s"], [0.15, 0.30, 0.55]),
    (["ST_A", "L4_unetd_100", "E_swins_prec", "F_swinv2s"], [0.10, 0.25, 0.30, 0.35]),
    (["ST_A", "L4_unetd_100", "E_swins_prec", "F_swinv2s"], [0.05, 0.25, 0.35, 0.35]),
    (["ST_E", "L4_unetd_100", "E_swins_prec", "F_swinv2s"], [0.15, 0.25, 0.25, 0.35]),
    (["ST_A", "E_swins_prec", "F_swinv2s"], [0.20, 0.40, 0.40]),
    (["ST_F", "L4_unetd_100", "E_swins_prec"], [0.20, 0.30, 0.50]),
    (["ST_F", "ST_A", "L4_unetd_100", "E_swins_prec"], [0.25, 0.05, 0.25, 0.45]),
]


def blend(kind, members, weights):
    names = z = None
    for r, w in zip(members, weights):
        n, pr = load_run(r, kind)
        if pr is None:
            return None
        if names is None:
            names = n
            y = load_gt(n, gt_lookup())
        elif set(n) != set(names):
            return None
        elif n != names:
            pr = pr[[n.index(x) for x in names]]
        p = np.clip(pr.astype(np.float32), 1e-4, 1 - 1e-4)
        z = float(w) * np.log(p / (1 - p)) if z is None else z + float(w) * np.log(p / (1 - p))
    prob = 1.0 / (1.0 + np.exp(-z))
    return prob, y, names


def post(masks, min_size):
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
    rows = []
    for members, weights in CANDS:
        if not all((ROOT / "runs" / m / "val_probs_v2_tta4.npz").exists() for m in members):
            print(f"[skip] {members}（缺缓存）")
            continue
        res = {}
        for kind in ("v1", "v2"):
            prob, y, _ = blend(kind, members, weights)
            q = np.clip(np.rint(prob * 255), 0, 255).astype(np.uint8)
            m = metrics_from_hist(hist_of(q, y))
            b = int(np.nanargmax(m["iou"]))
            res[kind] = dict(iou=float(m["iou"][b]), thr=b / 255,
                             miou=float(m["miou"][b]), f1=float(m["f1"][b]),
                             P=float(m["precision"][b]), R=float(m["recall"][b]))
            # 64px 过滤
            pred = q >= b
            m2 = metrics_from_hist(hist_of(post(pred, 64).astype(np.uint8) * 255, y))
            res[kind]["iou64"] = float(m2["iou"][255])
        res["avg"] = (res["v1"]["iou64"] + res["v2"]["iou64"]) / 2
        res["members"] = members
        res["weights"] = weights
        rows.append(res)
        print(f"{'+'.join(x[:6] for x in members):44s} "
              f"v1={res['v1']['iou']:.4f}/64:{res['v1']['iou64']:.4f} "
              f"v2={res['v2']['iou']:.4f}/64:{res['v2']['iou64']:.4f} "
              f"avg64={res['avg']:.4f} thr={res['v1']['thr']:.2f}/{res['v2']['thr']:.2f} "
              f"P={res['v1']['P']:.3f} R={res['v1']['R']:.3f}")
    rows.sort(key=lambda r: -r["avg"])
    print("\n[best]", rows[0]["members"], rows[0]["weights"], round(rows[0]["avg"], 4))
    (ROOT / "docs" / "final_candidates.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
