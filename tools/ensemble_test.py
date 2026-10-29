"""把若干成员的测试集概率图组合成提交包（不用重新推理，秒级）。

前提：每个成员先用 predictz 单独导出一份 300 张的 test 概率 npz：
    python predictz.py --ckpt runs/X/best.pt --out sub/tmp --threshold 0.5 --flip \
        --save-probs sub/test_probs_X.npz
本脚本再按给定成员/权重/阈值/后处理生成 masks 并打包 zip。

用法:
    python tools/ensemble_test.py --probs sub/test_probs_L1.npz sub/test_probs_L4.npz \
        sub/test_probs_N1.npz sub/test_probs_N2.npz --threshold 0.43 \
        --out-masks sub/masks_final --out-zip sub/result_final.zip
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]


def load(p: Path):
    d = np.load(p)
    return {str(k): d[k].astype(np.float32) for k in d.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=0.43)
    ap.add_argument("--min-size", type=int, default=64)
    ap.add_argument("--close-r", type=int, default=0)
    ap.add_argument("--fg-value", type=int, default=1)
    ap.add_argument("--logit", action="store_true",
                    help="blend in logit space: p = sigmoid(sum w_i * logit(p_i))")
    ap.add_argument("--out-masks", required=True)
    ap.add_argument("--out-zip", default=None)
    args = ap.parse_args()

    members = [load(Path(p)) for p in args.probs]
    keys = sorted(members[0])
    for m, path in zip(members[1:], args.probs[1:]):
        if sorted(m) != keys:
            raise SystemExit(f"tile set mismatch in {path}")
    w = np.array(args.weights if args.weights else [1.0] * len(members), dtype=np.float64)
    w = w / w.sum()

    out_dir = Path(args.out_masks)
    out_dir.mkdir(parents=True, exist_ok=True)
    fgs, empties = [], 0
    for k in keys:
        if args.logit:
            z = np.zeros_like(members[0][k], dtype=np.float32)
            for m, wi in zip(members, w):
                p = np.clip(m[k], 1e-4, 1 - 1e-4)
                z += float(wi) * np.log(p / (1 - p))
            prob = 1.0 / (1.0 + np.exp(-z))
        else:
            prob = np.zeros_like(members[0][k], dtype=np.float32)
            for m, wi in zip(members, w):
                prob += wi * m[k]
        mask = (prob >= args.threshold).astype(np.uint8)
        if args.close_r > 0:
            st = np.ones((2 * args.close_r + 1,) * 2, bool)
            mask = ndi.binary_closing(mask > 0, structure=st).astype(np.uint8)
        if args.min_size > 0:
            lab, n = ndi.label(mask > 0)
            if n:
                sizes = ndi.sum(mask, lab, range(1, n + 1))
                keep = np.zeros(n + 1, bool)
                keep[1:] = sizes >= args.min_size
                mask = keep[lab].astype(np.uint8)
        if mask.sum() == 0:
            empties += 1
        fgs.append(float(mask.mean()))
        Image.fromarray(mask * args.fg_value).save(out_dir / f"{k}.png")

    stats = {"n": len(keys), "threshold": args.threshold, "min_size": args.min_size,
             "close_r": args.close_r, "fg_value": args.fg_value,
             "members": [Path(p).stem for p in args.probs], "weights": w.tolist(),
             "mean_fg": float(np.mean(fgs)), "median_fg": float(np.median(fgs)),
             "empty_masks": empties}
    (out_dir / "predict_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))

    if args.out_zip:
        zp = Path(args.out_zip)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for k in keys:
                z.write(out_dir / f"{k}.png", f"result/{k}.png")
        with zipfile.ZipFile(zp) as z:
            bad = z.testzip()
            names = z.namelist()
        print(f"[zip] {zp} entries={len(names)} testzip={'OK' if bad is None else bad} "
              f"size={zp.stat().st_size/1e6:.2f} MB first={names[0]}")


if __name__ == "__main__":
    main()
