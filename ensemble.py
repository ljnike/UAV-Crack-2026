# -*- coding: utf-8 -*-
"""v4 logit-weighted ensemble: per-member probability maps -> masks -> result.zip.

This is step 2 of the two-step reproduction recipe (step 1 is `predictz.py`,
which exports one .npz probability file per member; that script is part of the
training code package and is not duplicated here).

Usage
    python ensemble.py --probs sub/p_STA.npz sub/p_E.npz sub/p_L4.npz sub/p_H.npz sub/p_H2.npz \
                       --config configs/ensemble.json --out result --zip result.zip

Each .npz must map "<tile stem>" -> (378, 672) float probability array.
"""

import argparse
import json
import os
import sys
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import fusion  # noqa: E402
from src.postprocess import binarize, foreground_ratio  # noqa: E402


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs=5, required=True,
                    metavar=("ST_A", "E", "L4", "H", "H2"),
                    help="五个成员各自的概率 npz，顺序固定")
    ap.add_argument("--config", default="configs/ensemble.json")
    ap.add_argument("--out", default="result")
    ap.add_argument("--zip", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    thr = cfg["threshold"] if args.threshold is None else args.threshold
    min_area = cfg["min_component_area_px"] if args.min_area is None else args.min_area

    maps = dict(zip(("ST_A", "E", "L4", "H", "H2"), (load_npz(p) for p in args.probs)))
    stems = sorted(set.intersection(*(set(m) for m in maps.values())))
    print(f"[ensemble] {len(stems)} 张; threshold={thr}, min_area={min_area}px")
    os.makedirs(args.out, exist_ok=True)

    ratios = []
    for stem in stems:
        probs = {k: maps[k][stem] for k in maps}
        prob = fusion.fuse(probs, cfg)
        mask = binarize(prob, thr, min_area)
        ratios.append(foreground_ratio(mask))
        Image.fromarray(mask, mode="L").save(os.path.join(args.out, stem + ".png"))

    r = np.asarray(ratios)
    print(f"[ensemble] 前景占比 mean={r.mean():.4f} median={np.median(r):.4f} "
          f"min={r.min():.4f} max={r.max():.4f}")

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for stem in stems:
                z.write(os.path.join(args.out, stem + ".png"), arcname=f"result/{stem}.png")
        print(f"[ensemble] -> {args.zip} ({os.path.getsize(args.zip)/1024/1024:.2f} MB)")
        print("[ensemble] 建议再跑: python check_result.py --result %s --zip %s" % (args.out, args.zip))


if __name__ == "__main__":
    main()
