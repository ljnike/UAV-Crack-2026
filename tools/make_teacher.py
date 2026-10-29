"""把若干成员的测试集概率按 logit 权重融合成一份"教师"npz，供 self_train.py --probs 使用。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--logit", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    members = [np.load(p) for p in args.probs]
    keys = sorted(members[0].files)
    for p, m in zip(args.probs[1:], members[1:]):
        if sorted(m.files) != keys:
            raise SystemExit(f"tile mismatch in {p}")
    w = np.array(args.weights if args.weights else [1.0] * len(members))
    w = w / w.sum()

    out = {}
    for k in keys:
        if args.logit:
            z = np.zeros_like(members[0][k], dtype=np.float32)
            for m, wi in zip(members, w):
                p = np.clip(m[k].astype(np.float32), 1e-4, 1 - 1e-4)
                z += float(wi) * np.log(p / (1 - p))
            out[k] = (1.0 / (1.0 + np.exp(-z))).astype(np.float16)
        else:
            p = np.zeros_like(members[0][k], dtype=np.float32)
            for m, wi in zip(members, w):
                p += float(wi) * m[k].astype(np.float32)
            out[k] = p.astype(np.float16)
    np.savez_compressed(args.out, **out)
    print(f"[teacher] {args.out} tiles={len(keys)} weights={np.round(w, 3).tolist()} logit={args.logit}")


if __name__ == "__main__":
    main()
