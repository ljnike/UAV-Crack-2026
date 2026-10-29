"""2× 推理探针：验证"输出栅格太粗"是不是边界误差的主因。

思路：模型本身不动，只把输入放大到 2×（1344×756）再前向，概率图按面积插值回 378×672。
若 stride-4 的输出栅格是瓶颈，2× 推理应该**同时**抬高 P 与 R；
若只是把裂缝整体画宽，则 P 降 R 升、IoU 不动。

两步（GPU 空闲时跑，先跑小规模确认显存）：
    python tools/probe_scale.py run  --ckpt runs/N1_swin_t_60/best.pt --scale 2.0 --limit 16
    python tools/probe_scale.py run  --ckpt runs/N1_swin_t_60/best.pt --scale 2.0 --out runs/N1_swin_t_60/val_probs_v1_s2.npz
    python tools/probe_scale.py report --npz runs/N1_swin_t_60/val_probs_v1_s2.npz

注意：2× 推理显存约为 1× 的 4 倍。8 GB 卡上 Swin-S/U-Net-DS 若 OOM，改用 Swin-T 或降到 1.5×。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import config as C  # noqa: E402

from splits import load_split  # noqa: E402
from predictz import load_model, probs_for  # noqa: E402
from select_model import gt_lookup, hist_of, load_gt, metrics_from_hist  # noqa: E402


def run(args) -> None:
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val, _ = load_split(0.2, 42, kind=args.kind)
    if args.limit:
        val = val[: args.limit]
    models = [load_model(Path(c), device)[0] for c in args.ckpt]
    print(f"[probe] tiles={len(val)} scale={args.scale} flip={args.flip} device={device}", flush=True)

    probs, names = [], []
    for i, s in enumerate(val):
        with Image.open(s.image) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.uint8)
        acc = np.zeros(img.shape[:2], dtype=np.float64)
        for m in models:
            acc += probs_for(m, img, device, scale=args.scale, flip=args.flip)
        probs.append((acc / len(models)).astype(np.float16))
        names.append(s.image.name)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(val)}", flush=True)
    dest = Path(args.out) if args.out else ROOT / "runs" / "probe_scale" / f"probs_{args.kind}_s{args.scale}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, probs=np.stack(probs), names=np.array(names))
    print(f"[saved] {dest}")


def report(args) -> None:
    d = np.load(args.npz)
    names = [str(x) for x in d["names"]]
    probs = d["probs"].astype(np.float32)
    y = load_gt(names, gt_lookup())
    m = metrics_from_hist(hist_of(np.clip(np.rint(probs * 255), 0, 255).astype(np.uint8), y))
    print(f"[{Path(args.npz).name}] tiles={len(names)}")
    for th in (0.35, 0.40, 0.45, 0.50, 0.55):
        i = int(round(th * 255))
        print(f"  thr={th:.2f} IoU={m['iou'][i]:.4f} mIoU={m['miou'][i]:.4f} "
              f"F1={m['f1'][i]:.4f} P={m['precision'][i]:.4f} R={m['recall'][i]:.4f}")
    b = int(np.nanargmax(m["iou"]))
    print(f"  best IoU={m['iou'][b]:.4f} @thr={b / 255:.2f} "
          f"(P={m['precision'][b]:.4f} R={m['recall'][b]:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", nargs="+", required=True)
    r.add_argument("--scale", type=float, default=2.0)
    r.add_argument("--flip", action="store_true", default=True)
    r.add_argument("--kind", default="v1", choices=["v1", "v2"])
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", default=None)
    q = sub.add_parser("report")
    q.add_argument("--npz", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
