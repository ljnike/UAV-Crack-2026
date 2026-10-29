"""Inference on the 300 unlabeled `leftImg8bit/val` tiles.

Produces 0/1 uint8 PNG masks with the *same basename* as the input tiles, which
is the naming convention used by the provided `result/` reference output and by
the VLP evaluator.

Single model:
    python predictz.py --ckpt runs/e1/best.pt --out sub/masks

Ensemble / multi-scale TTA (the high-score path):
    python predictz.py --ckpt runs/e1/best.pt runs/e2/best.pt \
        --scales 0.75 1.0 1.25 --flip --rot90 --threshold 0.45 --out sub/masks
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import config as C

os.environ.setdefault("TORCH_HOME", str(C.PROJECT_ROOT / "weights"))
import torch.nn.functional as F

from modelzoo import build_model
from splits import unlabeled_index

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


def load_model(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    name = ck.get("model_name") or ck.get("args", {}).get("model", "unet_r34")
    model = build_model(name, pretrained=False)
    state = {k.replace("module.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ck, name


@torch.no_grad()
def probs_for(model: torch.nn.Module, img: np.ndarray, device: torch.device,
              scale: float = 1.0, flip: bool = False, rot90: bool = False,
              amp: bool = True) -> np.ndarray:
    """Probability map for one tile at one (scale, tta) view."""
    from PIL import Image
    h, w = img.shape[:2]
    if scale != 1.0:
        nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
        img_s = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    else:
        img_s = img
    x = to_tensor(img_s).to(device)
    views: List[Tuple[torch.Tensor, callable]] = [(x, lambda t: t)]
    if flip:
        views.append((torch.flip(x, dims=[3]), lambda t: torch.flip(t, dims=[3])))
        views.append((torch.flip(x, dims=[2]), lambda t: torch.flip(t, dims=[2])))
        views.append((torch.flip(x, dims=[2, 3]), lambda t: torch.flip(t, dims=[2, 3])))
    if rot90:
        for k in (1, 2, 3):
            views.append((torch.rot90(x, k, dims=[2, 3]),
                          (lambda kk: (lambda t: torch.rot90(t, -kk, dims=[2, 3])))(k)))
    acc = None
    for xv, inv in views:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(amp and device.type == "cuda")):
            logit = model(xv)
        p = torch.sigmoid(inv(logit).float())
        acc = p if acc is None else acc + p
    p = (acc / len(views))
    if p.shape[-2:] != (h, w):
        p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
    return p[0, 0].cpu().numpy()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints")
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="optional ensemble weights (defaults to equal)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--scales", nargs="*", type=float, default=[1.0])
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--rot90", action="store_true")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--save-probs", default=None, help="optional .npz of raw probabilities")
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    from PIL import Image

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for c in args.ckpt:
        m, ck, name = load_model(Path(c), device)
        models.append(m)
        print(f"[ckpt] {c} model={name} epoch={ck.get('epoch')} which={ck.get('which')} "
              f"value={ck.get('value'):.4f}", flush=True)
    w = np.array(args.weights if args.weights else [1.0] * len(models), dtype=np.float64)
    w = w / w.sum()

    samples = unlabeled_index()
    if args.limit:
        samples = samples[: args.limit]
    print(f"[infer] {len(samples)} tiles, models={len(models)} scales={args.scales} "
          f"flip={args.flip} rot90={args.rot90} thr={args.threshold}", flush=True)

    prob_store: Dict[str, np.ndarray] = {}
    fg_stats = []
    for i, s in enumerate(samples):
        with Image.open(s.image) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.uint8)
        prob = np.zeros(img.shape[:2], dtype=np.float64)
        for m, wi in zip(models, w):
            for sc in args.scales:
                prob += wi * probs_for(m, img, device, scale=sc, flip=args.flip,
                                       rot90=args.rot90, amp=args.amp)
        prob /= max(1, len(args.scales))
        mask = (prob >= args.threshold).astype(np.uint8)
        Image.fromarray(mask * 255).save(out_dir / f"{Path(s.image.name).stem}.png")
        if args.save_probs:
            prob_store[Path(s.image.name).stem] = prob.astype(np.float16)
        fg_stats.append(float(mask.mean()))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} done", flush=True)

    if args.save_probs:
        np.savez_compressed(args.save_probs, **prob_store)
    stats = {"n": len(samples), "threshold": args.threshold,
             "mean_fg_fraction": float(np.mean(fg_stats)),
             "median_fg_fraction": float(np.median(fg_stats)),
             "empty_masks": int(sum(1 for f in fg_stats if f == 0.0)),
             "full_masks": int(sum(1 for f in fg_stats if f > 0.5))}
    with open(out_dir / "predict_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(f"[done] {stats}", flush=True)


if __name__ == "__main__":
    main()
