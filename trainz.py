"""Training entry point for the UAV-Crack segmentation experiments.

Improvements over the first-generation script (`train.py`):
  * full-resolution output (no stride-2 truncation of the logits);
  * photo-level leak-free split with a *stratified* holdout, so the local
    number correlates with the hidden leaderboard score;
  * foreground-aware sampling, crack-preserving augmentation;
  * composite losses (BCE / Dice / Tversky / Lovasz / soft-IoU);
  * EMA weights, cosine schedule with warmup, AMP, gradient clipping;
  * per-epoch threshold sweep + cached validation probabilities, so ensembles
    and thresholds can be re-tuned later without retraining.

Usage:
    python trainz.py --model unet_r34 --loss crack --epochs 60 --tag e1
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config as C

os.environ.setdefault("TORCH_HOME", str(C.PROJECT_ROOT / "weights"))
(C.PROJECT_ROOT / "weights" / "hub" / "checkpoints").mkdir(parents=True, exist_ok=True)

import metrics
from modelzoo import build_loss, build_model
from splits import load_split

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------- utils
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential moving average of model weights (kept in float32 on CPU)."""

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 50):
        self.decay = decay
        self.warmup = warmup
        self.updates = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / max(1, self.warmup)))
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                s.copy_(p)


# ------------------------------------------------------------------- dataset
class CrackDataset(Dataset):
    """Tiles with optional crack-preserving augmentation.

    The image cache stores *uint8* tiles (not float) so 1200 tiles cost ~0.9 GB
    instead of ~3.6 GB, and augmentation still sees raw pixels.
    """

    def __init__(self, samples: Sequence, size: Tuple[int, int] = C.IMG_SIZE,
                 augment: bool = False, cache: bool = True,
                 resized_crop: Tuple[float, float] = (0.55, 1.0),
                 color_jitter: float = 0.12, gamma: Tuple[float, float] = (0.75, 1.35),
                 noise: float = 0.02, rotate90: bool = True, blur: float = 0.1,
                 pad_target: bool = True, eval_mode: bool = False,
                 zoom: float = 1.0):
        self.samples = list(samples)
        self.size = size            # (H, W) network input
        self.augment = augment
        self.resized_crop = resized_crop
        self.zoom = float(zoom)     # >1 enables random zoom-in crops (scale aug)
        self.color_jitter = color_jitter
        self.gamma = gamma
        self.noise = noise
        self.rotate90 = rotate90
        self.blur = blur
        # When the mask must line up with a *padded* network input we pad it too
        # (training).  At validation the network restores the true input size, so
        # the mask is left untouched and metrics compare equal shapes.
        self.pad_target = pad_target
        # eval_mode makes __getitem__ return (image, filename) even for labelled
        # tiles - without it a labelled val tile returns (image, mask) and the
        # mask silently ends up where the filename is expected.
        self.eval_mode = eval_mode
        self._cache: Optional[List] = [None] * len(self.samples) if cache else None

    def __len__(self) -> int:
        return len(self.samples)

    def targets_at(self, idx: int) -> np.ndarray:
        """Ground-truth mask for evaluation.

        Validation runs with `pad_target=False`, so the network restores the true
        input size and the raw mask already lines up pixel-for-pixel.
        """
        _, mask = self._raw(idx)
        if mask is None:
            raise ValueError("no mask for this sample")
        if not self.pad_target and (self.size[0], self.size[1]) == mask.shape[:2]:
            return mask
        # Geometry must match what the network sees: resize to fit, then centre-pad.
        # (Do it on the mask itself - passing a dummy image to `_fit` would take the
        # scaling factor from that dummy and squash the mask.)
        from PIL import Image as _Image
        th, tw = self.size
        h, w = mask.shape[:2]
        scale = min(th / h, tw / w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        if (nh, nw) != (h, w):
            m_i = _Image.fromarray(mask * 255).resize((nw, nh), _Image.NEAREST)
            mask = (np.asarray(m_i, dtype=np.uint8) > 127).astype(np.uint8)
        if (nh, nw) != (th, tw):
            canvas = np.zeros((th, tw), dtype=np.uint8)
            dy, dx = (th - nh) // 2, (tw - nw) // 2
            canvas[dy:dy + nh, dx:dx + nw] = mask
            mask = canvas
        return mask

    # ---- raw tile access ------------------------------------------------
    def _raw(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self._cache is not None and self._cache[idx] is not None:
            return self._cache[idx]
        s = self.samples[idx]
        from PIL import Image
        with Image.open(s.image) as im:
            img = np.asarray(im.convert("RGB"), dtype=np.uint8)
        mask = None
        if getattr(s, "mask", None) is not None:
            with Image.open(s.mask) as im:
                mask = (np.asarray(im.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
        if self._cache is not None:
            self._cache[idx] = (img, mask)
        return img, mask

    # ---- geometry -------------------------------------------------------
    def _fit(self, img: np.ndarray, mask: Optional[np.ndarray], th: int, tw: int,
             allow_crop: bool, force_pad: bool = False):
        h, w = img.shape[:2]
        scale = min(th / h, tw / w)
        if allow_crop and self.zoom > 1.0:
            # random zoom-in: take a smaller window and resample it up to th x tw.
            # This is the *scale* augmentation the original recipe never applied
            # (378 < 384 made the resized-crop branch below unreachable).
            # Crops are crack-aware: a plain random window would land on empty
            # pavement ~60 % of the time at zoom 1.5 and teach the net to say "no crack".
            z = random.uniform(1.0, self.zoom)
            ch, cw = max(16, int(round(th / z))), max(16, int(round(tw / z)))
            if h >= ch and w >= cw:
                from PIL import Image
                if mask is not None:
                    need = 0.35 * float(mask.sum())
                    best = (-1.0, 0, 0)
                    for _ in range(8):
                        y0 = random.randint(0, h - ch)
                        x0 = random.randint(0, w - cw)
                        s = float(mask[y0:y0 + ch, x0:x0 + cw].sum())
                        if s > best[0]:
                            best = (s, y0, x0)
                        if s >= need:
                            break
                    _, y0, x0 = best
                else:
                    y0 = random.randint(0, h - ch)
                    x0 = random.randint(0, w - cw)
                img = img[y0:y0 + ch, x0:x0 + cw]
                img = np.asarray(Image.fromarray(img).resize((tw, th), Image.BILINEAR),
                                 dtype=np.uint8)
                if mask is not None:
                    mask = mask[y0:y0 + ch, x0:x0 + cw]
                    m_i = Image.fromarray(mask * 255).resize((tw, th), Image.NEAREST)
                    mask = (np.asarray(m_i, dtype=np.uint8) > 127).astype(np.uint8)
                return img, mask
        if allow_crop and h >= th and w >= tw:
            # random resized crop: zoom in, keep the raw aspect ratio, then take a
            # random th x tw window (cracks stay at their native width)
            z = random.uniform(*self.resized_crop)
            nh, nw = max(th, int(round(h * z))), max(tw, int(round(w * z)))
            nh, nw = min(nh, h), min(nw, w)
            from PIL import Image
            img_i = Image.fromarray(img).resize((nw, nh), Image.BILINEAR)
            img = np.asarray(img_i, dtype=np.uint8)
            if mask is not None:
                m_i = Image.fromarray(mask * 255).resize((nw, nh), Image.NEAREST)
                mask = (np.asarray(m_i, dtype=np.uint8) > 127).astype(np.uint8)
            y0 = random.randint(0, nh - th)
            x0 = random.randint(0, nw - tw)
            img = img[y0:y0 + th, x0:x0 + tw]
            if mask is not None:
                mask = mask[y0:y0 + th, x0:x0 + tw]
            return img, mask

        # resize to fit then centre-pad (used at validation time and for odd sizes)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        if (nh, nw) != (h, w):
            from PIL import Image
            img_i = Image.fromarray(img).resize((nw, nh), Image.BILINEAR)
            img = np.asarray(img_i, dtype=np.uint8)
            if mask is not None:
                m_i = Image.fromarray(mask * 255).resize((nw, nh), Image.NEAREST)
                mask = (np.asarray(m_i, dtype=np.uint8) > 127).astype(np.uint8)
        if (nh, nw) != (th, tw):
            canvas = np.zeros((th, tw, 3), dtype=np.uint8)
            do_pad = self.pad_target or force_pad
            if mask is not None and do_pad:
                cmask = np.zeros((th, tw), dtype=np.uint8)
            else:
                cmask = mask
            dy, dx = (th - nh) // 2, (tw - nw) // 2
            canvas[dy:dy + nh, dx:dx + nw] = img
            if mask is not None and do_pad:
                cmask[dy:dy + nh, dx:dx + nw] = mask
            img, mask = canvas, cmask
        return img, mask

    # ---- photometry -----------------------------------------------------
    def _photo(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image
        if self.color_jitter > 0:
            # brightness / contrast / saturation / hue, ImageEnhance style
            f = img.astype(np.float32)
            b = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            f = f * b
            c = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            m = f.mean()
            f = (f - m) * c + m
            if random.random() < 0.5:
                s = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
                g = f.mean(axis=2, keepdims=True)
                f = (f - g) * s + g
            img = np.clip(f, 0, 255).astype(np.uint8)
        if self.gamma != (1.0, 1.0) and random.random() < 0.5:
            lo, hi = self.gamma
            gm = random.uniform(lo, hi)
            lut = (np.clip((np.arange(256) / 255.0) ** gm, 0, 1) * 255).astype(np.uint8)
            img = lut[img]
        if self.noise > 0 and random.random() < 0.3:
            n = np.random.randn(*img.shape).astype(np.float32) * (self.noise * 255)
            img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        if self.blur > 0 and random.random() < self.blur:
            from PIL import ImageFilter
            img = np.asarray(Image.fromarray(img).filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.1))), dtype=np.uint8)
        return img

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img, mask = self._raw(idx)
        th, tw = self.size
        img, mask = self._fit(img, mask, th, tw, allow_crop=self.augment)
        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1], (mask[:, ::-1] if mask is not None else None)
            if random.random() < 0.5:
                img, mask = img[::-1], (mask[::-1] if mask is not None else None)
            if self.rotate90 and random.random() < 0.3:
                k = random.choice([1, 2, 3])
                img = np.rot90(img, k)
                if mask is not None:
                    mask = np.rot90(mask, k)
                img, mask = self._fit(np.ascontiguousarray(img),
                                      None if mask is None else np.ascontiguousarray(mask),
                                      th, tw, allow_crop=True)
            img = self._photo(img)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        if self.eval_mode or mask is None:
            return x, s.image.name
        return x, torch.from_numpy(mask.astype(np.float32))[None]


# ----------------------------------------------------------------- inference
@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device,
                  tta: int = 0, amp: bool = True) -> Tuple[List[np.ndarray], List[str]]:
    """Return per-tile probability maps and file names.

    `tta`: 0 = none, 4 = flips, 8 = flips + 90-degree rotations.
    """
    model.eval()
    probs: List[np.ndarray] = []
    names: List[str] = []
    use_amp = amp and device.type == "cuda"

    def run(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            return model(x)

    for batch in loader:
        x, name = batch
        x = x.to(device, non_blocking=True)
        logits = run(x).float()
        if tta >= 4:
            logits = logits + run(torch.flip(x, dims=[3])).float().flip(dims=[3])
            logits = logits + run(torch.flip(x, dims=[2])).float().flip(dims=[2])
            logits = logits + run(torch.flip(x, dims=[2, 3])).float().flip(dims=[2, 3])
            logits = logits / 4.0
        if tta >= 8:
            acc = logits * 4.0
            n = 4
            for k in (1, 2, 3):
                xr = torch.rot90(x, k, dims=[2, 3])
                lr = run(xr).float()
                lr = torch.rot90(lr, -k, dims=[2, 3])
                acc = acc + lr
                n += 1
            logits = acc / n
        p = torch.sigmoid(logits).squeeze(1)
        probs.extend([t for t in p.cpu().numpy()])
        names.extend(list(name))
    return probs, names


def load_targets(samples: Sequence) -> List[np.ndarray]:
    from PIL import Image
    out = []
    for s in samples:
        with Image.open(s.mask) as im:
            out.append((np.asarray(im.convert("L"), dtype=np.uint8) > 0).astype(np.uint8))
    return out


def vectorized_sweep(probs: Sequence[np.ndarray], targets: Sequence[np.ndarray],
                     ths: np.ndarray) -> Dict[str, float]:
    """Threshold sweep with numpy broadcasting.

    A per-threshold python loop over 196 tiles costs ~30 s per epoch; flattening
    once and comparing against every threshold at once makes it a few ms.
    """
    p = np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in probs])
    t = np.concatenate([np.asarray(y).ravel() for y in targets]).astype(bool)
    t_sum = int(t.sum())
    n = t.size
    out: Dict[str, float] = {}
    for key in ("iou", "miou", "f1"):
        best_v, best_th = -1.0, 0.5
        for th in ths:
            pred = p >= th
            tp = int(np.count_nonzero(pred & t))
            fp = int(pred.sum()) - tp
            fn = t_sum - tp
            tn = n - tp - fp - fn
            iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
            iou_bg = tn / (tn + fp + fn) if (tn + fp + fn) else float("nan")
            miou = np.nanmean([iou, iou_bg])
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / (tp + fn) if (tp + fn) else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
            v = {"iou": iou, "miou": miou, "f1": f1}[key]
            if np.isfinite(v) and v > best_v:
                best_v, best_th = float(v), float(th)
        out[f"{key}_best"] = best_v
        out[f"{key}_th"] = best_th
    return out


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp")
    ap.add_argument("--model", default="unet_r34")
    ap.add_argument("--loss", default="crack")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--img-size", type=int, nargs=2, default=list(C.TRAIN_SIZE))
    ap.add_argument("--val-size", type=int, nargs=2, default=list(C.VAL_SIZE))
    ap.add_argument("--init", default=None,
                    help="optional checkpoint to warm-start from (continue a run)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-train-batches", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--patience", type=int, default=0, help="0 = train the full schedule")
    ap.add_argument("--foreground-sample", action="store_true",
                    help="weight sampling towards crack-rich tiles")
    ap.add_argument("--tta", type=int, default=0, help="TTA views used for validation")
    ap.add_argument("--select", default="iou", choices=["iou", "miou", "f1", "score"])
    ap.add_argument("--save-val-probs", action="store_true")
    ap.add_argument("--min-epochs", type=int, default=0, help="epochs before early stop may trigger")
    ap.add_argument("--val-every", type=int, default=1, help="validate every N epochs")
    ap.add_argument("--steps-per-epoch", type=int, default=0,
                    help="cap optimiser steps per epoch (0 = use the whole split)")
    ap.add_argument("--aug-scale", type=float, default=1.0,
                    help="max random zoom-in factor for training (>1 enables scale aug)")
    ap.add_argument("--boundary-weight", type=float, default=0.0,
                    help="weight of the boundary-band BCE term (>0 enables it)")
    ap.add_argument("--tversky-alpha", type=float, default=0.3,
                    help="FP weight in the Tversky term (higher = precision-biased)")
    ap.add_argument("--tversky-beta", type=float, default=0.7,
                    help="FN weight in the Tversky term (higher = recall-biased)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out) if args.out else (C.OUT_DIR / args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[env] device={device} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else '-'}", flush=True)

    # ---- data
    tr, va, meta = load_split(args.val_fraction, args.split_seed)
    if args.limit_train:
        tr = tr[: args.limit_train]
    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump({"n_train": len(tr), "n_val": len(va),
                   "n_photos": meta["n_photos"],
                   "n_val_photos": len(meta["val_photos"]),
                   "val_photos": meta["val_photos"],
                   "val_files": [s.image.name for s in va],
                   "val_fg": meta.get("val_fg", []),
                   "train_fg": meta.get("train_fg", [])}, f, indent=1)

    tr_ds = CrackDataset(tr, size=tuple(args.img_size), augment=True, cache=not args.no_cache,
                         pad_target=True, zoom=args.aug_scale)
    va_ds = CrackDataset(va, size=tuple(args.val_size), augment=False, cache=not args.no_cache,
                         pad_target=False, eval_mode=True)
    pin = device.type == "cuda"

    sampler = None
    if args.foreground_sample:
        w = np.array([max(1.0, float(getattr(s, "fg", 0.01)) * 100.0) for s in tr], dtype=np.float64)
        w = np.clip(w, 1.0, 6.0)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(tr), replacement=True)
    train_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=args.workers, pin_memory=pin,
                              drop_last=True)
    val_loader = DataLoader(va_ds, batch_size=max(4, args.batch_size), shuffle=False,
                            num_workers=args.workers, pin_memory=pin)
    # deterministic prefix is enough when capping steps: every epoch still sees
    # a different shuffle of the same pool, so coverage stays uniform over time
    if args.steps_per_epoch:
        from torch.utils.data import Subset
        n_keep = min(len(tr_ds), args.steps_per_epoch * args.batch_size)
        train_loader = DataLoader(Subset(tr_ds, list(range(n_keep))),
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=pin, drop_last=True)

    targets = [va_ds.targets_at(i) for i in range(len(va_ds))]

    # ---- model / optimisation
    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict({k.replace("module.", ""): v for k, v in ck["model"].items()})
        print(f"[init] warm-started from {args.init} (ep {ck.get('epoch')}, "
              f"value {ck.get('value')})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = build_loss(args.loss, tversky_alpha=args.tversky_alpha,
                           tversky_beta=args.tversky_beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ema = None if args.no_ema else EMA(model, args.ema_decay)

    steps_per_epoch = args.max_train_batches or len(train_loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(steps_per_epoch * args.warmup_epochs))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * args.lr * (1 + math.cos(math.pi * min(1.0, prog)))

    print(f"[setup] model={args.model} params={n_params/1e6:.2f}M loss={args.loss} "
          f"train={len(tr)} val={len(va)} photos={meta['n_photos']} "
          f"steps/ep={steps_per_epoch} epochs={args.epochs} img={tuple(args.img_size)}", flush=True)

    history: List[Dict[str, float]] = []
    best = {"value": -1.0, "epoch": 0, "which": "raw"}
    bad = 0
    step = 0
    t_start = time.time()
    ths = np.round(np.arange(0.20, 0.81, 0.01), 2)

    def evaluate(net: nn.Module) -> Dict[str, float]:
        probs, _ = predict_probs(net, val_loader, device, tta=args.tta, amp=C.AMP)
        row = vectorized_sweep(probs, targets, ths)
        c = metrics.Confusion()
        for p, y in zip(probs, targets):
            c.update(p >= 0.5, y)
        row.update({"miou@50": c.miou, "iou@50": c.iou_crack, "f1@50": c.f1,
                    "P@50": c.precision, "R@50": c.recall})
        row["score"] = 0.5 * (row["miou_best"] + row["f1_best"])
        return row, probs

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n_seen = 0.0, 0
        t0 = time.time()
        for i, (x, y) in enumerate(train_loader):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(C.AMP and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
                if args.boundary_weight > 0:
                    # 5x5 dilate-erode band around the crack: put explicit weight on
                    # the 1-2 px boundary that dominates both FP and FN (see docs/)
                    with torch.no_grad():
                        dil = F.max_pool2d(y, 5, 1, 2)
                        ero = -F.max_pool2d(-y, 5, 1, 2)
                        band = (dil - ero).clamp_(0.0, 1.0)
                    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                    loss = loss + args.boundary_weight * (bce * band).sum() / (band.sum() + 1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            step += 1
            run_loss += float(loss.detach()) * x.size(0)
            n_seen += x.size(0)
        train_loss = run_loss / max(n_seen, 1)
        train_sec = time.time() - t0

        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
        if not do_val:
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "lr": optimizer.param_groups[0]["lr"], "train_sec": train_sec})
            print(f"[ep {epoch:3d}/{args.epochs}] loss={train_loss:.4f} | {train_sec:.0f}s "
                  f"(val skipped)", flush=True)
            with open(out_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=1)
            continue

        row_raw, probs_raw = evaluate(model)
        row = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"],
               "train_sec": train_sec}
        row.update({f"raw_{k}": v for k, v in row_raw.items()})

        probs_ema = None
        if ema is not None:
            row_ema, probs_ema = evaluate(ema.shadow)
            row.update({f"ema_{k}": v for k, v in row_ema.items()})

        row["val_sec"] = time.time() - t0 - train_sec
        history.append(row)
        key = f"{args.select}_best"
        cand = [("raw", row_raw[key], probs_raw)]
        if probs_ema is not None:
            cand.append(("ema", row_ema[key], probs_ema))
        which, value, probs = max(cand, key=lambda t: t[1])
        print(f"[ep {epoch:3d}/{args.epochs}] loss={train_loss:.4f} "
              f"raw IoU={row_raw['iou_best']:.4f}@{row_raw['iou_th']:.2f} "
              f"mIoU={row_raw['miou_best']:.4f} F1={row_raw['f1_best']:.4f}"
              + (f" | ema IoU={row_ema['iou_best']:.4f} mIoU={row_ema['miou_best']:.4f}"
                 f" F1={row_ema['f1_best']:.4f}" if probs_ema is not None else "")
              + f" | {train_sec:.0f}s+{row['val_sec']:.0f}s", flush=True)

        improved = value > best["value"]
        if improved:
            best = {"value": float(value), "epoch": epoch, "which": which}
            net = ema.shadow if which == "ema" else model
            torch.save({"model": net.state_dict(), "model_name": args.model,
                        "epoch": epoch, "which": which, "value": float(value),
                        "metrics": row, "args": vars(args)}, out_dir / "best.pt")
            if args.save_val_probs:
                np.savez_compressed(out_dir / "val_probs.npz",
                                    probs=np.stack(probs).astype(np.float16),
                                    names=np.array([s.image.name for s in va]))
            bad = 0
        else:
            bad += 1
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=1)
        if args.patience and bad >= args.patience and epoch >= args.min_epochs:
            print(f"[early-stop] no improvement for {bad} epochs", flush=True)
            break

    minutes = (time.time() - t_start) / 60
    summary = {"tag": args.tag, "model": args.model, "loss": args.loss,
               "epochs_run": len(history), "best": best, "n_params": n_params,
               "minutes": minutes, "n_train": len(tr), "n_val": len(va),
               "args": vars(args),
               "best_row": history[best["epoch"] - 1] if history else {}}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"[done] {args.tag}: best {args.select}={best['value']:.4f} "
          f"({best['which']}, ep {best['epoch']}) in {minutes:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
