"""伪标签自训练（transductive fine-tune）：在 1004 张有标注瓦片 + 300 张测试图伪标签上微调。

为什么：任务的官方难点就是 domain shift，而测试图本身是可得的（无标签）。
用当前最优集成的概率图（阈值化）当伪标签，低学习率微调 6–10 轮，通常能把模型
往测试域的成像风格/尺度分布上挪一点，属于"不花新数据"的提分手段。

评测口径：v1（196 张，照片隔离，且**不在**微调集里）仍然可信；v2 与微调集照片重叠，
只能当"乐观参考"。

    python tools/self_train.py --ckpt runs/N2_swin_s_60/best.pt --probs sub/test_probs_a4.npz \
        --tag ST_swin_s --epochs 8
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
import metrics  # noqa: E402
from modelzoo import build_loss, build_model  # noqa: E402
from splits import load_split, unlabeled_index, Sample  # noqa: E402
from trainz import CrackDataset, EMA, predict_probs, set_seed  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probs", required=True, help="npz of test-tile probabilities")
    ap.add_argument("--tag", default="ST")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--threshold", type=float, default=0.43)
    ap.add_argument("--pseudo-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--zoom", type=float, default=1.2,
                    help="training-time zoom aug; set 1.0 when fine-tuning a model "
                         "that was NOT trained with scale augmentation")
    ap.add_argument("--min-size", type=int, default=0)
    ap.add_argument("--img-size", type=int, nargs=2, default=list(C.TRAIN_SIZE),
                    help="train H W; use e.g. 480 840 to fine-tune at 1.25x resolution")
    ap.add_argument("--val-size", type=int, nargs=2, default=None,
                    help="validation H W (defaults to --img-size when given)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    out_dir = ROOT / "runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va, _ = load_split(0.2, 42, kind="v1")     # va == v1 holdout, kept clean
    pseudo_dir = ROOT / "sub" / "pseudo" / args.tag
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.probs)
    extra = []
    for s in unlabeled_index():
        stem = Path(s.image.name).stem
        if stem not in d.files:
            continue
        mask = (d[stem].astype(np.float32) >= args.threshold).astype(np.uint8)
        p = pseudo_dir / f"{stem}.png"
        Image.fromarray(mask * 1).save(p)
        extra.append(Sample(s.image, p, s.photo, "pseudo"))
    print(f"[pseudo] {len(extra)} tiles -> {pseudo_dir}", flush=True)

    tr_ds = CrackDataset(tr + extra, size=tuple(args.img_size), augment=True, cache=True,
                         pad_target=True, zoom=args.zoom)
    val_size = tuple(args.val_size) if args.val_size else (
        tuple(args.img_size) if tuple(args.img_size) != tuple(C.TRAIN_SIZE) else tuple(C.VAL_SIZE))
    va_ds = CrackDataset(va, size=val_size, augment=False, cache=True,
                         pad_target=False, eval_mode=True)
    n_pseudo = len(extra)
    train_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(va_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    targets = [va_ds.targets_at(i) for i in range(len(va_ds))]

    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck.get("model_name"), pretrained=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ck["model"].items()})
    model.to(device)
    ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    crit = build_loss("crack").to(device)
    steps = len(train_loader) * args.epochs

    ths = np.round(np.arange(0.20, 0.81, 0.01), 2)
    best = {"iou": -1.0, "epoch": 0, "thr": 0.5}
    hist = []
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            for g in opt.param_groups:
                g["lr"] = 0.5 * args.lr * (1 + math.cos(math.pi * min(1.0, step / steps)))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            step += 1
            run += float(loss.detach()) * x.size(0)
            seen += x.size(0)
        if epoch % args.val_every:
            continue
        probs, _ = predict_probs(ema.shadow, val_loader, device, tta=0)
        c = metrics.Confusion()
        best_th, best_iou = 0.5, -1.0
        for th in ths:
            cc = metrics.Confusion()
            for p, y_ in zip(probs, targets):
                cc.update(p >= th, y_)
            if cc.iou_crack > best_iou:
                best_iou, best_th = float(cc.iou_crack), float(th)
        row = {"epoch": epoch, "loss": run / max(seen, 1), "v1_iou": best_iou,
               "thr": best_th, "sec": round(time.time() - t0, 1)}
        hist.append(row)
        print(f"[{args.tag} ep {epoch}/{args.epochs}] loss={row['loss']:.4f} "
              f"v1 IoU={best_iou:.4f}@{best_th:.2f} ({row['sec']:.0f}s)", flush=True)
        if best_iou > best["iou"]:
            best = {"iou": best_iou, "epoch": epoch, "thr": best_th}
            torch.save({"model": ema.shadow.state_dict(), "model_name": ck.get("model_name"),
                        "epoch": epoch, "which": "ema", "value": best_iou,
                        "metrics": row, "args": vars(args)}, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")

    print(f"[done] {args.tag} best v1 IoU={best['iou']:.4f} @ep{best['epoch']} "
          f"thr={best['thr']:.2f} in {time.time() - t0:.0f}s", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(
        {"tag": args.tag, "best": best, "n_pseudo": n_pseudo, "args": vars(args)},
        indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
