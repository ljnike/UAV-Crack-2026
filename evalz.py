"""Evaluate checkpoints on the local holdout, tune thresholds, and build ensembles.

Everything works off cached probability maps, so threshold tuning, ensemble
weight search and test-time-augmentation decisions cost seconds instead of
another training run.

    # 1) cache probabilities for a run (one .npz per run directory)
    python evalz.py cache --ckpt runs/e1/best.pt
    # 2) report single-model metrics with a threshold sweep
    python evalz.py report --runs runs/e1
    # 3) greedy ensemble + threshold tuning over several runs
    python evalz.py ensemble --runs runs/e1 runs/e2 runs/e3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import config as C

os.environ.setdefault("TORCH_HOME", str(C.PROJECT_ROOT / "weights"))

import metrics
from splits import load_split

THS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def cache_run(ckpt: Path, tta: int, val_fraction: float, split_seed: int,
              out: Optional[Path], kind: str = "v1") -> Path:
    """Run inference on the holdout and store probabilities next to the ckpt."""
    from torch.utils.data import DataLoader

    from trainz import CrackDataset, predict_probs
    from modelzoo import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, va, _ = load_split(val_fraction, split_seed, kind=kind)
    ds = CrackDataset(va, size=tuple(C.VAL_SIZE), augment=False, cache=True,
                      pad_target=False, eval_mode=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    name = ck.get("model_name") or ck.get("args", {}).get("model", "unet_r34")
    model = build_model(name, pretrained=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ck["model"].items()})
    model.to(device).eval()
    probs, names = predict_probs(model, loader, device, tta=tta)
    dest = out or (ckpt.parent / f"val_probs_{kind}_tta{tta}.npz")
    np.savez_compressed(dest, probs=np.stack(probs).astype(np.float16),
                        names=np.array(names))
    print(f"[cache] {ckpt} kind={kind} tta={tta} -> {dest}")
    return dest


def metrics_at(probs: np.ndarray, targets: Sequence[np.ndarray], th: float) -> Dict[str, float]:
    c = metrics.Confusion()
    for p, y in zip(probs, targets):
        c.update(p >= th, y)
    return {"iou": c.iou_crack, "miou": c.miou, "f1": c.f1,
            "precision": c.precision, "recall": c.recall, "dice": c.dice,
            "accuracy": c.accuracy}


def sweep(probs: np.ndarray, targets: Sequence[np.ndarray], key: str = "iou"
          ) -> Tuple[float, float, List[Dict[str, float]]]:
    rows = []
    for th in THS:
        m = metrics_at(probs, targets, float(th))
        m["threshold"] = float(th)
        rows.append(m)
    best = max(rows, key=lambda r: (r[key] if np.isfinite(r[key]) else -1))
    return float(best["threshold"]), float(best[key]), rows


def load_targets(val, size=None) -> List[np.ndarray]:
    """Ground-truth masks resampled exactly like the network input.

    Uses the dataset's own geometry so probabilities and targets can never
    disagree on shape (the earlier raw-PNG version silently produced a
    `shape 384 vs 378` mismatch whenever the network input was padded).
    """
    from trainz import CrackDataset
    ds = CrackDataset(val, size=tuple(size or C.VAL_SIZE), augment=False,
                      cache=True, pad_target=False, eval_mode=True)
    return [ds.targets_at(i) for i in range(len(ds))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "report", "ensemble"])
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--key", default="iou", choices=["iou", "miou", "f1"])
    ap.add_argument("--kind", default="v1", choices=["v1", "v2"],
                    help="v1 = photo-disjoint holdout, v2 = photo-overlapping")
    args = ap.parse_args()

    if args.cmd == "cache":
        for c in args.ckpt:
            cache_run(Path(c), args.tta, args.val_fraction, args.split_seed, None, args.kind)
        return

    _, val, _ = load_split(args.val_fraction, args.split_seed, kind=args.kind)
    targets = load_targets(val, C.VAL_SIZE)
    names = [str(s.image.name) for s in val]
    with open(Path(args.runs[0]) / f"evalset_{args.kind}.json", "w", encoding="utf-8") as f:
        json.dump({"files": names, "n": len(names), "kind": args.kind}, f)

    runs = {}
    for r in args.runs:
        cands = sorted(Path(r).glob(f"val_probs_{args.kind}_tta*.npz"))
        if not cands:
            raise SystemExit(f"no cached probabilities for kind={args.kind} in {r}; run `cache` first")
        d = np.load(cands[-1])
        got = [str(x) for x in list(d["names"])]
        if got != names:
            raise SystemExit(f"validation set mismatch in {cands[-1]} "
                             f"({len(set(got) ^ set(names))} differing tiles)")
        runs[Path(r).name] = d["probs"].astype(np.float32)
    print(f"[eval] kind={args.kind} val tiles={len(names)} runs={list(runs)}")

    if args.cmd == "report":
        report = {}
        for name, pr in runs.items():
            th, v, rows = sweep(pr, targets, args.key)
            m50 = metrics_at(pr, targets, 0.5)
            report[name] = {"best_threshold": th, f"best_{args.key}": v, "at_0.5": m50,
                            "sweep": rows}
            print(f"  {name:24s} best {args.key}={v:.4f} @thr={th:.2f} | "
                  f"@0.5 IoU={m50['iou']:.4f} mIoU={m50['miou']:.4f} F1={m50['f1']:.4f}")
        dest = Path(args.out) if args.out else Path(args.runs[0]) / "report.json"
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"[report] -> {dest}")
        return

    # ---- greedy forward ensemble selection on the chosen metric
    keys = list(runs)
    chosen: List[str] = []
    best_val = -1.0
    while True:
        cand_best = None
        for k in keys:
            if k in chosen:
                continue
            trial = chosen + [k]
            pr = np.mean([runs[t] for t in trial], axis=0)
            _, v, _ = sweep(pr, targets, args.key)
            if cand_best is None or v > cand_best[1]:
                cand_best = (k, v)
        if cand_best is None or cand_best[1] <= best_val + 1e-5:
            break
        chosen.append(cand_best[0])
        best_val = cand_best[1]
        print(f"  + {cand_best[0]:24s} -> {args.key}={best_val:.4f}  ({chosen})")

    pr = np.mean([runs[t] for t in chosen], axis=0)
    th, v, rows = sweep(pr, targets, args.key)
    result = {"members": chosen, "threshold": th, f"{args.key}": v,
              "metrics": metrics_at(pr, targets, th),
              "metrics_at_0.5": metrics_at(pr, targets, 0.5)}
    print(f"[ensemble] members={chosen} thr={th:.2f} -> {json.dumps(result['metrics'], indent=1)}")
    with open(Path(args.runs[0]) / "ensemble.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
