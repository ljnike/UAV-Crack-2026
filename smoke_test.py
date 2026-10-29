"""GPU smoke test: forward shapes, param counts, speed and memory for every model.

Run:  python smoke_test.py [--size 378 672] [--batch 2]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import config as C

os.environ.setdefault("TORCH_HOME", str(C.PROJECT_ROOT / "weights"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[378, 672])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--losses", nargs="*", default=["bce_dice", "crack", "tversky", "focal", "lovasz"])
    args = ap.parse_args()

    from modelzoo import MODEL_TABLE, build_loss, build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__} cuda={torch.version.cuda}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"gpu={p.name} sm_{p.major}{p.minor} vram={p.total_memory/2**30:.2f}GB")

    h, w = args.size
    x = torch.randn(args.batch, 3, h, w, device=device)
    y = (torch.rand(args.batch, 1, h, w, device=device) > 0.97).float()

    rows = []
    for name in (args.models or list(MODEL_TABLE)):
        try:
            model = build_model(name, pretrained=True).to(device).eval()
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL-build] {name}: {exc}")
            continue
        n_params = sum(p.numel() for p in model.parameters())
        try:
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model(x)
            ok = tuple(out.shape[-2:]) == (h, w)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                for _ in range(3):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        model(x)
                torch.cuda.synchronize()
                ms = (time.time() - t0) / 3 * 1000
                peak = torch.cuda.max_memory_allocated() / 2**30
            else:
                ms, peak = float("nan"), float("nan")
            rows.append({"model": name, "params_M": round(n_params / 1e6, 2),
                         "out": list(out.shape), "exact_size": ok,
                         "ms_per_batch": round(ms, 1), "peak_GB": round(peak, 2)})
            print(f"[ok] {name:20s} params={n_params/1e6:6.2f}M out={tuple(out.shape)} "
                  f"exact={ok} {ms:7.1f} ms/batch peak={peak:.2f}GB")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL-fwd] {name}: {type(exc).__name__}: {exc}")
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n=== losses ===")
    logits = torch.randn(2, 1, 64, 96, device=device, requires_grad=True)
    target = (torch.rand(2, 1, 64, 96, device=device) > 0.98).float()
    for lname in args.losses:
        try:
            fn = build_loss(lname).to(device)
            loss = fn(logits, target)
            loss.backward()
            print(f"[ok] loss {lname:10s} = {float(loss):.4f} (grad ok)")
            logits.grad = None
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] loss {lname}: {type(exc).__name__}: {exc}")

    out_path = C.OUT_DIR / "smoke_gpu.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
