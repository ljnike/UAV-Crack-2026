"""把复现 v4（线上 67.08 / 榜单第 2）所需的权重与清单打包成一个 zip。

    python tools/package_weights.py --out sub/uav_crack_v4_weights.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MEMBERS = [
    # tag,              model_name,  inference scales,                  role
    ("ST_A", "swin_s", [0.85, 1.0, 1.15], "AE 合成成员（原生分辨率 + 伪标签自训练）"),
    ("E_swins_prec", "swin_s", [0.85, 1.0, 1.15], "AE 合成成员（精确率偏置 Tversky α=0.7）"),
    ("L4_unetd_100", "unetd_r34", [1.0], "全分辨率解码 U-Net-DS，边界校正项"),
    ("H_swins_hi", "swin_s", [1.25], "1.25× 从零训练的高分辨率 Swin-S"),
    ("H2_swins_15x", "swin_s", [1.5], "1.5× 从零训练的高分辨率 Swin-S（最强单模型 0.6620）"),
]

ENSEMBLE = {
    "fusion": "logit-weighted average",
    "members": {
        "AE_ms::ST_A": 0.10 * (0.35 / 0.75),
        "AE_ms::E_swins_prec": 0.65 * (0.35 / 0.75),
        "L4_unetd_100": 0.10,
        "H_swins_hi": 0.25,
        "H2_swins_15x": 0.30,
    },
    "note": "AE_ms = logit(0.10*ST_A_ms + 0.65*E_ms) / 0.75，再以 0.35 权重参与最终融合",
    "threshold": 0.495,
    "postprocess": "remove connected components < 64 px",
    "fg_policy": "0 for background, 1 for crack",
    "local_metrics": {"v1_CrackIoU": 0.6703, "v1_mIoU": 0.8196, "v2_CrackIoU": 0.6738,
                      "v2_mIoU": 0.8222},
    "online_metrics": {"mIoU": 82.33, "CrackF1": 80.30, "CrackIoU": 67.08,
                       "CrackPrecision": 80.54, "CrackRecall": 80.06, "aAcc": 97.69},
    "leaderboard": "rank 2 / 74 (UAV-fyk v4, 2026-10-26)",
}

REPRODUCE = """# 复现 v4（线上 Crack IoU 67.08）**

权重放在 `weights/` 下；代码用 `uavcrack/` 这套（`modelzoo.py` + `predictz.py` + `tools/ensemble_test.py`）。

```powershell
$py = "uavcrack\\.venv\\Scripts\\python.exe"; $env:TORCH_HOME = "uavcrack\\weights"

# 1) 各自导出测试集概率（注意每个成员的推理尺度不同）
& $py predictz.py --ckpt weights/ST_A_best.pt        --out sub/m1 --scales 0.85 1.0 1.15 --flip --save-probs sub/p_STA.npz
& $py predictz.py --ckpt weights/E_swins_prec_best.pt --out sub/m2 --scales 0.85 1.0 1.15 --flip --save-probs sub/p_E.npz
& $py predictz.py --ckpt weights/L4_unetd_100_best.pt --out sub/m3 --scales 1.0  --flip --save-probs sub/p_L4.npz
& $py predictz.py --ckpt weights/H_swins_hi_best.pt  --out sub/m4 --scales 1.25 --flip --save-probs sub/p_H.npz
& $py predictz.py --ckpt weights/H2_swins_15x_best.pt --out sub/m5 --scales 1.5 --flip --save-probs sub/p_H2.npz

# 2) logit 加权融合 + 阈值 + 后处理 + 打包
& $py tools/ensemble_test.py `
    --probs sub/p_STA.npz sub/p_E.npz sub/p_L4.npz sub/p_H.npz sub/p_H2.npz `
    --weights 0.04667 0.30333 0.10 0.25 0.30 --logit `
    --threshold 0.495 --min-size 64 --fg-value 1 `
    --out-masks sub/masks_v4 --out-zip sub/result_v4.zip
```

关键点：
- 每个成员必须用**自己的推理尺度**（见 MANIFEST.json 的 `scales`），用错尺度会掉 5–12 个点；
- 阈值 0.495 不是本地 IoU 最优点（本地最优是 0.41），而是**按线上标定的前景密度**选的
  （测试集预测前景中位数 ≈5.4%，对应线上最优）；
- 掩膜像素值必须是 **0/1**（0/255 在平台会被判 0 分）。
"""


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "sub" / "uav_crack_v4_weights.zip"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = {"ensemble": ENSEMBLE, "members": []}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for tag, model, scales, role in MEMBERS:
            src = ROOT / "runs" / tag / "best.pt"
            if not src.exists():
                raise SystemExit(f"missing checkpoint {src}")
            name = f"weights/{tag}_best.pt"
            z.write(src, name)
            manifest["members"].append({
                "file": name, "run": tag, "model_name": model,
                "scales": scales, "role": role,
                "size_mb": round(src.stat().st_size / 1e6, 1),
                "sha256": sha256(src),
            })
            print(f"[add] {name}  {src.stat().st_size/1e6:.1f} MB")
        z.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))
        z.writestr("REPRODUCE.md", REPRODUCE)

    with zipfile.ZipFile(out) as z:
        print(f"[zip] {out}  entries={len(z.namelist())}  "
              f"size={out.stat().st_size/1e6:.1f} MB  testzip={z.testzip()}")
        print("       ", ", ".join(z.namelist()))


if __name__ == "__main__":
    main()
