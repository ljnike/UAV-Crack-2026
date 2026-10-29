# -*- coding: utf-8 -*-
"""Export an ensemble member to ONNX (the challenge asks for .pth *and* .onnx).

All five members are exported at the exact input size used by `predictz.py`:

    python export_onnx.py --member ST_A   # 378x672
    python export_onnx.py --member E      # 378x672
    python export_onnx.py --member L4     # 378x672
    python export_onnx.py --member H      # 472x840
    python export_onnx.py --member H2     # 568x1008
    python export_onnx.py --all

Verified: torch vs onnxruntime max |delta| <= 8e-5 (float32 noise).
Requires `onnx` and `onnxruntime` (see requirements.txt); the TorchScript exporter
(`dynamo=False`) is used, so `onnxscript` is not needed.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import modelzoo  # noqa: E402

# member -> (checkpoint, onnx output, (H, W) used at inference)
MEMBERS = {
    "ST_A": ("weights/ST_A_best.pt", "weights/ST_A_best.onnx", (378, 672)),
    "E": ("weights/E_swins_prec_best.pt", "weights/E_swins_prec_best.onnx", (378, 672)),
    "L4": ("weights/L4_unetd_100_best.pt", "weights/L4_unetd_100_best.onnx", (378, 672)),
    "H": ("weights/H_swins_hi_best.pt", "weights/H_swins_hi_best.onnx", (472, 840)),
    "H2": ("weights/H2_swins_15x_best.pt", "weights/H2_swins_15x_best.onnx", (568, 1008)),
}


def export_one(member, opset=17):
    ckpt_path, out_path, (h, w) = MEMBERS[member]
    ckpt_path = os.path.join(HERE, ckpt_path)
    out_path = os.path.join(HERE, out_path)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = modelzoo.build_model(ck["model_name"], pretrained=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    dummy = torch.zeros(1, 3, h, w)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )

    msg = ""
    try:
        import onnxruntime as ort
        with torch.no_grad():
            ref = model(dummy).numpy()
        got = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"]).run(
            None, {"input": dummy.numpy()})[0]
        msg = f"  max|delta|={float(np.abs(ref - got).max()):.2e}"
    except ImportError:
        msg = "  (onnxruntime 未安装，跳过数值校验)"

    print(f"{os.path.basename(out_path):26s} {ck['model_name']:10s} {h}x{w}  "
          f"{os.path.getsize(out_path)/1e6:7.1f} MB{msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", choices=list(MEMBERS), help="单个成员")
    ap.add_argument("--all", action="store_true", help="导出全部 5 个")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not args.all and not args.member:
        ap.error("指定 --member 或 --all")
    for m in (MEMBERS if args.all else [args.member]):
        export_one(m, args.opset)


if __name__ == "__main__":
    main()
