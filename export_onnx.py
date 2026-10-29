# -*- coding: utf-8 -*-
"""Export one ensemble member to ONNX (the challenge asks for .pth *and* .onnx).

The Swin-S / U-Net-DS definitions live in the training package (`uavcrack/modelzoo.py`).
Drop that module into `src/modelzoo.py` (it must expose `build(model_name)`), then:

    python export_onnx.py --member H2 --ckpt weights/H2_swins_15x_best.pt \
                          --out weights/H2_swins_15x.onnx --height 568 --width 1008

Check the result:
    python -c "import onnxruntime as ort, numpy as np; \
s=ort.InferenceSession('weights/H2_swins_15x.onnx'); \
print(s.run(None, {'input': np.zeros((1,3,568,1008), np.float32)})[0].shape)"
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src import modelzoo            # noqa: E402  (user-provided)
except ImportError:
    modelzoo = None

# spatial size each member was trained at (must match the checkpoint)
TRAIN_SIZE = {
    "ST_A": (378, 672),
    "E": (378, 672),
    "L4": (378, 672),
    "H": (472, 840),
    "H2": (568, 1008),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", choices=list(TRAIN_SIZE), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic", action="store_true")
    args = ap.parse_args()

    if modelzoo is None:
        raise SystemExit(
            "缺少 src/modelzoo.py：Swin-S / U-Net-DS 的定义在训练代码包（uavcrack/modelzoo.py）里。\n"
            "把该文件复制到 src/modelzoo.py 并实现 build(model_name) 后重跑。"
        )

    h, w = args.height or TRAIN_SIZE[args.member][0], args.width or TRAIN_SIZE[args.member][1]
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    name = ckpt.get("model_name")
    model = modelzoo.build(name)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        model, torch.zeros(1, 3, h, w), args.out,
        input_names=["input"], output_names=["logits"], opset_version=args.opset,
        dynamic_axes={"input": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}} if args.dynamic else None,
        do_constant_folding=True,
    )
    print(f"[export] {name} {h}x{w} -> {args.out} "
          f"({os.path.getsize(args.out)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
