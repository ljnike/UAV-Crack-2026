# UAV-Crack-2026 · 2026 VLP 挑战赛参赛作品

长安大学人工智能创新大赛（2026）专项命题挑战赛 · **赛道 D：UAV-Crack 无人机路面裂缝分割**

> 本仓库为 **2026 VLP 挑战赛参赛作品**，遵循 MIT 许可，仅供赛事评审与非商业研究使用。

线上成绩（方法名 `UAV-fyk v4`）：**mIoU 82.33 / 第 5（79 个方法）**，提交时为第 2 / 74。
Crack F1 80.30、Crack IoU 67.08、Crack Precision 80.54、Crack Recall 80.06、aAcc 97.69。

---

## 1. 赛题与数据

- 数据集：UAV-CrackX，1500 张像素级标注无人机影像（X4 / X8 / X16 各 500 张）
- 原始影像 2688×1512，按 4×4 切块为 **672×378** 分块参与训练与评测
- 划分：train 1200 张（有标注）、val 300 张（无标注，线上评测）
- 指标：mIoU、Crack F1、Crack IoU、Crack Precision、Crack Recall、aAcc

> ⚠️ 官方 val 与 train 的**源照片重叠率约 96.5%**。按 tile 随机划分会让验证指标虚高；
> 本仓库的本地评估一律按 `source_photo_id` 做 **photo-disjoint** 划分。

## 2. 方法：5 成员 Swin-S / U-Net-DS 异构集成

| 成员 | 结构 | 参数量 | 训练分辨率 | 推理尺度 | 单模型验证 IoU |
|---|---|---|---|---|---|
| `ST_A` | Swin-S（软伪标签自训练） | 49.7 M | 384×672 | 0.85 / 1.0 / 1.15 | 0.6346 |
| `E` | Swin-S（Tversky α=0.7，精确率偏置） | 49.7 M | 384×672 | 0.85 / 1.0 / 1.15 | 0.6288 |
| `L4` | U-Net-DS / ResNet34（边界校正） | 25.7 M | 384×672 | 1.0 | 0.5599 |
| `H` | Swin-S（1.25× 高分辨率，从零训练） | 49.7 M | 480×840 | 1.25 | 0.6259 |
| `H2` | Swin-S（1.5× 高分辨率，从零训练） | 49.7 M | 568×1008 | 1.5 | 0.6469 |

**推理与融合**

```
val tile ─ 每个成员按自己的尺度 + 翻转 TTA → 概率图
        ─ AE 组合项 = logit(0.10·ST_A + 0.65·E)/0.75        权重 0.35
        ─ L4  0.10 ｜ H  0.25 ｜ H2  0.30
        ─ logit 域加权平均 → 阈值 0.495 → 去除 <64 px 连通域 → result/*.png
```

- **尺度必须与成员匹配**：高分辨率成员若按 1.0 推理，本地 IoU 会掉 5–12 个点（细裂缝被重采样抹平）
- **logit 域融合而非概率域**：更陡的响应，能压掉中等置信度的背景
- **阈值 0.495** 由预测前景密度标定（对齐训练标注中位数 5.9%），不是本地 IoU 最优点（≈0.41）
- 输出 `uint8` 单通道 PNG，取值 **{0, 1}**（0=路面，1=裂缝），672×378，文件名与输入一致

**本地 photo-disjoint 验证**：Crack IoU 0.6703 / mIoU 0.8196（v1）→ 0.6738 / 0.8222（v2 融合权重）；
最强单模型（含多尺度 + 翻转）0.6620，集成净增 **+1.2 IoU 点**。

## 3. 环境

```bash
pip install -r requirements.txt
```

## 4. 数据准备

```
data/UAV-Crack-dataset/
├── leftimg8bit/
│   ├── train/{UAV-CrackX4,UAV-CrackX8,UAV-CrackX16}/*.jpg
│   └── val/*.jpg                      # 300 张，评测输入
└── gtFine/
    └── train/{UAV-CrackX4,UAV-CrackX8,UAV-CrackX16}/*.png
```

## 5. 复现（两步）

```bash
# 步骤 1：各成员导出测试集概率（每个成员用各自的尺度 + 翻转 TTA）
python predictz.py --ckpt weights/ST_A_best.pt        --out sub/m1 --scales 0.85 1.0 1.15 --flip --save-probs sub/p_STA.npz
python predictz.py --ckpt weights/E_swins_prec_best.pt --out sub/m2 --scales 0.85 1.0 1.15 --flip --save-probs sub/p_E.npz
python predictz.py --ckpt weights/L4_unetd_100_best.pt --out sub/m3 --scales 1.0 --flip --save-probs sub/p_L4.npz
python predictz.py --ckpt weights/H_swins_hi_best.pt  --out sub/m4 --scales 1.25 --flip --save-probs sub/p_H.npz
python predictz.py --ckpt weights/H2_swins_15x_best.pt --out sub/m5 --scales 1.5 --flip --save-probs sub/p_H2.npz

# 步骤 2：logit 加权融合 + 阈值 + 后处理 + 打包自检
#   原始实现（产出提交文件的那一条）：tools/ensemble_test.py
python tools/ensemble_test.py --probs sub/p_STA.npz sub/p_E.npz sub/p_L4.npz sub/p_H.npz sub/p_H2.npz \
    --weights 0.04667 0.30333 0.10 0.25 0.30 --logit \
    --threshold 0.495 --min-size 64 --fg-value 1 \
    --out-masks sub/masks_v4 --out-zip sub/result_v4.zip
#   等价的精简版（含 AE 组合项与自检提示）：ensemble.py
python ensemble.py --probs sub/p_STA.npz sub/p_E.npz sub/p_L4.npz sub/p_H.npz sub/p_H2.npz \
                   --config configs/ensemble.json --out result --zip result.zip

# 步骤 3：提交格式自检
python check_result.py --result sub/masks_v4 --data-root data/UAV-Crack-dataset --pack result.zip
```

`check_result.py` 逐项校验：300 个文件、672×378、取值 {0,1}、文件名与 val 一一对应、
压缩包内仅 `result/` 目录。

## 6. 权重

| 文件 | 大小 | SHA256（前 16 位） |
|---|---|---|
| `weights/ST_A_best.pt` | 199.1 MB | `824a1843889e6049` |
| `weights/E_swins_prec_best.pt` | 199.1 MB | `b833f6ec26d43312` |
| `weights/L4_unetd_100_best.pt` | 103.0 MB | `a856b378caed584f` |
| `weights/H_swins_hi_best.pt` | 199.1 MB | `5a680ed72d2ac9c0` |
| `weights/H2_swins_15x_best.pt` | 199.1 MB | `17471427b0dd98ce` |

**ONNX 版本**（比赛要求：PyTorch 模型需另附 ONNX）：`weights/*.onnx` 共 5 个，约 916 MB，
由 `export_onnx.py --all` 生成，输入尺寸与成员一一对应（378×672 / 472×840 / 568×1008），
与 PyTorch 输出的最大误差 ≤ 8e-5。

完整校验值、融合权重、每个成员的推理尺度见 [`weights/MANIFEST.json`](weights/MANIFEST.json)。
下载：**[百度网盘](https://pan.baidu.com/s/1Iu5rxPz44ByAOzEznl4Exg?pwd=1234)（提取码 1234，链接长期有效）**。

> 权重不入 git（见 `.gitignore`），随 Release / 网盘分发。

## 7. 目录结构

```
UAV-Crack-2026/
├── modelzoo.py         # 全部网络定义（Swin-UNet / SegFormer / DeepLabV3+ / U-Net-DS 等）
├── trainz.py           # 训练主程序（Tversky / BCE-Dice / boundary 项、EMA、早停）
├── predictz.py         # 单成员推理：多尺度 + 翻转 TTA，导出概率 npz
├── evalz.py            # 本地评估（Crack IoU / mIoU / F1 / P / R）
├── splits.py           # photo-disjoint 数据划分（按 source_photo_id 分组）
├── metrics.py          # 指标实现
├── config.py           # 路径与全局配置
├── ensemble.py         # 融合精简版：AE 组合项 + logit 加权 + 阈值 + 去噪
├── check_result.py     # 提交格式自检（数量/尺寸/取值/命名/目录）
├── export_onnx.py      # PyTorch -> ONNX（--all 一次导 5 个）
├── tools/              # 原始实验脚本（ensemble_test / blend / self_train / 阈值标定…）
├── configs/ensemble.json
├── src/{fusion,postprocess}.py
├── weights/            # 5 个 .pt + 5 个 .onnx + MANIFEST.json（不入库）
├── requirements.txt
├── LICENSE             # MIT
└── README.md
```

## 8. 许可与致谢

MIT License，见 [LICENSE](LICENSE)。数据集遵循 CC BY-NC-SA 3.0，仅限本次比赛非商业使用，禁止二次分发。
基线参考 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)。
