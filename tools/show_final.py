import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "docs" / "final_select.json", encoding="utf-8"))
print("singles:", {k: round(v, 4) for k, v in sorted(d["singles"].items(), key=lambda kv: -kv[1])})
for k, v in sorted(d["table"].items(), key=lambda kv: -kv[1]["avg_iou"])[:8]:
    print(f"{k:52s} v1={v['v1']['iou']:.4f} v2={v['v2']['iou']:.4f} "
          f"avg={v['avg_iou']:.4f} thr={v['v1']['threshold']}")
