"""量化比较两张同 alpha 的合成图, 谁的边缘白边更重。

原理: alpha 越低越靠近主体外缘。如果模型把背景"白化"了, 低 alpha 带的
亮度会明显高于主体内部; 反之则接近。

用法:
    python.exe compare_halo.py <A.png> <B.png> [--label-a A] [--label-b B]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
BANDS = ((1, 32), (32, 64), (64, 128), (128, 192), (192, 254), (255, 255))


def analyze(path: Path) -> dict:
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im)
    rgb = arr[:, :, :3].astype(np.float32)
    a = arr[:, :, 3]
    lum = rgb @ LUMA
    out = {"path": str(path), "size": im.size, "bands": {}}
    for lo, hi in BANDS:
        m = (a >= lo) & (a <= hi)
        if m.any():
            out["bands"][(lo, hi)] = (float(lum[m].mean()), int(m.sum()))
    inner = out["bands"].get((255, 255), (float(lum.mean()), 1))[0]
    fringe = out["bands"].get((1, 32), (inner, 1))[0]
    out["inner_lum"] = inner
    out["fringe_lum"] = fringe
    out["halo_index"] = fringe - inner
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="边缘白边量化对比")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    ra = analyze(Path(args.a))
    rb = analyze(Path(args.b))

    print(f"{'':26s} {args.label_a:>14s} {args.label_b:>14s}")
    print("-" * 58)
    print(f"{'尺寸':26s} {str(ra['size']):>14s} {str(rb['size']):>14s}")
    for lo, hi in BANDS:
        va = ra["bands"].get((lo, hi))
        vb = rb["bands"].get((lo, hi))
        fa = f"{va[0]:8.1f} ({va[1]})" if va else "-"
        fb = f"{vb[0]:8.1f} ({vb[1]})" if vb else "-"
        print(f"{f'alpha {lo}-{hi} 平均亮度':26s} {fa:>18s} {fb:>18s}")
    print("-" * 58)
    print(f"{'主体内部亮度 (alpha=255)':26s} {ra['inner_lum']:>14.1f} {rb['inner_lum']:>14.1f}")
    print(f"{'外缘亮度 (alpha 1-32)':26s} {ra['fringe_lum']:>14.1f} {rb['fringe_lum']:>14.1f}")
    print(f"{'白边指数 (外缘 - 内部)':26s} {ra['halo_index']:>14.1f} {rb['halo_index']:>14.1f}")
    print()
    print("白边指数越负 / 越接近 0 = 边缘越干净; 明显为正 = 存在白化描边")
    winner = args.label_a if ra["halo_index"] <= rb["halo_index"] else args.label_b
    print(f"=> 更干净的是: {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
