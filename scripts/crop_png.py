"""按矩形框裁剪 PNG, 保留原始模式 (RGB 或 L)。

用法:
    python.exe crop_png.py <in.png> <out.png> --box x0,y0,x1,y1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="裁剪 PNG")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--box", required=True, help="x0,y0,x1,y1")
    args = ap.parse_args()

    x0, y0, x1, y1 = (int(v) for v in args.box.replace("x", ",").split(","))
    im = Image.open(args.src)
    w, h = im.size
    box = (max(0, x0), max(0, y0), min(w, x1), min(h, y1))
    out = im.crop(box)

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, format="PNG")
    print(f"{Path(args.src).name} {im.mode} {im.size}  --crop {box}-->  "
          f"{dst.name} {out.mode} {out.size}   输出尺寸x4 = {out.size[0] * 4} x {out.size[1] * 4}")
    if out.size[0] * out.size[1] * 4 * 3 > 11 * 2**20:
        print(f"  注: x4 后全分辨率 float32 缓冲 = "
              f"{out.size[0] * 4 * out.size[1] * 4 * 3 * 4 / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
