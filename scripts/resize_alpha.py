"""把 alpha 蒙版独立放大到与模型输出同尺寸。

用法:
    python.exe resize_alpha.py <alpha.png> <out.png> --factor 4
                              [--target WxH] [--mode lanczos|nearest|bicubic]
                              [--blur 1.0] [--erode 0]

要点:
  * 蒙版绝对不要过 AI 模型(会把蒙版啃出洞、长出彩带), 只用普通插值。
  * 原蒙版有 256 级羽化 -> 用 lanczos 放大, 羽化会自然保留。
  * 放大后再补一点点高斯模糊, 抹掉插值可能产生的硬台阶。
  * --erode N: 把蒙版向内收 N 像素, 用来切掉合成后残留的一圈浅色描边。
  * --target 用于强制对齐模型输出的实际尺寸(优先于 --factor)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageFilter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODES = {
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
}


def parse_size(text: str) -> tuple[int, int]:
    sep = "x" if "x" in text.lower() else ","
    a, b = text.lower().split(sep)[:2]
    return int(a), int(b)


def main() -> int:
    ap = argparse.ArgumentParser(description="独立放大 alpha 蒙版")
    ap.add_argument("alpha")
    ap.add_argument("out")
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--target", default=None, help="目标尺寸, 形如 8908x8760")
    ap.add_argument("--mode", choices=tuple(MODES), default="lanczos")
    ap.add_argument("--blur", type=float, default=1.0, help="放大后补的模糊半径, 0 = 不补")
    ap.add_argument("--erode", type=int, default=0, help="向内收缩像素数, 用于切掉浅色描边")
    args = ap.parse_args()

    src = Path(args.alpha)
    if not src.is_file():
        print(f"[错误] 找不到: {src}")
        return 2

    im = Image.open(src).convert("L")
    w0, h0 = im.size

    if args.target:
        tw, th = parse_size(args.target)
    else:
        tw, th = int(round(w0 * args.factor)), int(round(h0 * args.factor))

    out = im.resize((tw, th), MODES[args.mode])

    if args.blur and args.blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(args.blur))

    if args.erode and args.erode > 0:
        size = args.erode * 2 + 1
        for _ in range(1):
            out = out.filter(ImageFilter.MinFilter(size))

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, format="PNG")

    print(f"输入   : {src}  {w0} x {h0}")
    print(f"模式   : {args.mode}   blur={args.blur}  erode={args.erode}")
    print(f"输出   : {dst}  {tw} x {th}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
