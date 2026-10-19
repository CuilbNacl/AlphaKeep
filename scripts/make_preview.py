"""生成对比预览, 用来肉眼确认"填色"这一步有没有把白底问题解决掉。

用法:
    python.exe make_preview.py <src_rgba.png> <rgb.png> <alpha.png> <out.png> [--width 720]

三块并排:
  1) 原始 RGB —— alpha 被丢掉后模型实际会看到的样子(近白背景)
  2) 填色后的 RGB —— 我们真正喂给模型的图
  3) 填色后 RGB + alpha 贴棋盘底 —— 确认主体本身没被填色破坏
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


def checkerboard(w: int, h: int, cell: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((yy // cell) + (xx // cell)) % 2
    v = np.where(m == 0, 214, 168).astype(np.uint8)
    return np.repeat(v[:, :, None], 3, axis=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成填色前后对比预览")
    ap.add_argument("src")
    ap.add_argument("rgb")
    ap.add_argument("alpha")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=720)
    args = ap.parse_args()

    src = Image.open(args.src).convert("RGBA")
    rgba = np.asarray(src)
    orig_rgb = Image.fromarray(rgba[:, :, :3].copy(), "RGB")

    filled = Image.open(args.rgb).convert("RGB")
    alpha = Image.open(args.alpha).convert("L")

    w = args.width
    h = int(round(filled.height * w / filled.width))
    orig_s = orig_rgb.resize((w, h), Image.LANCZOS)
    fill_s = filled.resize((w, h), Image.LANCZOS)
    al_s = alpha.resize((w, h), Image.LANCZOS)

    comp = np.asarray(fill_s).astype(np.float32)
    af = np.asarray(al_s).astype(np.float32)[:, :, None] / 255.0
    bg = checkerboard(w, h).astype(np.float32)
    panel3 = (comp * af + bg * (1 - af)).round().astype(np.uint8)

    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    canvas = np.concatenate(
        [np.asarray(orig_s), sep, np.asarray(fill_s), sep, panel3], axis=1
    )

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, "RGB").save(dst, format="PNG")

    # 同时给出量化对比: 透明区平均颜色 / 边缘一圈的对比度
    print(f"预览 -> {dst}   ({canvas.shape[1]} x {canvas.shape[0]})")
    print("左: 原始RGB(丢掉alpha后模型看到的)   中: 填色后RGB   右: 填色后RGB+alpha(棋盘底)")

    a0 = rgba[:, :, 3]
    m = a0 == 0
    if m.any():
        print(f"\n完全透明区平均颜色:")
        print(f"  填色前 : {rgba[:, :, :3][m].mean(axis=0).round(1).tolist()}")
        print(f"  填色后 : {np.asarray(filled)[m].mean(axis=0).round(1).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
