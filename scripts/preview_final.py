"""最终成品的验收预览: 贴不同底色 + 1:1 边缘放大, 用来找白边/彩带。

用法:
    python.exe preview_final.py <final_rgba.png> <out_dir> [--width 620] [--crop 700]

产出:
    <out_dir>/final_on_bg.png     棋盘底 / 纯黑底 / 纯白底 三块并排
    <out_dir>/final_edge_crop.png 在最强的 alpha 边缘处取 1:1 放大, 棋盘底 vs 纯黑底
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


def checkerboard(w: int, h: int, cell: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((yy // cell) + (xx // cell)) % 2
    v = np.where(m == 0, 214, 168).astype(np.uint8)
    return np.repeat(v[:, :, None], 3, axis=2)


def over(rgba: np.ndarray, bg: np.ndarray) -> np.ndarray:
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return (rgba[:, :, :3].astype(np.float32) * a + bg.astype(np.float32) * (1 - a)
            ).round().clip(0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description="最终成品验收预览")
    ap.add_argument("final")
    ap.add_argument("out_dir")
    ap.add_argument("--width", type=int, default=620)
    ap.add_argument("--crop", type=int, default=700)
    args = ap.parse_args()

    im = Image.open(args.final).convert("RGBA")
    rgba = np.asarray(im)
    H, W = rgba.shape[:2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) 整图贴三种底 ----
    w = args.width
    h = int(round(H * w / W))
    small = np.asarray(im.resize((w, h), Image.LANCZOS))
    panels = [
        over(small, checkerboard(w, h, 16)),
        over(small, np.zeros((h, w, 3), np.uint8)),
        over(small, np.full((h, w, 3), 255, np.uint8)),
    ]
    sep = np.full((h, 3, 3), 128, np.uint8)
    canvas = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
    p1 = out_dir / "final_on_bg.png"
    Image.fromarray(canvas, "RGB").save(p1)

    # ---- 2) 最强边缘处 1:1 放大 ----
    a = rgba[:, :, 3].astype(np.int16)
    gy = np.abs(np.diff(a, axis=0))[:-1, :]
    gx = np.abs(np.diff(a, axis=1))[:, :-1]
    g = gy[: gx.shape[0], : gx.shape[1]] + gx[: gy.shape[0], : gy.shape[1]]
    cy, cx = np.unravel_index(int(np.argmax(g)), g.shape)

    c = args.crop
    y0 = int(np.clip(cy - c // 2, 0, max(0, H - c)))
    x0 = int(np.clip(cx - c // 2, 0, max(0, W - c)))
    crop = rgba[y0:y0 + c, x0:x0 + c]
    ch, cw = crop.shape[:2]
    edge_panels = [
        over(crop, checkerboard(cw, ch, 16)),
        over(crop, np.zeros((ch, cw, 3), np.uint8)),
    ]
    canvas2 = np.concatenate(
        [edge_panels[0], np.full((ch, 3, 3), 128, np.uint8), edge_panels[1]], axis=1
    )
    p2 = out_dir / "final_edge_crop.png"
    Image.fromarray(canvas2, "RGB").save(p2)

    print(f"整图预览 -> {p1}   左:棋盘底 中:纯黑底 右:纯白底")
    print(f"边缘放大 -> {p2}   左:棋盘底 右:纯黑底")
    print(f"最强边缘位置: x={cx} y={cy}   裁剪窗口 x{x0}..{x0 + cw} y{y0}..{y0 + ch} (1:1 像素)")
    print(f"全图尺寸: {W} x {H}")

    # 量化: 沿边缘取样, 看主体外侧有没有明显变亮(白边)或跳色(彩边)
    band = 12
    ys, xs = np.where(a > 0)
    print(f"\nalpha>0 像素: {len(ys)}   alpha==0 像素: {int((a == 0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
