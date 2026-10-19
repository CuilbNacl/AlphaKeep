"""合成一张用于对比图的测试素材（RGBA，带柔和 alpha 渐变）。

设计目标：让「透明边缘」和「细节」同时可见
- 卡片有柔和的投影（alpha 渐变）—— 劣质放大会在这里出白边/黑边
- 卡片内有细线条和小字 —— 体现细节重绘
- 有一圈硬边的描边 —— 体现抗锯齿
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

W = H = 512
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample.png")

# ---------------------------------------------------------------- 渐变
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
t = (yy / H * 0.75 + xx / W * 0.25)
r = (40 + 200 * t).clip(0, 255)
g = (110 + 90 * (1 - t) ** 1.4).clip(0, 255)
b = (230 - 120 * t ** 0.8).clip(0, 255)
grad = np.dstack([r, g, b]).astype(np.uint8)
grad_im = Image.fromarray(grad, "RGB").convert("RGBA")

# ---------------------------------------------------------------- 卡片遮罩
card_mask = Image.new("L", (W, H), 0)
ImageDraw.Draw(card_mask).rounded_rectangle(
    (80, 80, W - 80, H - 80), radius=44, fill=255)

# ---------------------------------------------------------------- 投影
shadow = Image.new("L", (W, H), 0)
ImageDraw.Draw(shadow).rounded_rectangle(
    (84, 96, W - 76, H - 68), radius=44, fill=110)
shadow = shadow.filter(ImageFilter.GaussianBlur(18))

canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
canvas.paste(Image.new("RGBA", (W, H), (12, 18, 30, 255)), (0, 0), shadow)

# ---------------------------------------------------------------- 卡片本体
card = Image.new("RGBA", (W, H), (0, 0, 0, 0))
card.paste(grad_im, (0, 0), card_mask)
canvas = Image.alpha_composite(canvas, card)

draw = ImageDraw.Draw(canvas)

# 内侧硬边描边（体现抗锯齿）
draw.rounded_rectangle((80, 80, W - 80, H - 80), radius=44,
                       outline=(255, 255, 255, 235), width=3)

# 细线条（体现细节重绘）
for i, y in enumerate(range(160, 361, 46)):
    x2 = 140 + int(150 * (0.5 + 0.5 * np.sin(i * 1.9)))
    draw.line((128, y, x2, y), fill=(255, 255, 255, 205), width=2)
    draw.line((x2, y, 384, y), fill=(255, 255, 255, 90), width=1)

# 柔和的发光圆（alpha 渐变，肉眼最容易看出差别）
glow = Image.new("L", (W, H), 0)
ImageDraw.Draw(glow).ellipse((300, 300, 452, 452), fill=210)
glow = glow.filter(ImageFilter.GaussianBlur(26))
canvas.paste(Image.new("RGBA", (W, H), (255, 246, 216, 255)), (0, 0), glow)

canvas.save(OUT)
im = Image.open(OUT)
print(f"   已生成 {OUT}   {im.format} {im.mode} {im.size}  {OUT.stat().st_size/1024:.1f} KB")
print(f"   透明像素 {int((np.asarray(im)[:,:,3]==0).sum())} / {W*H}")
print(f"   半透明像素 {int(((np.asarray(im)[:,:,3]>0)&(np.asarray(im)[:,:,3]<255)).sum())}")
