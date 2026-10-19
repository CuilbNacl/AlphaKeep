"""把原图和 4 倍放大结果拼成 README 用的对比图（4 块）。

布局：
  ① 原图（棋盘格）        ② 放大 4× 后缩回同尺寸（棋盘格）
  ③ 原图的像素密度（×4 显示）  ④ 同一区域放大后的 1:1

③④ 是同一块区域、同样的显示尺寸：左边能看到原图的像素块，
右边能看到放大后的平滑细节 —— 这才是「4 倍」带来的实际差别。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

BASE = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "_cmp" / "vosr2_input" / "sample.png"
UP = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "_cmp" / "vosr2_output" / "sample_x4.png"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else BASE / "docs" / "compare.png"

PANEL = 340
GAP = 16
MARGIN = 26
TITLE_H = 54
LABEL_H = 32
NOTE_H = 42
BG = (34, 38, 46)
FG = (234, 237, 242)
DIM = (152, 160, 174)


def font(size: int, bold: bool = False):
    for p in ((r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
              r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def checker(w: int, h: int, cell: int = 14) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((yy // cell) + (xx // cell)) % 2
    v = np.where(m == 0, 216, 172).astype(np.uint8)
    return Image.fromarray(np.repeat(v[:, :, None], 3, axis=2), "RGB")


def on_checker(im: Image.Image, size: int) -> Image.Image:
    bg = checker(size, size)
    im = im.convert("RGBA")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    bg.paste(im, (0, 0), im)
    return bg


src = Image.open(SRC).convert("RGBA")
up = Image.open(UP).convert("RGBA")
ratio = up.width // src.width
print(f"   原图 {src.size}   放大 {up.size}   倍数 {ratio}")

# 取景：卡片左下角（源坐标 80,432 附近），100x100
REG = 100
cx, cy = 80, 432
l, t = max(0, cx - REG // 2), max(0, cy - REG // 2)
src_crop = src.crop((l, t, l + REG, t + REG))
up_crop = up.crop((l * ratio, t * ratio, (l + REG) * ratio, (t + REG) * ratio))
print(f"   源取景 ({l},{t}) {REG}x{REG}   放大取景 {up_crop.size}")

panels = [
    (on_checker(src, PANEL), "① 原图", f"{src.width}×{src.height}"),
    (on_checker(up, PANEL), "② 放大 4×", f"{up.width}×{up.height}"),
    (on_checker(src_crop.resize((PANEL, PANEL), Image.NEAREST), PANEL),
     "③ 原图的像素", "同一区域 ×4 显示"),
    (on_checker(up_crop, PANEL), "④ 放大后的细节", "同一区域 1:1"),
]

W = MARGIN * 2 + PANEL * 4 + GAP * 3
H = MARGIN * 2 + TITLE_H + LABEL_H + PANEL + NOTE_H

fig = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(fig)
d.text((MARGIN, MARGIN + 2),
       "透明放大镜 AlphaKeep —— 4 倍放大，透明通道完整保留",
       font=font(22, True), fill=FG)

y = MARGIN + TITLE_H
for i, (img, main, sub) in enumerate(panels):
    x = MARGIN + i * (PANEL + GAP)
    d.text((x + 2, y), main, font=font(16, True),
           fill=FG if i >= 2 else DIM)
    tw = d.textlength(main, font=font(16, True))
    d.text((x + 2 + tw + 10, y + 1), sub, font=font(14), fill=DIM)
    fig.paste(img, (x, y + LABEL_H))
    d.rectangle((x, y + LABEL_H, x + PANEL - 1, y + LABEL_H + PANEL - 1),
                outline=(60, 66, 78), width=1)


d.text((MARGIN, H - MARGIN - 22),
       "棋盘格透出的区域即为透明。① ② 同尺寸对比；③ ④ 是同一块区域的像素级对比 —— "
       "放大后边缘依旧是平滑过渡，没有白边或黑边。",
       font=font(15), fill=DIM)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.save(OUT, "PNG", optimize=True)

a = np.asarray(up)[:, :, 3]
print(f"   已生成 {OUT}   {fig.size}   {OUT.stat().st_size/1024:.1f} KB")
print(f"   4 倍图 alpha: 0..255   全透明 {(a==0).sum()}   半透明 {((a>0)&(a<255)).sum()}")
