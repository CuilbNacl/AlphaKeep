"""校验拆分无损: alpha>0 的区域 RGB 必须与原图逐像素相同。

用法:
    python.exe check_split.py <src_rgba.png> <rgb.png> <alpha.png>
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    src_p, rgb_p, a_p = sys.argv[1:4]

    src_im = Image.open(src_p)
    rgb_im = Image.open(rgb_p)
    a_im = Image.open(a_p)

    src = np.asarray(src_im.convert("RGBA"))
    rgb = np.asarray(rgb_im.convert("RGB"))
    alpha = np.asarray(a_im.convert("L"))

    ok = True

    print(f"源图    : mode={src_im.mode} size={src_im.size}")
    print(f"rgb.png : mode={rgb_im.mode} size={rgb_im.size}  通道数={rgb.shape[2]}")
    print(f"alpha   : mode={a_im.mode} size={a_im.size}")

    if rgb.shape[2] != 3:
        print("  [失败] rgb.png 不是三通道")
        ok = False
    else:
        print("  [通过] rgb.png 是三通道, 不含 alpha")

    if src.shape[:2] != rgb.shape[:2] or src.shape[:2] != alpha.shape:
        print("  [失败] 尺寸不一致")
        ok = False

    if not np.array_equal(src[:, :, 3], alpha):
        d = np.abs(src[:, :, 3].astype(int) - alpha.astype(int))
        print(f"  [失败] alpha 通道不一致, 最大差异 {d.max()}")
        ok = False
    else:
        print("  [通过] alpha 通道与原图逐像素一致")

    m = src[:, :, 3] > 0
    d = np.abs(src[:, :, :3][m].astype(int) - rgb[m].astype(int))
    if d.max() != 0:
        print(f"  [失败] alpha>0 区域 RGB 被改动, 最大差异 {d.max()}, 涉及 {int((d.max(axis=1) > 0).sum())} 像素")
        ok = False
    else:
        print(f"  [通过] alpha>0 的 {int(m.sum())} 个像素 RGB 完全未改动")

    mz = src[:, :, 3] == 0
    print(f"  alpha==0 区域已重填 {int(mz.sum())} 像素 (这是预期行为)")

    print("\n结论:", "拆分无损" if ok else "有问题")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
