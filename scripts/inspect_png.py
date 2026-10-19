"""只读分析单个 PNG 的 alpha 情况。

用法:
    python.exe inspect_png.py <png 路径>
不写任何文件，只打印。
"""
import sys

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main(path: str) -> None:
    im = Image.open(path)
    print(f"file      : {path}")
    print(f"mode      : {im.mode}")
    print(f"size      : {im.size}  (w x h)")
    print(f"format    : {im.format}")
    print(f"info keys : {sorted(im.info.keys())}")

    rgba = im.convert("RGBA")
    arr = np.asarray(rgba)
    rgb = arr[:, :, :3]
    a = arr[:, :, 3]
    h, w = a.shape
    total = a.size

    print("\n-- alpha 分档 --")
    buckets = {
        "alpha == 0   完全透明": int((a == 0).sum()),
        "0 < alpha < 255  半透明": int(((a > 0) & (a < 255)).sum()),
        "alpha == 255 完全不透明": int((a == 255).sum()),
    }
    for k, v in buckets.items():
        print(f"  {k:>22s} : {v:10d}  ({v / total * 100:6.2f}%)")

    uniq_a = np.unique(a)
    print(f"\nalpha 取值种类数 : {len(uniq_a)}")
    if len(uniq_a) <= 24:
        vals, cnts = np.unique(a, return_counts=True)
        print("  取值 -> 计数 :", {int(v): int(c) for v, c in zip(vals, cnts)})

    # 有效内容的包围盒
    ys, xs = np.where(a > 0)
    if len(ys):
        print(f"\n内容包围盒(alpha>0) : x {xs.min()}..{xs.max()}  y {ys.min()}..{ys.max()}")

    # 完全透明区里"藏着"的 RGB —— 白底/黑底的根源
    if (a == 0).any():
        t = rgb[a == 0]
        print("\n-- 完全透明区藏着的 RGB --")
        print("  均值 :", t.mean(axis=0).round(1).tolist())
        print("  中位 :", np.median(t, axis=0).astype(int).tolist())
        print("  最小 :", t.min(axis=0).tolist())
        print("  最大 :", t.max(axis=0).tolist())
        u = np.unique(t.reshape(-1, 3), axis=0)
        print("  不同颜色数 :", len(u))
        if len(u) <= 12:
            print("  具体颜色 :", u.tolist())

    # 半透明边缘像素
    edge = (a > 0) & (a < 255)
    if edge.any():
        e = rgb[edge]
        print("\n-- 半透明边缘像素 RGB --")
        print("  数量 :", int(edge.sum()), " 均值 :", e.mean(axis=0).round(1).tolist())

    # 判断是"直通 alpha (straight)" 还是"已经合成到白底"
    print("\n-- 按 alpha 分档看 RGB 亮度（判断是否已被合成到某底色）--")
    lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    for lo, hi in ((1, 32), (32, 64), (64, 128), (128, 192), (192, 254), (255, 255)):
        m = (a >= lo) & (a <= hi)
        if m.any():
            print(f"  alpha {lo:3d}-{hi:3d} : n={int(m.sum()):9d}  "
                  f"RGB均值={rgb[m].mean(axis=0).round(1).tolist()}  亮度={lum[m].mean():6.1f}")
    print("  说明: 若 alpha 越低亮度越接近 255 -> 已合成到白底(需要反合成);")
    print("        若各档亮度大致持平或越低越暗 -> 直通 alpha, 边缘 RGB 可直接用。")

    # 不透明区颜色（填色策略参考）
    op = rgb[a == 255]
    if len(op):
        print("\n-- 不透明区颜色 --")
        print("  中位色 :", np.median(op, axis=0).astype(int).tolist())

    # 四条边一圈的 alpha：判断是不是"底边/边缘"问题
    print("\n-- 最外一圈 alpha (min / mean / max) --")
    for name, line in (("上", a[0]), ("下", a[-1]), ("左", a[:, 0]), ("右", a[:, -1])):
        print(f"  {name} : {line.min():3d} / {line.mean():6.1f} / {line.max():3d}")

    # 四周 1px 内边框里非零 alpha 的占比，判断贴边程度
    ring = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    print(f"\n最外圈非零 alpha 占该圈 {int((ring > 0).sum())}/{ring.size} "
          f"({(ring > 0).mean() * 100:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1])
