"""验收: 确认最终文件真的是透明 PNG, 且尺寸/边角符合预期。

用法:
    python.exe verify_png.py <file.png> [--expect-size WxH] [--expect-factor 4 --meta meta.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def parse_size(t: str) -> tuple[int, int]:
    a, b = t.lower().replace(",", "x").split("x")[:2]
    return int(a), int(b)


def main() -> int:
    ap = argparse.ArgumentParser(description="验收透明 PNG")
    ap.add_argument("file")
    ap.add_argument("--expect-size", default=None)
    ap.add_argument("--expect-factor", type=float, default=None)
    ap.add_argument("--meta", default=None)
    args = ap.parse_args()

    p = Path(args.file)
    if not p.is_file():
        print(f"[错误] 找不到: {p}")
        return 2

    im = Image.open(p)
    ok = True

    print(f"文件   : {p}")
    print(f"mode   : {im.mode}")
    print(f"size   : {im.size}")
    print(f"format : {im.format}")

    if im.mode != "RGBA":
        print("  [失败] mode 不是 RGBA -> 没有透明通道")
        ok = False
    else:
        print("  [通过] RGBA, 带透明通道")

    arr = np.asarray(im.convert("RGBA"))
    a = arr[:, :, 3]
    print(f"\nalpha  min={a.min()}  max={a.max()}  "
          f"全透明={int((a == 0).sum())}  半透明={int(((a > 0) & (a < 255)).sum())}  "
          f"不透明={int((a == 255).sum())}")

    if a.min() != 0:
        print("  [失败] 没有任何全透明像素 -> 底没抠掉")
        ok = False
    else:
        corners = [int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1])]
        print(f"  四角 alpha: {corners}")

    if args.expect_size:
        want = parse_size(args.expect_size)
        if im.size != want:
            print(f"  [失败] 尺寸不符: 实际 {im.size} != 期望 {want}")
            ok = False
        else:
            print(f"  [通过] 尺寸 == {want}")

    if args.expect_factor and args.meta:
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
        want = (int(round(meta["width"] * args.expect_factor)),
                int(round(meta["height"] * args.expect_factor)))
        if im.size != want:
            print(f"  [失败] 与源图 x{args.expect_factor} 不符: 实际 {im.size} != 期望 {want}")
            ok = False
        else:
            print(f"  [通过] 尺寸 == 源图 x{args.expect_factor} = {want}")

    print("\n结论:", "全部通过" if ok else "有问题, 见上面 [失败] 行")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
