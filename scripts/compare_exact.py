"""逐像素比对两张 PNG 是否完全一致（用于回归验证）。

用法:
    python.exe compare_exact.py <A.png> <B.png>
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    pa, pb = sys.argv[1], sys.argv[2]

    ia, ib = Image.open(pa), Image.open(pb)
    print(f"A: {pa}")
    print(f"   mode={ia.mode} size={ia.size}")
    print(f"B: {pb}")
    print(f"   mode={ib.mode} size={ib.size}")

    ok = True
    if ia.mode != ib.mode:
        print(f"  [差异] mode 不同: {ia.mode} vs {ib.mode}")
        ok = False
    if ia.size != ib.size:
        print(f"  [差异] size 不同")
        return 1

    a = np.asarray(ia.convert("RGBA"))
    b = np.asarray(ib.convert("RGBA"))

    print(f"\nRGBA 整体 sha256:")
    da, db = digest(a), digest(b)
    print(f"   A = {da}")
    print(f"   B = {db}")
    same = da == db

    if same:
        print("\n  [通过] 两图 RGBA 逐像素完全一致")
    else:
        print("\n  [差异] 存在不一致，逐通道分析：")
        names = ("R", "G", "B", "A")
        for i, n in enumerate(names):
            x, y = a[:, :, i].astype(np.int16), b[:, :, i].astype(np.int16)
            d = np.abs(x - y)
            nz = int((d > 0).sum())
            print(f"   {n}: 最大差 {int(d.max()):3d}  不同像素 {nz}  ({nz / d.size * 100:.4f}%)")
        ok = False

    print(f"\n结论: {'完全一致' if same else '存在差异'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
