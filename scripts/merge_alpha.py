"""把放大后的 RGB 和放大后的 alpha 合回一张透明 PNG。

用法:
    python.exe merge_alpha.py <rgb_x4.png> <alpha_x4.png> <out.png>
                              [--meta meta.json] [--resize-alpha]

默认要求两张图尺寸完全一致, 不一致直接报错(多半是模型输出被裁剪了)。
--resize-alpha 允许把蒙版插值到 RGB 的尺寸后继续。
--meta 会从 split 阶段留下的 meta.json 里恢复 dpi / xmp 等 PNG 元数据。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="合成透明 PNG")
    ap.add_argument("rgb")
    ap.add_argument("alpha")
    ap.add_argument("out")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--resize-alpha", action="store_true")
    args = ap.parse_args()

    rgb_path, a_path = Path(args.rgb), Path(args.alpha)
    for p in (rgb_path, a_path):
        if not p.is_file():
            print(f"[错误] 找不到: {p}")
            return 2

    rgb = Image.open(rgb_path).convert("RGB")
    alpha = Image.open(a_path).convert("L")

    if rgb.size != alpha.size:
        if not args.resize_alpha:
            print(f"[错误] 尺寸不一致: RGB={rgb.size}  alpha={alpha.size}")
            print("      模型输出可能被裁剪/对齐过。先确认倍数, 或用 --resize-alpha 强制对齐。")
            return 3
        print(f"[警告] 尺寸不一致 RGB={rgb.size} alpha={alpha.size}, 按 RGB 尺寸插值蒙版")
        alpha = alpha.resize(rgb.size, Image.LANCZOS)

    out = rgb.convert("RGBA")
    out.putalpha(alpha)

    pnginfo = None
    if args.meta and Path(args.meta).is_file():
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
        info = meta.get("png_info") or {}
        pnginfo = PngImagePlugin.PngInfo()
        if "XML:com.adobe.xmp" in info:
            pnginfo.add_text("XML:com.adobe.xmp", str(info["XML:com.adobe.xmp"]))
        if "xmp" in info and "XML:com.adobe.xmp" not in info:
            pnginfo.add_text("XML:com.adobe.xmp", str(info["xmp"]))
        dpi = info.get("dpi")
        if isinstance(dpi, (list, tuple)) and len(dpi) == 2:
            out.info["dpi"] = tuple(float(x) for x in dpi)

    # 复核: 写出去之前确认 alpha 真的挂上了
    chk = np.asarray(out)[:, :, 3]
    if chk.max() == 255 and chk.min() == 0:
        pass
    else:
        print(f"[提醒] alpha 范围异常: min={chk.min()} max={chk.max()} "
              f"(若源蒙版本就是全不透明, 可忽略)")

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"format": "PNG"}
    if pnginfo is not None:
        kwargs["pnginfo"] = pnginfo
    out.save(dst, **kwargs)

    print(f"RGB    : {rgb_path}  {rgb.size}")
    print(f"alpha  : {a_path}  {alpha.size}")
    print(f"输出   : {dst}   mode={out.mode}")
    print(f"alpha  : min={chk.min()}  max={chk.max()}  "
          f"全透明={int((chk == 0).sum())}  半透明={int(((chk > 0) & (chk < 255)).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
