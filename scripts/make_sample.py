"""从全尺寸拆分产物生成一套小样, 用于先跑通链路再上全尺寸。

用法:
    python.exe make_sample.py <rgb.png> <alpha.png> <out_dir> [--width 600] [--tag sample]

产出:
    <out_dir>/<tag>.rgb.png     缩小后的 RGB (交给模型)
    <out_dir>/<tag>.alpha.png   缩小后的 alpha (自己保管)
    <out_dir>/<tag>.meta.json   记录小样尺寸, 供 verify_png.py --expect-factor 用
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="生成小样")
    ap.add_argument("rgb")
    ap.add_argument("alpha")
    ap.add_argument("out_dir")
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--tag", default="sample")
    args = ap.parse_args()

    rgb_im = Image.open(args.rgb).convert("RGB")
    a_im = Image.open(args.alpha).convert("L")
    if rgb_im.size != a_im.size:
        print(f"[错误] 尺寸不一致: {rgb_im.size} vs {a_im.size}")
        return 2

    w0, h0 = rgb_im.size
    w = args.width
    h = max(1, int(round(h0 * w / w0)))

    rgb_s = rgb_im.resize((w, h), Image.LANCZOS)
    a_s = a_im.resize((w, h), Image.LANCZOS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_p = out_dir / f"{args.tag}.rgb.png"
    a_p = out_dir / f"{args.tag}.alpha.png"
    m_p = out_dir / f"{args.tag}.meta.json"

    rgb_s.save(rgb_p, format="PNG")
    a_s.save(a_p, format="PNG")
    m_p.write_text(json.dumps({
        "tag": args.tag,
        "width": w,
        "height": h,
        "source_width": w0,
        "source_height": h0,
        "scale_from_source": round(w / w0, 6),
        "rgb_file": rgb_p.name,
        "alpha_file": a_p.name,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"源     : {w0} x {h0}")
    print(f"小样   : {w} x {h}   (缩放系数 {w / w0:.4f})")
    print(f"RGB    : {rgb_p}")
    print(f"alpha  : {a_p}")
    print(f"meta   : {m_p}")
    print(f"\n预期模型输出 (x4): {w * 4} x {h * 4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
