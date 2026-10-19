"""把 RGBA PNG 拆成「模型能吃的 RGB」+「自己保管的 alpha」。

用法:
    python.exe split_alpha.py <src.png> <out_dir> [--tag NAME]
                             [--fill blur|flat|none] [--radius 48]

产出:
    <out_dir>/<tag>.rgb.png     三通道; 完全透明区已按 --fill 填色
    <out_dir>/<tag>.alpha.png   8-bit 灰度蒙版, 原样保存
    <out_dir>/<tag>.meta.json   源尺寸 / 填色参数 / 需要保留的 PNG 元数据

只读 src.png, 只写 out_dir。半透明像素的 RGB 一律不动(已确认是直通 alpha)。

为什么只填 alpha==0 的区域:
    VOSR2 在喂模型前是 img.convert("RGB"), alpha 直接丢掉。完全透明区
    露出来的 RGB 就是一张近白的背景残影, 模型会把它当真实背景 -> 白边 + 彩带。
    把这块换成"主体自己的边缘色", 模型看到的边界就是连续的, halo 基本消失。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def opaque_median(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
    """不透明像素的中位色, 作为"扩散不到的地方"的兜底色。"""
    op = rgb[a == 255]
    if len(op) == 0:
        op = rgb[a > 0]
    if len(op) == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)
    return np.median(op, axis=0).astype(np.float32)


def blur_extend(rgb: np.ndarray, a: np.ndarray, radius: int, fallback: np.ndarray) -> np.ndarray:
    """归一化模糊外推: ext = blur(rgb * known) / blur(known)。

    在紧贴主体的几十像素内, 这等价于"用四周真实颜色平滑地补出去";
    远处 known 模糊到接近 0, 就退回兜底纯色(反正那里会被 alpha 切掉)。
    """
    known = (a > 0).astype(np.float32)
    premul = rgb.astype(np.float32) * known[..., None]

    k_img = Image.fromarray((known * 255.0).round().astype(np.uint8), "L")
    kb = np.asarray(k_img.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32) / 255.0

    p_img = Image.fromarray(np.clip(premul, 0, 255).round().astype(np.uint8), "RGB")
    pb = np.asarray(p_img.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32)

    ext = pb / np.maximum(kb, 1e-6)[..., None]
    conf = np.clip(kb / 0.05, 0.0, 1.0)[..., None]
    return ext * conf + fallback.reshape(1, 1, 3) * (1.0 - conf)


def main() -> int:
    ap = argparse.ArgumentParser(description="拆分 RGBA PNG 为 RGB + alpha")
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--tag", default=None, help="输出文件名前缀, 默认取源文件名主干")
    ap.add_argument("--fill", choices=("blur", "flat", "none"), default="blur",
                    help="完全透明区的填色方式 (默认 blur)")
    ap.add_argument("--radius", type=int, default=48, help="blur 模式的扩散半径 (像素)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_file():
        print(f"[错误] 找不到源文件: {src}")
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or src.stem

    im = Image.open(src)
    info = {k: v for k, v in im.info.items() if isinstance(v, (str, int, float))}
    rgba = np.asarray(im.convert("RGBA"))
    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3].copy()
    h, w = alpha.shape

    n_zero = int((alpha == 0).sum())
    n_semi = int((((alpha > 0) & (alpha < 255))).sum())
    n_opaque = int((alpha == 255).sum())

    fallback = opaque_median(rgb, alpha)
    if args.fill != "none" and n_zero:
        if args.fill == "flat":
            fill = np.broadcast_to(fallback, (h, w, 3)).astype(np.float32)
        else:
            fill = blur_extend(rgb, alpha, args.radius, fallback)
        fill = np.clip(fill, 0, 255).round().astype(np.uint8)
        rgb[alpha == 0] = fill[alpha == 0]

    rgb_path = out_dir / f"{tag}.rgb.png"
    alpha_path = out_dir / f"{tag}.alpha.png"
    meta_path = out_dir / f"{tag}.meta.json"

    Image.fromarray(rgb, "RGB").save(rgb_path, format="PNG", optimize=False)
    Image.fromarray(alpha, "L").save(alpha_path, format="PNG", optimize=False)

    meta = {
        "src": str(src.resolve()),
        "tag": tag,
        "width": w,
        "height": h,
        "fill_mode": args.fill,
        "fill_radius": args.radius,
        "fallback_color": [float(x) for x in fallback],
        "alpha_zero_px": n_zero,
        "alpha_semi_px": n_semi,
        "alpha_opaque_px": n_opaque,
        "png_info": info,
        "rgb_file": rgb_path.name,
        "alpha_file": alpha_path.name,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"源文件        : {src}")
    print(f"尺寸          : {w} x {h}")
    print(f"完全透明      : {n_zero:>9d}  ({n_zero / alpha.size * 100:5.2f}%)")
    print(f"半透明        : {n_semi:>9d}  ({n_semi / alpha.size * 100:5.2f}%)")
    print(f"完全不透明    : {n_opaque:>9d}  ({n_opaque / alpha.size * 100:5.2f}%)")
    print(f"填色模式      : {args.fill}   radius={args.radius}")
    print(f"兜底底色      : {[int(x) for x in fallback]}")
    print("-" * 60)
    print(f"RGB   -> {rgb_path}")
    print(f"alpha -> {alpha_path}")
    print(f"meta  -> {meta_path}")
    print("=" * 60)
    print("下一步: 只把 .rgb.png 交给推理 (run_standalone.py 或 一键放大.bat); .alpha.png 自己保管, 不要过模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
