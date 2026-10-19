"""独立版推理 CLI —— `run_worker.py` 的替代品，不依赖 VOSR2Tool。

手工流程里用它跑推理，参数与旧的 run_worker.py 保持一致，
但后端换成 vosr2_standalone（纯 torch，无编译模块）。

用法:
    python run_standalone.py -c <权重根目录> -i <输入目录> -o <输出目录> [-u 4]

例子:
    python run_standalone.py -c D:\\vosr2\\models -i work\\full\\in -o work\\full\\out -u 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import vosr2_standalone as V


def main() -> int:
    ap = argparse.ArgumentParser(description="VOSR 2 standalone inference (no VOSR2Tool)")
    ap.add_argument("-c", "--checkpoint", required=True,
                    help="权重根目录（含 VOSR2/、Qwen-Image-vae-2d/、torch_cache/）")
    ap.add_argument("-i", "--input_image", required=True, help="输入目录（不是文件）")
    ap.add_argument("-o", "--output_dir", required=True, help="输出目录")
    ap.add_argument("-u", "--upscale", type=int, default=4, choices=(1, 2, 3, 4))
    ap.add_argument("--tile_size", type=int, default=512,
                    help="DiT 分块（0 = 不分块）")
    ap.add_argument("--tile_overlap", type=int, default=64)
    ap.add_argument("--vae_tile_size", type=int, default=1024,
                    help="VAE 分块（0 在大图上会 OOM）")
    ap.add_argument("--vae_tile_overlap", type=int, default=128)
    ap.add_argument("--align_method", default="wavelet",
                    choices=("wavelet", "adain", "nofix"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    ap.add_argument("--noise_scale", type=float, default=1.0,
                    help="潜空间噪声尺度。1.0 = 官方契约（默认），不要轻易改")
    args = ap.parse_args()

    inp, out = Path(args.input_image), Path(args.output_dir)
    if not inp.is_dir():
        print(f"[错误] 输入目录不存在：{inp}（-i 要传目录，不是文件）")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    print(f"  权重    : {args.checkpoint}")
    print(f"  输入    : {inp}")
    print(f"  输出    : {out}")
    print(f"  倍数    : x{args.upscale}   DiT分块={args.tile_size}  "
          f"VAE分块={args.vae_tile_size}  噪声尺度={args.noise_scale}")

    t0 = time.time()

    def show(stage: str, done: int, total: int) -> None:
        if stage in ("vae_encode", "dino", "dit", "vae_decode") and total > 1:
            print(f"\r    {stage:<11} {done}/{total}", end="", flush=True)
            if done >= total:
                print(flush=True)

    try:
        res = V.upscale_batch(
            inp, out,
            upscale=args.upscale,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            vae_tile_size=args.vae_tile_size,
            vae_tile_overlap=args.vae_tile_overlap,
            seed=args.seed,
            color_alignment=args.align_method,
            dtype=args.dtype,
            models_root=args.checkpoint,
            noise_scale=args.noise_scale,
            progress=show,
        )
    except V.VOSR2LoadError as exc:
        print(f"\n[错误] 加载失败：{exc}")
        return 3

    print(f"\n  用时 {time.time() - t0:.1f} 秒")
    ok = res.get("ok") or []
    bad = res.get("failed") or {}
    print(f"  成功 {len(ok)} 张，失败 {len(bad)} 张")
    for k, v in bad.items():
        print(f"    [失败] {k}: {v}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
