@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1

rem ===========================================================================
rem  VOSR2 one-click transparent PNG upscaler -- standalone build
rem
rem  * No dependency on VOSR2Tool: the VOSR 2 inference code is embedded here
rem  * Only needs a Python with torch / numpy / Pillow / safetensors
rem  * Drop this .bat into a folder full of transparent PNGs and double-click
rem
rem  Usage:
rem      this.bat                process the current folder (double-click)
rem      this.bat --list         dry run, show what would be processed
rem      this.bat -u 2           upscale 2x instead of 4x
rem      this.bat --force        re-run even if outputs already exist
rem      this.bat --dump         export the embedded Python sources
rem
rem  Settings live in the lower half of this file (search for "CFG = dict").
rem
rem  NOTE: keep everything above the #PYSTART marker pure ASCII.  cmd.exe
rem  re-reads the batch file with the new code page after `chcp`, and any
rem  multi-byte character up here gets the following line mis-parsed and
rem  executed as a command.  Chinese notes belong inside the Python payload.
rem ===========================================================================

set "SELF=%~f0"

rem ---- 1. locate a Python interpreter ----
rem  `where python` often returns the Microsoft Store stub (a 0-byte
rem  forwarder) which fails with 9009.  So every candidate is actually run.
rem  Prefer one that can `import torch`; fall back to any working one.
rem  Order: .venv, python\, VOSR2_PYTHON, vosr2_env.txt, PATH, VOSR2Tool
set "PYEXE="
set "PYFALLBACK="

call :try_py "%~dp0.venv\Scripts\python.exe"
call :try_py "%~dp0python\python.exe"
call :try_py "%~dp0..\python\python.exe"
call :try_py "%~dp0..\..\python\python.exe"
call :try_py "%VOSR2_PYTHON%"
if not defined PYEXE if exist "%~dp0vosr2_env.txt" (
    for /f "usebackq delims=" %%L in ("%~dp0vosr2_env.txt") do call :try_py "%%L"
)
if not defined PYFALLBACK if not defined PYEXE (
    for /f "delims=" %%P in ('where python 2^>nul') do call :try_py "%%P"
    for /f "delims=" %%P in ('where python3 2^>nul') do call :try_py "%%P"
)
call :try_py "E:\VOSR2Tool\VOSR2Tool\_internal\python\python.exe"
call :try_py "C:\Python312\python.exe"
call :try_py "C:\Python311\python.exe"

if not defined PYEXE if defined PYFALLBACK (
    set "PYEXE=%PYFALLBACK%"
    echo.
    echo  [WARN] The selected Python has no torch:
    echo         %PYEXE%
    echo         Install one with the bundled environment installer if this fails.
    echo.
)

if not defined PYEXE (
    echo.
    echo  [ERROR] No usable Python found.
    echo.
    echo  Tried:  .venv  /  python\  /  VOSR2_PYTHON  /  vosr2_env.txt  /  PATH  /  E:\VOSR2Tool
    echo.
    echo  Fix it in ONE of these ways:
    echo    1^) Run the environment installer .bat that ships next to this file.
    echo       It creates a  python\  folder right here, found automatically.
    echo       Or put a file named  vosr2_env.txt  next to this .bat containing
    echo       one line - the full path to python.exe, e.g.
    echo           D:\vosr2\python\python.exe
    echo    2^) Set the environment variable VOSR2_PYTHON
    echo    3^) Make sure a working python is on PATH
    echo.
    echo  NOTE: the Microsoft Store stub in WindowsApps is not a real Python.
    echo.
    pause
    exit /b 1
)

echo  [i] Python : %PYEXE%

rem ---- 2. extract the two embedded Python payloads to a temp folder ----
set "TMPDIR=%TEMP%\vosr2_oc_%RANDOM%%RANDOM%"
mkdir "%TMPDIR%" 2>nul
set "DRIVER=%TMPDIR%\driver.py"
set "MODULE=%TMPDIR%\vosr2_standalone.py"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$l = Get-Content -LiteralPath $env:SELF -Encoding UTF8; $i = [Array]::IndexOf($l, '#PYSTART'); $j = [Array]::IndexOf($l, '#MODSTART'); if ($i -lt 0 -or $j -lt 0) { exit 2 }; $enc = New-Object System.Text.UTF8Encoding($false); [System.IO.File]::WriteAllLines($env:DRIVER, $l[($i + 1)..($j - 1)], $enc); [System.IO.File]::WriteAllLines($env:MODULE, $l[($j + 1)..($l.Count - 1)], $enc)"

if not exist "%DRIVER%" (
    echo.
    echo  [ERROR] Failed to extract the embedded Python payload.
    echo          Make sure this file still has CRLF line endings and the
    echo          '#PYSTART' / '#MODSTART' marker lines are intact.
    echo.
    pause
    exit /b 1
)
if not exist "%MODULE%" (
    echo  [ERROR] Embedded VOSR2 module missing.
    pause
    exit /b 1
)

rem ---- 3. --dump: export the sources only, do not run ----
if /i "%~1"=="--dump" (
    copy /y "%DRIVER%" "%~dp0_vosr2_driver.py" >nul
    copy /y "%MODULE%" "%~dp0_vosr2_standalone.py" >nul
    echo  Dumped to:
    echo     %~dp0_vosr2_driver.py
    echo     %~dp0_vosr2_standalone.py
    rmdir /s /q "%TMPDIR%" >nul 2>&1
    exit /b 0
)

rem ---- 4. run ----
rem  Must be "%~dp0." and not "%~dp0": the latter ends with a backslash, and
rem  the Windows CRT turns the following `"` into an escaped quote, which
rem  swallows the rest of the arguments.
rem  No arguments at all (a plain double-click) -> start the browser UI.
rem  Any argument (-y, -u 2, --list, ...) -> the CLI, exactly as before.
set "PYARGS=%*"
if not defined PYARGS set "PYARGS=--gui"
"%PYEXE%" "%DRIVER%" --bat-dir "%~dp0." %PYARGS%
set "RC=%ERRORLEVEL%"
rmdir /s /q "%TMPDIR%" >nul 2>&1

echo.
if "%RC%"=="0" (echo  [OK] Done.) else (echo  [FAILED] exit code = %RC%)
pause
exit /b %RC%


rem ===========================================================================
rem  Subroutine: validate a candidate Python
rem    can `import torch`  -> adopt it immediately
rem    can only `import sys` -> keep as fallback, keep looking
rem  This is what skips the Microsoft Store 0-byte stub.
rem ===========================================================================
:try_py
if "%~1"=="" exit /b 0
if not exist "%~1" exit /b 0
if defined PYEXE exit /b 0

"%~1" -c "import torch" >nul 2>&1
if not errorlevel 1 (
    set "PYEXE=%~1"
    set "PYHAS_TORCH=1"
    exit /b 0
)

if not defined PYFALLBACK (
    "%~1" -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYFALLBACK=%~1"
)
exit /b 0

#PYSTART
# -*- coding: utf-8 -*-
r"""
VOSR2 透明底 PNG 一键放大  —— 内嵌 Python 主体

原理
----
VOSR2Tool 在代码层面就丢弃 alpha（推理前 img.convert("RGB")，保存时也明写
"不支持透明，先合成到白底"）。所以唯一出路是把 alpha 拆到工具外面自己管：

    RGBA --拆分--> [填色后的 RGB]  --模型-->  [放大后的 RGB]  --合成--> RGBA
                      [alpha]      --LANCZOS-->  [放大后的 alpha]

透明区不能留白（模型会把白背景当真实背景 -> 白边 + 彩带），
要用主体自己的边缘色平滑外推填上。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image, ImageFilter, PngImagePlugin

# 内嵌的 VOSR 2 独立实现（由 #MODSTART 段落释放到同目录）
#
# 注意：官方 embedded Python 若带 python311._pth，会进入隔离模式，
# 不再自动把「脚本所在目录」放进 sys.path（它是 python311.zip）。
# 所以这里必须显式插入一次，否则同目录的 vosr2_standalone.py 会 import 不到。
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 图形界面需要（embeddable Python 里都是标准库，无需额外安装）
import threading
import gc
import os

V_IMPORT_ERROR = None
try:
    import vosr2_standalone as V
except BaseException as _e:
    V = None
    V_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"


# ===========================================================================
#  配置区  ——  双击运行时就用这里的默认值；也可以在命令行覆盖，例如
#              一键放大.bat -u 2
#              一键放大.bat --force
# ===========================================================================
# ===========================================================================
#  配置区  ——  双击运行时用这里的默认值；也可以用命令行覆盖，例如
#              一键放大.bat -u 2
#              一键放大.bat --force
# ===========================================================================
CFG = dict(
    # ---- 文件夹（相对本 .bat 所在目录）----
    #   input_dir  图片放这里。留空 = 直接用本 .bat 所在的文件夹。
    #              设了但该文件夹不存在时，会自动退回本 .bat 所在文件夹，
    #              所以把 bat 单独复制到别处用也不会出问题。
    #   output_dir 结果放这里。留空 = 放在图片文件夹里的 vosr2_output。
    input_dir        = "vosr2_input",
    output_dir       = "vosr2_output",

    upscale          = 4,          # 放大倍数
    tile_overlap     = 64,         # DiT 分块重叠
    vae_tile_overlap = 128,        # VAE 分块重叠
    dtype            = "auto",     # auto / bf16 / fp16 / fp32

    # ★★ 分块大小怎么定 —— low_vram 决定取值方式
    #    "auto"  = 自动检测显存容量，按下表选（推荐，默认）
    #    True    = 强制小分块（最省显存，适合 8GB 及以下）
    #    False   = 用下面手写的 tile_size / vae_tile
    #
    #    auto 的对照表（显存总量 -> DiT 分块 / VAE 分块）：
    #        ≤ 6 GB   ->  256 / 384
    #        ≤ 9 GB   ->  384 / 512     ← 8GB 卡走这一档
    #        其它     ->  512 / 1024    ← 12GB 及以上
    #
    #    要手动覆盖，就把它改成 False，再改下面的两个数字：
    #        tile_size = 256, vae_tile = 384    （最省，但更慢）
    low_vram         = "auto",
    tile_size        = 512,        # DiT 分块（low_vram=False 时才生效）
    vae_tile         = 1024,       # VAE 分块（low_vram=False 时才生效）

    # ★ 潜空间噪声尺度 —— 1.0 是官方契约，也是 VOSR2Tool 的真实行为
    #   官方 loader.py:  return z - u        z 为单位方差噪声
    #   探针实测工具引擎：同样生成单位方差 z（seed 42），逐位相同
    #   注：早期移植里 Attention 有个笔误 self.k_norm(q)（应为 k），
    #      导致 k 未归一化、DiT 输出错误；那时把这里设成 0 只是掩盖症状。
    #      笔误已修复，36 个 block 与工具逐位一致，故此处恢复为 1.0。
    noise_scale      = 1.0,

    align            = "wavelet",  # 色彩对齐：wavelet / adain / nofix

    # 输出格式：png / webp / jpeg / same（same = 与原图格式相同）
    #   png  —— 无损，支持透明（默认）
    #   webp —— 有损，支持透明，体积通常只有 PNG 的 1/5
    #   jpeg —— 有损，不支持透明；透明区域会铺成 jpeg_bg 的底色
    out_format       = "png",
    jpeg_quality     = 95,         # 1-100
    webp_quality     = 95,         # 1-100
    jpeg_bg          = "white",    # jpeg 底色：white / black
    seed             = 42,         # 随机种子

    fill             = "blur",     # 透明区填色：blur（推荐）/ flat / none
    fill_radius      = 48,         # blur 模式的扩散半径（像素）
    mask_mode        = "lanczos",  # 蒙版插值：lanczos（推荐）/ bicubic / nearest
    mask_blur        = 2.0,        # 蒙版放大后补的羽化半径
    mask_erode       = 0,          # 蒙版向内收缩像素数，用来切掉浅色描边

    skip_exist       = True,       # 输出已存在就跳过（可重复运行）
    keep_temp        = False,      # 保留中间文件（排查问题时打开）
    preview          = True,       # 生成预览图

    # 图形界面跑完一张任务后是否保留模型。
    #   False（默认）—— 跑完立刻释放约 4 GB 内存；下次点开始要重新载入（约 10 秒）
    #   True          —— 常驻，连续跑更快，但内存会一直占着直到关掉窗口
    keep_loaded      = False,

    # ★ 每次启动先清掉上次的派生文件：_预览\、_日志.txt、_工作文件\
    #   已经输出好的 _xN.png 成品一个都不动。
    #   如果 _日志.txt 被记事本等程序占着删不掉，会自动跳过（稍后覆盖）。
    clean_output     = True,

    # ★ 启动前清理其它程序占着的内存（同各类启动器的「内存清理」按钮）
    #   原理：对其它进程调用 EmptyWorkingSet，把它们暂时用不到的物理页挤到备用列表。
    #   代价：切回那些程序时会稍微卡一下；数据不会丢。系统进程会被跳过。
    trim_memory      = True,
)
# ===========================================================================

# ===========================================================================


OUT_DIR_NAME = "vosr2_output"
WORK_DIR_NAME = "_工作文件"
PREVIEW_DIR_NAME = "_预览"
LOG_NAME = "_日志.txt"

# 扫描时要排除的自家产物后缀
EXCLUDE_SUFFIXES = ("_x4.png", "_x3.png", "_x2.png", "_x1.png",
                    ".rgb.png", ".alpha.png", "_preview.png")


# ---------------------------------------------------------------------------
#  权重定位
# ---------------------------------------------------------------------------
# 权重根目录下应有这 5 个文件（与 VOSR2Tool 的 ckpts 布局一致，可直接拷贝）：
#     VOSR2/args.json
#     VOSR2/checkpoints/ema_model.safetensors
#     Qwen-Image-vae-2d/config.json
#     Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors
#     torch_cache/checkpoints/dinov2_vitl14_pretrain.pth
REQUIRED_REL = (
    ("VOSR2", "args.json"),
    ("VOSR2", "checkpoints", "ema_model.safetensors"),
    ("Qwen-Image-vae-2d", "config.json"),
    ("Qwen-Image-vae-2d", "diffusion_pytorch_model.safetensors"),
    ("torch_cache", "checkpoints", "dinov2_vitl14_pretrain.pth"),
)


def required_files(root: Path) -> list:
    return [root.joinpath(*parts) for parts in REQUIRED_REL]


def find_models_root(bat_dir: Path):
    """按 5 级优先级找权重根目录。"""
    import os
    cands = []

    env = os.environ.get("VOSR2_MODELS")
    if env:
        cands.append(Path(env))

    txt = bat_dir / "vosr2_models.txt"
    if txt.is_file():
        try:
            line = txt.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            if line:
                p = Path(line)
                cands.append(p if p.is_absolute() else (bat_dir / p))
        except Exception:
            pass

    cands.append(bat_dir / "models")
    cands.append(bat_dir / "ckpts")
    # 上一层：把「一键放大.bat」放在图片子文件夹里、而 python/ 和 models/ 在
    # 部署根目录时，这种布局才找得到（打整个文件夹拷到别的电脑就是这种情况）。
    cands.append(bat_dir.parent / "models")
    cands.append(bat_dir.parent / "ckpts")
    cands.append(Path(r"E:\VOSR2Tool\VOSR2Tool\models\preset\ckpts"))

    for c in cands:
        try:
            if (c / "VOSR2" / "args.json").is_file():
                return c
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
#  图像处理
# ---------------------------------------------------------------------------
def load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def alpha_info(arr: np.ndarray) -> tuple[bool, int, int]:
    a = arr[:, :, 3]
    n_zero = int((a == 0).sum())
    n_semi = int(((a > 0) & (a < 255)).sum())
    return (n_zero > 0 or n_semi > 0), n_zero, n_semi


def opaque_median(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
    op = rgb[a == 255]
    if len(op) == 0:
        op = rgb[a > 0]
    if len(op) == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)
    return np.median(op, axis=0).astype(np.float32)


def blur_extend(rgb: np.ndarray, a: np.ndarray, radius: int,
                fallback: np.ndarray) -> np.ndarray:
    """归一化模糊外推：ext = blur(rgb*known) / blur(known)"""
    known = (a > 0).astype(np.float32)
    premul = rgb.astype(np.float32) * known[..., None]

    k = Image.fromarray((known * 255.0).round().astype(np.uint8), "L")
    kb = np.asarray(k.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32) / 255.0

    p = Image.fromarray(np.clip(premul, 0, 255).round().astype(np.uint8), "RGB")
    pb = np.asarray(p.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32)

    ext = pb / np.maximum(kb, 1e-6)[..., None]
    conf = np.clip(kb / 0.05, 0.0, 1.0)[..., None]
    return ext * conf + fallback.reshape(1, 1, 3) * (1.0 - conf)


def split_one(src: Path, rgb_dst: Path, alpha_dst: Path, cfg: dict) -> dict:
    """把一张 RGBA 拆成 填色RGB + 独立 alpha，返回统计信息。"""
    im = Image.open(src)
    keep_info = {k: v for k, v in im.info.items() if isinstance(v, (str, int, float))}
    arr = np.asarray(im.convert("RGBA"))
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3].copy()
    h, w = alpha.shape

    n_zero = int((alpha == 0).sum())
    n_semi = int((((alpha > 0) & (alpha < 255))).sum())

    fallback = opaque_median(rgb, alpha)
    if cfg["fill"] != "none" and n_zero:
        if cfg["fill"] == "flat":
            fill = np.broadcast_to(fallback, (h, w, 3)).astype(np.float32)
        else:
            fill = blur_extend(rgb, alpha, cfg["fill_radius"], fallback)
        fill = np.clip(fill, 0, 255).round().astype(np.uint8)
        rgb[alpha == 0] = fill[alpha == 0]

    rgb_dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(rgb_dst, format="PNG")
    Image.fromarray(alpha, "L").save(alpha_dst, format="PNG")

    return {
        "src": str(src), "width": w, "height": h,
        "alpha_zero": n_zero, "alpha_semi": n_semi,
        "fallback_color": [float(x) for x in fallback],
        "png_info": keep_info,
    }


def make_alpha(out_dir: Path, alpha_src: Path, target: tuple[int, int],
               cfg: dict) -> Path:
    im = Image.open(alpha_src).convert("L")
    mode = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST}.get(cfg["mask_mode"], Image.LANCZOS)
    out = im.resize(target, mode)
    if cfg["mask_blur"] and cfg["mask_blur"] > 0:
        out = out.filter(ImageFilter.GaussianBlur(cfg["mask_blur"]))
    if cfg["mask_erode"] and cfg["mask_erode"] > 0:
        out = out.filter(ImageFilter.MinFilter(cfg["mask_erode"] * 2 + 1))
    dst = out_dir / f"{alpha_src.stem}_x{cfg['upscale']}.alpha.png"
    out.save(dst, format="PNG")
    return dst


def merge_one(rgb_path: Path, alpha_path: Path, dst: Path, meta: dict) -> tuple:
    rgb = Image.open(rgb_path).convert("RGB")
    alpha = Image.open(alpha_path).convert("L")
    if rgb.size != alpha.size:
        alpha = alpha.resize(rgb.size, Image.LANCZOS)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)

    info = meta.get("png_info") or {}
    pnginfo = PngImagePlugin.PngInfo()
    xmp = info.get("XML:com.adobe.xmp") or info.get("xmp")
    if xmp:
        pnginfo.add_text("XML:com.adobe.xmp", str(xmp))
    dpi = info.get("dpi")
    if isinstance(dpi, (list, tuple)) and len(dpi) == 2:
        out.info["dpi"] = tuple(float(x) for x in dpi)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, format="PNG", pnginfo=pnginfo)
    chk = np.asarray(out)[:, :, 3]
    return dst, int(chk.min()), int(chk.max()), int((chk == 0).sum())


def checkerboard(w: int, h: int, cell: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((yy // cell) + (xx // cell)) % 2
    v = np.where(m == 0, 214, 168).astype(np.uint8)
    return np.repeat(v[:, :, None], 3, axis=2)


def over(rgba: np.ndarray, bg: np.ndarray) -> np.ndarray:
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return (rgba[:, :, :3].astype(np.float32) * a
            + bg.astype(np.float32) * (1 - a)).round().clip(0, 255).astype(np.uint8)


def make_preview(final: Path, dst_dir: Path, width: int = 620, crop: int = 700) -> Path | None:
    im = Image.open(final).convert("RGBA")
    arr = np.asarray(im)
    H, W = arr.shape[:2]
    dst_dir.mkdir(parents=True, exist_ok=True)

    w = min(width, W)
    h = max(1, int(round(H * w / W)))
    small = np.asarray(im.resize((w, h), Image.LANCZOS))
    sep = np.full((h, 3, 3), 128, np.uint8)
    canvas = np.concatenate([
        over(small, checkerboard(w, h)), sep,
        over(small, np.zeros((h, w, 3), np.uint8)), sep,
        over(small, np.full((h, w, 3), 255, np.uint8)),
    ], axis=1)

    a = arr[:, :, 3].astype(np.int16)
    if a.min() < a.max():
        gy = np.abs(np.diff(a, axis=0))[:-1, :]
        gx = np.abs(np.diff(a, axis=1))[:, :-1]
        n = min(gy.shape[0], gx.shape[0]), min(gy.shape[1], gx.shape[1])
        g = gy[:n[0], :n[1]] + gx[:n[0], :n[1]]
        cy, cx = np.unravel_index(int(np.argmax(g)), g.shape)
        c = min(crop, H, W)
        y0 = int(np.clip(cy - c // 2, 0, max(0, H - c)))
        x0 = int(np.clip(cx - c // 2, 0, max(0, W - c)))
        cr = arr[y0:y0 + c, x0:x0 + c]
        ch, cw = cr.shape[:2]
        canvas2 = np.concatenate([
            over(cr, checkerboard(cw, ch)), np.full((ch, 3, 3), 128, np.uint8),
            over(cr, np.zeros((ch, cw, 3), np.uint8)),
        ], axis=1)
        Image.fromarray(canvas2, "RGB").save(
            dst_dir / f"{final.stem}_预览_边缘.png", format="PNG")

    p = dst_dir / f"{final.stem}_预览.png"
    Image.fromarray(canvas, "RGB").save(p, format="PNG")
    return p


# ---------------------------------------------------------------------------
#  干跑模式
# ---------------------------------------------------------------------------
#  干跑模式
# ---------------------------------------------------------------------------
def print_list(folder: Path, out_root: Path, candidates: list, excluded: list,
               cfg: dict, args) -> int:
    """只报告会处理哪些图。不加载模型，不创建或修改任何文件。"""
    print("  【干跑模式 --list】不加载模型，不写任何文件")
    print("=" * 68)

    will, skip_exist, no_alpha, broken = [], [], [], []
    for p in candidates:
        try:
            arr = load_rgba(p)
        except BaseException as exc:
            broken.append((p, f"{type(exc).__name__}: {exc}"))
            continue
        has, n_zero, n_semi = alpha_info(arr)
        dst = out_root / f"{p.stem}_x{cfg['upscale']}{out_ext_for(cfg, p)}"
        if cfg["skip_exist"] and dst.is_file() and not args.force:
            skip_exist.append(p)
        elif not has:
            no_alpha.append(p)   # 无 alpha：按不透明处理
        else:
            h, w = arr.shape[:2]
            will.append((p, w, h, n_zero, n_semi))

    print(f"\n■ 会处理：{len(will)} 张")
    total_px = 0
    for p, w, h, nz, ns in will:
        total_px += w * h
        print(f"    {p.name}")
        print(f"        {w}x{h}  ->  {w * cfg['upscale']}x{h * cfg['upscale']}"
              f"    透明像素 {nz}   半透明 {ns}")
    if will:
        est = max(0.2, total_px * cfg["upscale"] ** 2 / 5.66e6 * 14.4 / 60)
        out_mp = total_px * cfg["upscale"] ** 2 / 1e6
        print(f"\n    输出合计约 {out_mp:.1f} 百万像素，"
              f"预计耗时约 {est:.1f} 分钟（不含模型加载）")

    def block(title: str, items: list) -> None:
        if not items:
            return
        print(f"\n■ {title}：{len(items)} 张")
        for it in items:
            print(f"    {it.name}")

    block("跳过 · 已有输出（要重跑加 --force）", skip_exist)
    block("跳过 · 没有透明通道（想强行放大加 --allow-no-alpha）", no_alpha)
    block("按文件名排除 · 程序自己的产物", excluded)
    if broken:
        print(f"\n■ 打不开 / 已损坏：{len(broken)} 张")
        for p, msg in broken:
            print(f"    {p.name}")
            print(f"        {msg}")

    print("\n" + "=" * 68)
    print(f"结论：正式运行会处理 {len(will)} 张图。去掉 --list 即为实跑。")
    return 0


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------
class ProgressView:
    """把各阶段的瓦片进度渲染成一行原地刷新的状态条。

    阶段由模块的进度钩子推送：load / image / vae_encode / dino / dit /
    vae_decode / align / done。每个阶段内部按瓦片计数。
    """

    NAME = {
        "load": "载入模型",
        "image": "图片",
        "vae_encode": "VAE 编码",
        "dino": "视觉特征",
        "dit": "DiT 推理",
        "vae_decode": "VAE 解码",
        "align": "颜色校正",
        "save": "写文件",
        "done": "全部完成",
    }
    BAR = 22

    def __init__(self, log) -> None:
        self.log = log
        self.t0 = time.time()
        self.stage = None
        self.done = 0
        self.total = 0
        self.stage_t0 = self.t0
        self.img_i = 0
        self.img_n = 0
        self.w = 0
        self.last_mark = -1
        self.last_sig = None
        self.tty = False
        try:
            self.tty = bool(sys.stdout.isatty())
        except Exception:
            self.tty = False

    def _newline(self) -> None:
        if self.tty and self.w:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.w = 0

    def _draw(self, msg: str, force: bool = False) -> None:
        if self.tty:
            pad = max(self.w, len(msg))
            sys.stdout.write("\r" + msg + " " * (pad - len(msg)))
            sys.stdout.flush()
            self.w = pad
            return
        # 非终端（写日志）：按 10% 一档记录，避免刷成几百行
        if not force:
            mark = int(self.done / self.total * 10) if self.total else 10
            if mark <= self.last_mark:
                return
            self.last_mark = mark
        self.log(msg)

    def __call__(self, st: str, d: int, n: int) -> None:
        now = time.time()
        sig = (st, int(d), int(n))
        if sig == self.last_sig:
            return
        self.last_sig = sig
        changed = (st != self.stage)
        if changed:
            self._newline()
            self.stage = st
            self.stage_t0 = now
            self.last_mark = -1
        self.done, self.total = int(d), int(n)
        if st == "image":
            self.img_i, self.img_n = self.done, self.total

        el = now - self.stage_t0
        tot = now - self.t0
        frac = (self.done / self.total) if self.total else 0.0
        filled = int(self.BAR * frac)
        bar = "#" * filled + "." * (self.BAR - filled)

        # image 在每张图开始(start)和结束(start+len)各推一次，这里换算成 1 基的当前张号
        cur = min(self.img_i + 1, self.img_n) if self.img_n else self.img_i

        eta = ""
        if self.done > 0 and self.total > self.done:
            sec = el / self.done * (self.total - self.done)
            if sec >= 1:
                eta = f"  剩余约 {sec:.0f}s"

        label = self.NAME.get(st, st)
        if st == "done":
            who, body = "", label
        elif st == "image":
            who = ""
            body = f"第 {cur}/{self.img_n} 张" if self.img_n else label
        elif st == "save":
            who = f"图{cur}/{self.img_n} " if self.img_n else ""
            body = f"写文件 {min(self.done + 1, self.total)}/{self.total}"
        else:
            who = f"图{cur}/{self.img_n} " if self.img_n else ""
            body = (label if self.done else f"{label} ...") if self.total <= 1 \
                else f"{label} [{bar}] {self.done}/{self.total} {frac * 100:3.0f}%"

        force = changed or (bool(self.total) and self.done >= self.total)
        self._draw(f"    {who}{body}   {tot:5.1f}s{eta}", force=force)

        if self.total and self.done >= self.total:
            self._newline()


def vram_info():
    """返回 (总量GB, 空闲GB)。检测不到返回 (None, None)。"""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return total / float(2 ** 30), free / float(2 ** 30)
    except BaseException:
        pass
    return None, None


def tiles_for_vram(gb_total, gb_free=None):
    """按显存选分块，返回 (tile_size, vae_tile)。

    主依据是**总量**（稳定、不受当前占用影响），因为实测 384/512 在
    10240x10240 输出上峰值 4.42 GB（其中 3.36 GB 是模型权重本身）。
    空闲显存特别少时再降一档，避免被别的程序挤爆。
    """
    if gb_total is None:
        ts, vt = 384, 512
    elif gb_total <= 6.5:
        ts, vt = 256, 384
    elif gb_total <= 9.5:
        ts, vt = 384, 512
    else:
        ts, vt = 512, 1024

    if gb_free is not None:
        if gb_free < 4.5:
            ts, vt = 256, 384
        elif gb_free < 5.5 and ts > 384:
            ts, vt = 384, 512
    return ts, vt


def resolve_tiles(cfg):
    """把 low_vram 配置解析成实际分块，并打印依据。"""
    mode = cfg.get("low_vram", "auto")
    ts0, vt0 = cfg["tile_size"], cfg["vae_tile"]

    if isinstance(mode, bool):
        flag = "true" if mode else "false"
    else:
        flag = str(mode).strip().lower()

    gb_total, gb_free = vram_info()

    if flag in ("false", "0", "off", "no", "manual"):
        print(f"  [分块] 手动设置 DiT={ts0}  VAE={vt0}")
        return ts0, vt0

    if flag in ("true", "1", "on", "yes", "low"):
        ts, vt = min(ts0, 384), min(vt0, 512)
        print(f"  [分块] 低显存模式 DiT={ts}  VAE={vt}")
        return ts, vt

    ts, vt = tiles_for_vram(gb_total, gb_free)
    if gb_total is None:
        print(f"  [显存] 检测不到显卡信息")
        print(f"  [分块] 保守使用 DiT={ts}  VAE={vt}")
    else:
        free_txt = f"，当前空闲 {gb_free:.1f} GB" if gb_free is not None else ""
        print(f"  [显存] 总容量 {gb_total:.1f} GB{free_txt}")
        print(f"  [分块] DiT={ts}  VAE={vt}")
        if gb_free is not None and gb_free < gb_total * 0.55:
            print("         （空闲偏少，已按空闲值降档；"
                  "关掉别的占显卡的程序会更快）")
    return ts, vt


def mem_info_gb():
    """返回 (物理内存总量GB, 当前可用GB)。失败返回 (None, None)。"""
    try:
        import ctypes as _c

        class _M(_c.Structure):
            _fields_ = [("dwLength", _c.c_ulong), ("dwMemoryLoad", _c.c_ulong),
                        ("ullTotalPhys", _c.c_ulonglong), ("ullAvailPhys", _c.c_ulonglong),
                        ("ullTotalPageFile", _c.c_ulonglong),
                        ("ullAvailPageFile", _c.c_ulonglong),
                        ("ullTotalVirtual", _c.c_ulonglong),
                        ("ullAvailVirtual", _c.c_ulonglong),
                        ("ullAvailExtendedVirtual", _c.c_ulonglong)]

        m = _M()
        m.dwLength = _c.sizeof(m)
        if _c.windll.kernel32.GlobalMemoryStatusEx(_c.byref(m)):
            return m.ullTotalPhys / float(2 ** 30), m.ullAvailPhys / float(2 ** 30)
    except BaseException:
        pass
    return None, None


def trim_other_processes():
    """清掉其它进程的物理工作集（EmptyWorkingSet），返回 (前可用GB, 后可用GB, 成功数)。

    这就是各种启动器「内存清理」按钮干的事：把别的程序占着但暂时用不到的物理页
    挤到备用列表，让大块连续内存更容易分配出来。已经打开的其它程序下次切换回去时
    会稍微卡一下（页要从磁盘换回），但数据不会丢。
    系统进程和别的用户的进程没权限，会被跳过。
    """
    try:
        before = mem_info_gb()[1]
    except BaseException:
        before = None
    done = 0
    try:
        import ctypes as _c
        import ctypes.wintypes as _wt

        _psapi = _c.WinDLL("psapi.dll")
        _k32 = _c.WinDLL("kernel32.dll")
        _psapi.EnumProcesses.argtypes = [_c.POINTER(_wt.DWORD), _wt.DWORD,
                                         _c.POINTER(_wt.DWORD)]
        _k32.OpenProcess.argtypes = [_wt.DWORD, _wt.BOOL, _wt.DWORD]
        _k32.OpenProcess.restype = _wt.HANDLE
        _psapi.EmptyWorkingSet.argtypes = [_wt.HANDLE]
        _psapi.EmptyWorkingSet.restype = _wt.BOOL

        QUERY, SETQ = 0x0400, 0x0100
        QUERY_LIMITED = 0x1000
        cap = 8192
        arr = (_wt.DWORD * cap)()
        need = _wt.DWORD()
        if not _psapi.EnumProcesses(arr, _c.sizeof(arr), _c.byref(need)):
            return before, before, 0
        n = min(need.value // _c.sizeof(_wt.DWORD), cap)
        me = _k32.GetCurrentProcessId()
        for i in range(n):
            pid = arr[i]
            if pid == 0 or pid == me:
                continue
            h = _k32.OpenProcess(QUERY | SETQ, False, pid)
            if not h:
                # 严格权限打不开时退一步：受限查询权限在很多同用户进程上也够用
                h = _k32.OpenProcess(QUERY_LIMITED | SETQ, False, pid)
            if not h:
                continue
            try:
                if _psapi.EmptyWorkingSet(h):
                    done += 1
            finally:
                _k32.CloseHandle(h)
    except BaseException:
        pass
    try:
        after = mem_info_gb()[1]
    except BaseException:
        after = before
    return before, after, done


def _enable_privilege(priv: str) -> bool:
    """给当前进程启用一个令牌特权。非管理员时该特权通常不在令牌里，会失败。"""
    try:
        import ctypes as _c
        import ctypes.wintypes as _wt

        class LUID(_c.Structure):
            _fields_ = [("LowPart", _wt.DWORD), ("HighPart", _wt.LONG)]

        class LUID_AND_ATTRIBUTES(_c.Structure):
            _fields_ = [("Luid", LUID), ("Attributes", _wt.DWORD)]

        class TOKEN_PRIVILEGES(_c.Structure):
            _fields_ = [("PrivilegeCount", _wt.DWORD),
                        ("Privileges", LUID_AND_ATTRIBUTES * 1)]

        adv = _c.WinDLL("advapi32.dll", use_last_error=True)
        k32 = _c.WinDLL("kernel32.dll", use_last_error=True)
        tok = _wt.HANDLE()
        if not adv.OpenProcessToken(k32.GetCurrentProcess(), 0x0028,
                                    _c.byref(tok)):        # ADJUST|QUERY
            return False
        try:
            luid = LUID()
            if not adv.LookupPrivilegeValueW(None, priv, _c.byref(luid)):
                return False
            tp = TOKEN_PRIVILEGES()
            tp.PrivilegeCount = 1
            tp.Privileges[0].Luid = luid
            tp.Privileges[0].Attributes = 0x0002       # SE_PRIVILEGE_ENABLED
            if not adv.AdjustTokenPrivileges(tok, False, _c.byref(tp),
                                             _c.sizeof(tp), None, None):
                return False
            return _c.get_last_error() == 0            # 1300 = 特权不存在
        finally:
            k32.CloseHandle(tok)
    except BaseException:
        return False


def trim_memory():
    """参考 Mem Reduct 的内存清理。返回 (前可用GB, 后可用GB, 做了什么)。

    比逐个进程 EmptyWorkingSet 更彻底，多出四步（都需要管理员）：
      1. NtSetSystemInformation(80, 2)  MemoryEmptyWorkingSets   —— 清全系统工作集
      2. NtSetSystemInformation(80, 3)  MemoryFlushModifiedList  —— 刷修改页列表
      3. NtSetSystemInformation(80, 4)  MemoryPurgeStandbyList   —— 清备用列表
      4. SetSystemFileCacheSize(-1, -1) —— 清系统文件缓存
    没有管理员权限时这几步会失败，自动退回逐进程 EmptyWorkingSet。
    """
    before = mem_info_gb()[1]
    notes = []

    if _enable_privilege("SeProfileSingleProcessPrivilege"):
        try:
            import ctypes as _c
            import ctypes.wintypes as _wt

            ntdll = _c.WinDLL("ntdll.dll")
            ntdll.NtSetSystemInformation.restype = _wt.LONG
            ntdll.NtSetSystemInformation.argtypes = [_wt.ULONG, _c.c_void_p,
                                                     _wt.ULONG]
            for cmd, label in ((2, "全系统工作集"), (3, "修改页列表"),
                               (4, "备用列表")):
                v = _wt.ULONG(cmd)
                if ntdll.NtSetSystemInformation(80, _c.byref(v),
                                                _c.sizeof(v)) == 0:
                    notes.append(label)
        except BaseException:
            pass

    if _enable_privilege("SeIncreaseQuotaPrivilege"):
        try:
            import ctypes as _c
            import ctypes.wintypes as _wt

            k32 = _c.WinDLL("kernel32.dll", use_last_error=True)
            k32.SetSystemFileCacheSize.argtypes = [_c.c_size_t, _c.c_size_t,
                                                   _wt.DWORD]
            k32.SetSystemFileCacheSize.restype = _wt.BOOL
            if k32.SetSystemFileCacheSize(_c.c_size_t(-1).value,
                                          _c.c_size_t(-1).value, 0):
                notes.append("系统文件缓存")
        except BaseException:
            pass

    if not notes:
        # 没权限走系统级 —— 退回逐个进程清工作集
        _, _, cnt = trim_other_processes()
        if cnt:
            notes.append(f"{cnt} 个进程工作集")

    after = mem_info_gb()[1]
    return before, after, notes


def is_admin() -> bool:
    """当前进程是否以管理员身份运行。"""
    try:
        import ctypes as _c
        return bool(_c.windll.shell32.IsUserAnAdmin())
    except BaseException:
        return False


def estimate_need_gb(out_px: int) -> float:
    """估算单张图处理时需要的常驻内存（GB）。

    基于实测标定：模型权重 + 框架本身约 4.5 GB；处理部分随输出像素线性增长，
    约 36 字节/像素（8432x8432 = 7110 万像素 -> 约 2.5 GB）。
    """
    return 4.5 + out_px * 36e-9


def ask_upscale(cur: int, w: int, h: int, mem_avail: float) -> int:
    """内存不够时让用户选一个能跑得动的放大倍率。"""
    def need(f: int) -> float:
        return estimate_need_gb(w * h * f * f)

    print()
    print("  " + "=" * 62)
    print(f"  !! 内存可能不够：用 x{cur} 预计需要 {need(cur):.1f} GB，"
          f"当前可用 {mem_avail:.1f} GB")
    print("  " + "=" * 62)

    choices = []
    for f in (4, 3, 2, 1):
        if f > cur:
            continue
        n = need(f)
        fits = n <= mem_avail * 0.9
        print(f"     {len(choices) + 1}) x{f}   {w * f}x{h * f}"
              f"   预计需要 {n:5.1f} GB   {'OK' if fits else '内存不够'}")
        choices.append(f)

    default = next((f for f in choices if need(f) <= mem_avail * 0.9), choices[-1])
    print()
    print(f"  输入 1-{len(choices)} 后回车（直接回车 = x{default}）: ", end="")
    try:
        raw = input().strip()
    except (EOFError, KeyboardInterrupt):
        raw = ""
    print()
    pick = default
    if raw.isdigit() and 1 <= int(raw) <= len(choices):
        pick = choices[int(raw) - 1]
    print(f"  -> 使用 x{pick}，预计需要 {need(pick):.1f} GB")
    return pick


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("folder", nargs="?", default=None,
                    help="要处理的文件夹，默认是 .bat 所在目录")
    ap.add_argument("--bat-dir", default=None, help="内部使用")
    ap.add_argument("-u", "--upscale", type=int, default=None, choices=(1, 2, 3, 4))
    ap.add_argument("-f", "--format", default=None,
                    choices=("png", "webp", "jpeg", "same"), help="输出格式")
    ap.add_argument("--force", action="store_true", help="已存在也重跑")
    ap.add_argument("--keep-temp", action="store_true", help="保留中间文件")
    ap.add_argument("--allow-no-alpha", action="store_true",
                    help="没有 alpha 的图也放大（输出不透明 PNG）")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="干跑：只列出会处理哪些图，不加载模型、不写任何文件")
    ap.add_argument("-y", "--yes", action="store_true", help="跳过倒计时")
    args = ap.parse_args()

    cfg = dict(CFG)
    if args.upscale:
        cfg["upscale"] = args.upscale
    if getattr(args, "format", None):
        cfg["out_format"] = args.format
    if args.keep_temp:
        cfg["keep_temp"] = True
    if args.no_preview:
        cfg["preview"] = False

    # 解析分块大小：low_vram = "auto" 自动按显存选，True 强制低，False 用手写值
    cfg["tile_size"], cfg["vae_tile"] = resolve_tiles(cfg)

    bat_dir = Path(args.bat_dir) if args.bat_dir else Path(__file__).resolve().parent

    # --- 决定处理哪个文件夹 ---
    #  优先级：命令行参数 > 配置区 input_dir > 本 .bat 所在目录
    if args.folder:
        folder = Path(args.folder)
    elif cfg.get("input_dir"):
        cand = bat_dir / cfg["input_dir"]
        folder = cand if cand.is_dir() else bat_dir
    else:
        folder = bat_dir
    folder = Path(folder).resolve()
    if not folder.is_dir():
        print(f"[错误] 找不到文件夹：{folder}")
        return 2

    # --- 决定输出到哪 ---
    _od = cfg.get("output_dir")
    if _od:
        _p = Path(_od)
        out_root = (_p if _p.is_absolute() else (bat_dir / _p)).resolve()
    else:
        out_root = folder / OUT_DIR_NAME

    print("=" * 68)
    print("  VOSR2 透明底 PNG 一键放大  ——  独立版（不依赖 VOSR2Tool）")
    print("=" * 68)
    print(f"  图片文件夹 : {folder}")
    print(f"  输出文件夹 : {out_root}")
    print(f"  放大倍数   : x{cfg['upscale']}")
    print(f"  分块       : DiT={cfg['tile_size']}  VAE={cfg['vae_tile']}"
          f"  噪声尺度={cfg['noise_scale']}")
    _fmt = str(cfg["out_format"]).lower()
    if _fmt == "same":
        print("  输出格式   : 与原图格式相同")
    elif _fmt == "jpeg":
        print("  输出格式   : JPEG（不支持透明，透明区域铺白底）")
    elif _fmt == "webp":
        print("  输出格式   : WEBP（支持透明）")
    else:
        print("  输出格式   : PNG（支持透明）")

    candidates, excluded = [], []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if any(p.name.lower().endswith(s) for s in EXCLUDE_SUFFIXES):
            excluded.append(p)
        else:
            candidates.append(p)

    # --- 干跑 ---
    if args.list:
        return print_list(folder, out_root, candidates, excluded, cfg, args)

    # --- 权重 ---
    models_root = find_models_root(bat_dir)
    if models_root is None:
        print("[错误] 找不到模型权重。用以下任一方式指定：")
        print("       1) 设置环境变量 VOSR2_MODELS 指向权重根目录")
        print("       2) 在本 .bat 同目录建 vosr2_models.txt，写一行路径")
        print("       3) 把权重放到 .bat 同目录的 models\\ 子目录")
        print("       权重根目录里需要有这 5 个文件：")
        for parts in REQUIRED_REL:
            print("         " + "/".join(parts))
        return 2
    missing = [p for p in required_files(models_root) if not p.is_file()]
    if missing:
        print(f"[错误] 权重不全（{models_root}），缺少：")
        for p in missing:
            print(f"        {p}")
        return 2
    print(f"  权重目录   : {models_root}")

    # --- 清掉上一次的派生文件（预览 / 日志 / 中间文件）---
    #  只删这些由程序生成的东西；已经输出好的 _xN.png 成品一个都不动。
    if cfg.get("clean_output", True) and out_root.is_dir():
        killed = []
        for name, always in ((PREVIEW_DIR_NAME, True), (WORK_DIR_NAME, False)):
            d = out_root / name
            if not d.is_dir():
                continue
            if not always and cfg.get("keep_temp"):
                continue          # 用户明确要留中间文件，就别删
            try:
                n = sum(1 for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d, ignore_errors=True)
                if n:
                    killed.append(f"{name}({n} 个)")
            except BaseException:
                pass
        lf = out_root / LOG_NAME
        if lf.is_file():
            try:
                lf.unlink()
                killed.append(LOG_NAME)
            except OSError:
                pass          # 被别的程序占用就算了，稍后会覆盖
        if killed:
            print(f"  [清理] 已移除上次的 {'、'.join(killed)}（成品图保留）")

    # --- 目录与日志 ---
    work = out_root / WORK_DIR_NAME
    work_in = work / "in"
    work_out = work / "out"
    preview_dir = out_root / PREVIEW_DIR_NAME
    for d in (out_root, work, work_in, work_out):
        d.mkdir(parents=True, exist_ok=True)
    log_f = open(out_root / LOG_NAME, "w", encoding="utf-8")

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    print(f"  输出目录   : {out_root}")
    print("-" * 68)
    log(f"开始运行  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"文件夹={folder}  倍数=x{cfg['upscale']}  DiT分块={cfg['tile_size']} "
        f"VAE分块={cfg['vae_tile']} 噪声尺度={cfg['noise_scale']} seed={cfg['seed']}")

    if not candidates:
        log("[提示] 这个文件夹里没有支持的图片（PNG / JPEG / WebP）。")
        log_f.close()
        return 0

    # --- 分类 ---
    todo, no_alpha, skip_exist = [], [], []
    metas: dict = {}
    for p in candidates:
        arr = load_rgba(p)
        has, n_zero, n_semi = alpha_info(arr)
        if not has:
            no_alpha.append(p)   # 无 alpha：按不透明处理，不再跳过
        dst = out_root / f"{p.stem}_x{cfg['upscale']}{out_ext_for(cfg, p)}"
        if cfg["skip_exist"] and dst.is_file() and not args.force:
            skip_exist.append(p)
            continue
        todo.append(p)
        metas[p.stem] = {"n_zero": n_zero, "n_semi": n_semi}

    log(f"待处理 {len(todo)} 张 | 已有输出跳过 {len(skip_exist)} 张 | "
        f"无 alpha（按不透明处理）{len(no_alpha)} 张")
    for p in skip_exist:
        log(f"  [跳过·已存在] {p.name}")
    for p in no_alpha:
        log(f"  [无 alpha·按不透明处理] {p.name}")

    # --- 补回被跳过图片的预览 ---
    #  启动时清空过 _预览\，被跳过的图（已有输出）不会重跑，所以它们的预览要在这里补，
    #  否则 _预览\ 会只剩本次新处理的那几张。要放在 "if not todo: return" 之前，
    #  否则「全部跳过」时这段永远不执行。
    if cfg["preview"] and skip_exist:
        pdir = out_root / PREVIEW_DIR_NAME
        try:
            pdir.mkdir(parents=True, exist_ok=True)
        except BaseException:
            pass
        fixed = 0
        for p in skip_exist:
            final = out_root / f"{p.stem}_x{cfg['upscale']}{out_ext_for(cfg, p)}"
            if not final.is_file():
                continue
            try:
                make_preview(final, pdir)
                fixed += 1
            except BaseException as exc:
                log(f"    [预览补回失败] {p.name}: {type(exc).__name__}: {exc}")
        if fixed:
            log(f"已补回 {fixed} 张被跳过图片的预览")

    if not todo:
        log("没有需要处理的图片。")
        log_f.close()
        return 0

    total_px = sum(int(np.asarray(Image.open(p).size).prod()) for p in todo)

    # --- 内存检查：预估最大一张的常驻需求，不够就让用户选倍率 ---
    sizes = [Image.open(p).size for p in todo]
    w_max, h_max = max(sizes, key=lambda s: s[0] * s[1])
    mem_total, mem_avail = mem_info_gb()

    # 先整理一次内存：把别的程序占着的物理页挤走，好给大图腾地方
    if cfg.get("trim_memory", True):
        b, a, notes = trim_memory()
        if notes:
            gain = (a - b) if (a is not None and b is not None) else 0.0
            print(f"  [清理内存] {'、'.join(notes)}，"
                  f"可用 {b:.1f} GB -> {a:.1f} GB"
                  + (f"（+{gain:.1f} GB）" if gain > 0.05 else ""))
        else:
            print("  [清理内存] 没有可整理的内容（权限不足时会这样，不影响运行）")
        mem_total, mem_avail = mem_info_gb()

    if mem_total is not None:
        print(f"  内存       : 总容量 {mem_total:.1f} GB，"
              f"当前可用 {mem_avail:.1f} GB")
    need = estimate_need_gb(w_max * h_max * cfg["upscale"] ** 2)
    print(f"  最大一张   : {w_max}x{h_max} x{cfg['upscale']} -> "
          f"{w_max * cfg['upscale']}x{h_max * cfg['upscale']}，"
          f"预计需要 {need:.1f} GB")

    if mem_total is not None and need > mem_avail * 0.9:
        if args.upscale is None and not args.yes:
            cfg["upscale"] = ask_upscale(cfg["upscale"], w_max, h_max, mem_avail)
            log(f"用户选择放大倍率 x{cfg['upscale']}")
            total_px = sum(int(np.asarray(Image.open(p).size).prod()) for p in todo)
        else:
            print("  !! 内存可能不够（指定了 -u 或加了 -y 时不询问）。")
            print("     如果中途失败，去掉 -y 重跑会让你选倍率。")
        if not is_admin():
            print("  [提示] 右键「以管理员身份运行」可以清得更彻底：")
            print("         清全系统工作集 + 修改页列表 + 备用列表 + 系统文件缓存")

    est_min = max(0.3, total_px * cfg["upscale"] ** 2 / 5.66e6 * 14.4 / 60 + 0.2)
    log(f"预计耗时约 {est_min:.1f} 分钟（含模型加载）")
    if not args.yes:
        print("  3 秒后开始，按 Ctrl+C 可取消 ...")
        for i in (3, 2, 1):
            print(f"    {i}")
            time.sleep(1)

    # --- 载入模型（只加载一次，后面每张图复用，避免反复读写 5 GB 权重）---
    print("-" * 68)
    log("[1/2] 载入模型")
    if V is None:
        log("[错误] 无法导入内嵌的 VOSR 2 模块：")
        log(f"        {V_IMPORT_ERROR}")
        log("")
        log("        最常见原因：这个 Python 没有装 torch。")
        log("        请先运行 安装Python环境.bat，")
        log("        或在 .bat 同目录的 vosr2_env.txt 里指向正确的 python.exe。")
        log_f.close()
        return 3

    prog = ProgressView(log)
    pipeline = None
    t_load = time.time()
    try:
        prog("load", 0, 1)
        pipeline = V.load_pipeline(models_root=str(models_root), dtype=cfg["dtype"])
        prog("load", 1, 1)
    except BaseException as exc:
        log(f"  [错误] 模型加载失败：{type(exc).__name__}: {exc}")
        log_f.close()
        return 3
    log(f"  模型就绪，用时 {time.time() - t_load:.1f} 秒")

    def _release_vram() -> None:
        """把这一张用过的显存还给驱动，下一张从干净状态开始。"""
        try:
            import gc
            gc.collect()                      # 先让 Python 真正回收张量
        except BaseException:
            pass
        try:
            _torch = sys.modules.get("torch")
            if _torch is not None and _torch.cuda.is_available():
                _torch.cuda.empty_cache()     # 再把空闲块还给驱动
        except BaseException:
            pass

    # --- 逐张处理：拆分 → 推理 → 合成 → 预览 → 释放 ---
    print("-" * 68)
    log("[2/2] 逐张处理（每张做完立即出图并释放显存）")
    done, failed = [], []
    n = len(todo)
    t0 = time.time()

    for idx, p in enumerate(todo, 1):
        print("-" * 68)
        log(f"[{idx}/{n}] {p.name}")
        prog.img_i, prog.img_n = idx - 1, n
        seg = time.time()
        try:
            # 清空中间目录，保证这一张独占（否则 upscale_batch 会把上一张再跑一遍）
            for d in (work_in, work_out):
                for f in d.glob("*"):
                    if f.is_file():
                        f.unlink()

            # 1) 拆分
            info = split_one(p, work_in / f"{p.stem}.rgb.png",
                             work / f"{p.stem}.alpha.png", cfg)
            metas[p.stem].update(info)
            log(f"    拆分  {info['width']}x{info['height']}  "
                f"透明{info['alpha_zero']} 半透明{info['alpha_semi']}")

            # 2) 推理（复用已加载的 pipeline，只跑这一张）
            #    显存不足时自动降档重试；降成功的那一档会记住给后面的图用
            def _sub(st, d, tot):
                if st != "image":
                    prog(st, d, tot)

            ladder = []
            for pair in ((cfg["tile_size"], cfg["vae_tile"]),
                         (384, 512), (256, 384), (192, 256)):
                if pair not in ladder:
                    ladder.append(pair)

            res = None
            for li, (ts, vt) in enumerate(ladder):
                if li:
                    print()
                    log(f"    显存不足，降到 DiT={ts} VAE={vt} 重试 ...")
                    _release_vram()
                res = V.upscale_batch(
                    work_in, work_out,
                    upscale=cfg["upscale"],
                    tile_size=ts,
                    tile_overlap=min(cfg["tile_overlap"], max(16, ts // 8)),
                    vae_tile_size=vt,
                    vae_tile_overlap=min(cfg["vae_tile_overlap"], max(32, vt // 8)),
                    seed=cfg["seed"],
                    color_alignment=cfg["align"],
                    dtype=cfg["dtype"],
                    models_root=str(models_root),
                    noise_scale=cfg["noise_scale"],
                    batch_size=1,
                    pipeline=pipeline,
                    progress=_sub,
                )
                bad = "; ".join(f"{k}: {v}"
                                for k, v in (res.get("failed") or {}).items())
                if not bad:
                    if li:
                        cfg["tile_size"], cfg["vae_tile"] = ts, vt
                        log(f"    降档成功，后面的图沿用 DiT={ts} VAE={vt}")
                    break
                if "out of memory" in bad.lower() or "outofmemory" in bad.lower():
                    continue
                raise RuntimeError(bad)
            else:
                raise RuntimeError("显存不足：已降到最小分块 DiT=192 VAE=256 仍然失败")

            if res.get("failed"):
                raise RuntimeError("; ".join(f"{k}: {v}"
                                             for k, v in res["failed"].items()))

            # 3) 合成
            rgb_out = work_out / f"{p.stem}.rgb.png"
            if not rgb_out.is_file():
                raise RuntimeError(f"模型没有产生输出：{rgb_out.name}")
            target = Image.open(rgb_out).size
            a_big = make_alpha(work, work / f"{p.stem}.alpha.png", target, cfg)
            final = out_root / f"{p.stem}_x{cfg['upscale']}{out_ext_for(cfg, p)}"
            _, amin, amax, n0 = merge_one(rgb_out, a_big, final, metas[p.stem], cfg)
            log(f"    合成  {final.name}  {target[0]}x{target[1]}  "
                f"alpha {amin}~{amax}  透明{n0}  "
                f"{final.stat().st_size / 2**20:.1f} MB")

            # 4) 预览
            if cfg["preview"]:
                try:
                    make_preview(final, preview_dir)
                except BaseException as exc:
                    log(f"    [预览失败] {type(exc).__name__}: {exc}")

            log(f"    这一张用时 {time.time() - seg:.1f} 秒"
                f"（累计 {time.time() - t0:.1f} 秒）")
            done.append((p, final))
        except BaseException as exc:
            log(f"    [失败] {type(exc).__name__}: {exc}")
            failed.append(p)
        finally:
            _release_vram()
            log("    显存已释放")

    # --- 清理 ---
    if not cfg["keep_temp"] and not failed:
        shutil.rmtree(work, ignore_errors=True)
        log("中间文件已清理（要保留请把配置区的 keep_temp 改成 True）")
    elif failed:
        log(f"有 {len(failed)} 张失败，中间文件保留在 {work} 以便排查")

    # --- 汇总 ---
    print("=" * 68)
    log(f"完成 {len(done)} 张 | 跳过已存在 {len(skip_exist)} 张 | "
        f"无 alpha {len(no_alpha)} 张（已按不透明处理）| 失败 {len(failed)} 张")
    for p, final in done:
        log(f"  OK   {final.name}")
    for p in failed:
        log(f"  FAIL {p.name}")

    if failed:
        log("")
        log("排查建议：")
        log("  · 显存不足：把配置区的 tile_size / vae_tile 调小")
        log('      nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv')
        log("  · 若占用方是已经不存在的 PID，任何降分块都没用，重启电脑即可")
    log(f"日志：{out_root / LOG_NAME}")
    log_f.close()

    return 0 if not failed else 1


# ===========================================================================
#  图形界面 —— 本地网页 UI
#  只监听 127.0.0.1，用浏览器当界面。embeddable Python 自带 http.server，
#  不需要任何额外依赖；文件夹选择走 ctypes 弹原生对话框（网页拿不到真实路径）。
# ===========================================================================

# 拖拽上传的临时文件统一放这个子目录：成功后自动删除，失败则保留。
# 放在子目录里就绝不会碰到用户手动放进 vosr2_input 根目录的图。
UPLOAD_DIR_NAME = "_拖入"

GUI_HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>VOSR 2 一键放大</title>
<style>
 *{box-sizing:border-box}
 body{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#1b1e24;color:#e6e6e6;
      margin:0;padding:24px}
 .wrap{max-width:880px;margin:0 auto}
 h1{font-size:19px;font-weight:600;margin:0 0 18px;color:#f1f5f9}
 .card{background:#242830;border-radius:10px;padding:18px;margin-bottom:14px;
       box-shadow:0 1px 3px rgba(0,0,0,.3)}
 .row{display:flex;align-items:center;gap:10px;margin:10px 0}
 label{width:92px;flex:none;color:#9aa4b2;font-size:13px}
 input[type=text]{flex:1;background:#171a20;border:1px solid #333a45;color:#e6e6e6;
   padding:8px 10px;border-radius:6px;font-size:13px;outline:none}
 input[type=text]:focus{border-color:#3b82f6}
 button{background:#3b82f6;color:#fff;border:0;padding:8px 16px;border-radius:6px;
   font-size:13px;cursor:pointer;font-family:inherit}
 button.ghost{background:#333a45}
 button:disabled{opacity:.4;cursor:not-allowed}
 .seg{display:flex;gap:6px}
 .seg button{background:#333a45;padding:7px 18px}
 .seg button.on{background:#3b82f6}
 .bar{height:24px;background:#171a20;border-radius:6px;overflow:hidden;position:relative}
 .bar>i{display:block;height:100%;width:0;background:linear-gradient(90deg,#2563eb,#60a5fa);
   transition:width .25s}
 .bar>span{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
   font-size:12px;letter-spacing:.4px;text-shadow:0 1px 2px rgba(0,0,0,.6)}
 .info{font-size:13px;color:#9aa4b2;line-height:1.9;margin-top:10px;white-space:pre-line}
 pre{background:#14171c;border-radius:8px;padding:12px;height:230px;overflow:auto;
   font-size:12px;line-height:1.65;white-space:pre-wrap;margin:10px 0 0;
   font-family:Consolas,monospace}
 .warn{color:#f59e0b}
 select{background:#171a20;border:1px solid #333a45;color:#e6e6e6;padding:7px 10px;
   border-radius:6px;font-size:13px;font-family:inherit;outline:none;cursor:pointer}
 select:focus{border-color:#3b82f6}
 .hint{color:#6b7280;font-size:12px;margin-left:4px}
 .drop{border:2px dashed #3b4757;border-radius:10px;padding:24px;text-align:center;
   color:#9aa4b2;font-size:14px;line-height:2;transition:.15s}
 .drop.over{border-color:#3b82f6;background:#1e2a3d;color:#f1f5f9}
 .qlist{margin-top:10px;font-size:12px;color:#9aa4b2;max-height:132px;overflow:auto}
 .qlist div{padding:3px 2px;border-bottom:1px solid #2b3038}
 .qlist .tag{color:#60a5fa;margin:0 6px}
</style></head><body><div class="wrap">
<h1>VOSR 2 透明底 PNG 一键放大</h1>

<div class="card">
  <div class="row"><label>图片文件夹</label>
    <input id="in" type="text" placeholder="选一个装着透明 PNG 的文件夹">
    <button class="ghost" onclick="pick('in')">浏览</button></div>
  <div class="row"><label>输出文件夹</label>
    <input id="out" type="text" placeholder="留空 = 图片文件夹\vosr2_output">
    <button class="ghost" onclick="pick('out')">浏览</button></div>
  <div class="row"><label>放大倍率</label>
    <div class="seg" id="up">
      <button data-v="1">x1</button><button data-v="2">x2</button>
      <button data-v="3">x3</button><button data-v="4" class="on">x4</button></div></div>
  <div class="row"><label>显存模式</label>
    <div class="seg" id="lv">
      <button data-v="auto" class="on">自动</button>
      <button data-v="on">低显存</button>
      <button data-v="off">手动</button></div></div>
  <div class="row" id="tilerow" style="display:none"><label>分块</label>
    <span class="hint">DiT</span>
    <select id="dit">
      <option value="192">192</option><option value="256">256</option>
      <option value="384">384</option><option value="512">512</option>
      <option value="768">768</option><option value="1024">1024</option>
    </select>
    <span class="hint">VAE</span>
    <select id="vae">
      <option value="256">256</option><option value="384">384</option>
      <option value="512">512</option><option value="1024">1024</option>
      <option value="1280">1280</option><option value="1536">1536</option>
    </select>
    <span class="hint">越小越省显存，越大越快</span></div>
  <div class="row"><label>输出格式</label>
    <div class="seg" id="fmt">
      <button data-v="png" class="on">PNG</button>
      <button data-v="webp">WebP</button>
      <button data-v="jpeg">JPEG</button>
      <button data-v="same">与原图相同</button></div>
    <span class="hint" id="fmthint"></span></div>
  <div class="row"><label>其它</label>
    <label style="width:auto;color:#e6e6e6"><input type="checkbox" id="prev" checked> 生成预览</label>
    <label style="width:auto;color:#e6e6e6"><input type="checkbox" id="clean" checked> 清理上次的派生文件</label>
  </div>
  <div class="row" style="margin-top:18px">
    <label></label>
    <button id="go" onclick="start()">开始</button>
    <button id="stop" class="ghost" onclick="cancel()" disabled>取消</button>
    <button class="ghost" onclick="openOut()">打开输出文件夹</button>
    <button class="ghost" onclick="quitApp()">退出程序</button>
  </div>
</div>

<div class="card">
  <div class="bar"><i id="fill"></i><span id="pct">就绪</span></div>
  <div class="info" id="stat"></div>
  <pre id="log"></pre>
</div>

<div class="card">
  <div class="drop" id="drop">
    <b>把图片拖到这里（PNG / JPEG / WebP）</b><br>
    <span class="hint">图片可以在电脑任意位置，输出统一放到 vosr2_output</span>
  </div>
  <div class="row" style="margin-top:14px">
    <label></label>
    <button class="ghost" id="pickfiles" onclick="pickFiles()">选择图片文件…</button>
    <button class="ghost" onclick="clearQueue()">清空列表</button>
    <span class="hint" id="qhint"></span>
  </div>
  <div id="qlist" class="qlist"></div>
</div>

<script>
const $ = s => document.querySelector(s);
let S = {in:'', out:'', upscale:4, low_vram:'auto', dit:512, vae:1024,
         fmt:'png', preview:true, clean:true};
let saveTimer = null;

function paint() {
  document.querySelectorAll('#up button').forEach(b =>
    b.classList.toggle('on', +b.dataset.v === S.upscale));
  document.querySelectorAll('#lv button').forEach(b =>
    b.classList.toggle('on', b.dataset.v === S.low_vram));
  document.querySelectorAll('#fmt button').forEach(b =>
    b.classList.toggle('on', b.dataset.v === (S.fmt || 'png')));
  const fh = $( '#fmthint' );
  if (fh) fh.textContent = (S.fmt === 'same') ? '每张跟随它自己的源格式' :
    (S.fmt === 'jpeg')
    ? 'JPEG 不支持透明，透明区域会铺白底'
    : '支持透明，文件更小';
  $('#tilerow').style.display = (S.low_vram === 'off') ? 'flex' : 'none';
  $('#dit').value = String(S.dit);
  $('#vae').value = String(S.vae);
  if (document.activeElement !== $('#in')) $('#in').value = S.in;
  if (document.activeElement !== $('#out')) $('#out').value = S.out;
  $('#prev').checked = !!S.preview;
  $('#clean').checked = !!S.clean;
}

function touch() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 350);
}

async function save() {
  try {
    await fetch('/api/save', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify(S)});
  } catch (e) {}
}

$('#up').onclick = e => {
  if (e.target.dataset.v) { S.upscale = +e.target.dataset.v; paint(); touch(); }
};
$('#lv').onclick = e => {
  if (e.target.dataset.v) { S.low_vram = e.target.dataset.v; paint(); touch(); }
};
$('#fmt').onclick = e => {
  if (e.target.dataset.v) { S.fmt = e.target.dataset.v; paint(); touch(); }
};
['dit','vae'].forEach(k => {
  $('#' + k).onchange = e => { S[k] = +e.target.value; touch(); };
});
$('#in').oninput = e => { S.in = e.target.value; touch(); };
$('#out').oninput = e => { S.out = e.target.value; touch(); };
$('#prev').onchange = e => { S.preview = e.target.checked; touch(); };
$('#clean').onchange = e => { S.clean = e.target.checked; touch(); };

async function pick(kind) {
  const r = await fetch('/api/browse', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({kind})});
  const j = await r.json();
  if (j.path) { S[kind] = j.path; paint(); touch(); }
}

async function openOut() {
  await fetch('/api/open_out', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({dir: S.out})});
}

async function quitApp() {
  if (!confirm('退出后就释放所有内存了。下次要用请重新双击「一键放大.bat」。\n确定退出吗？'))
    return;
  try { await fetch('/api/quit', {method:'POST'}); } catch (e) {}
  document.body.innerHTML =
    '<div style="padding:60px;text-align:center;color:#9aa4b2;line-height:2">' +
    '<h2 style="color:#e6e6e6">已退出</h2>' +
    '内存已释放。可以关掉这个标签页和那个命令行窗口了。</div>';
}

async function cancel() {
  await fetch('/api/cancel', {method:'POST'});
  $('#stat').textContent = '正在取消…（当前这一张会被立即中止）';
}

const drop = document.getElementById('drop');
['dragenter', 'dragover'].forEach(ev => {
  drop.addEventListener(ev, e => {
    e.preventDefault();
    e.stopPropagation();
    drop.classList.add('over');
  });
});
['dragleave', 'dragend'].forEach(ev => {
  drop.addEventListener(ev, e => {
    e.preventDefault();
    drop.classList.remove('over');
  });
});
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());

let QUEUE = [];      // 待处理列表：{path, mode:'drop'|'src', label}

function baseName(p) { return p.split(/[\\/]/).pop(); }

function renderQueue() {
  const el = $('#qlist');
  el.innerHTML = QUEUE.map((q, i) =>
    '<div>' + (i + 1) + '. <span class="tag">' +
    (q.mode === 'drop' ? '输出到 vosr2_output' : '输出到原图旁') +
    '</span>' + q.label + '</div>').join('');
  $('#qhint').textContent = QUEUE.length
    ? ('已选 ' + QUEUE.length + ' 张，点「开始」才处理')
    : '列表为空时，点「开始」= 处理上面文件夹里的全部图';
  $('#go').textContent = QUEUE.length ? ('开始（' + QUEUE.length + ' 张）') : '开始';
}

function addQueue(items) {
  for (const it of items) {
    if (!QUEUE.some(q => q.path === it.path)) QUEUE.push(it);
  }
  renderQueue();
}

async function clearQueue() {
  const drops = QUEUE.filter(q => q.mode === 'drop').map(q => q.path);
  if (drops.length) {
    try {
      const r = await fetch('/api/clear_queue', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({paths: drops})});
      const j = await r.json();
      if (!j.ok) { alert(j.msg || '临时文件没删掉'); return; }
    } catch (e) { return; }
  }
  QUEUE = [];
  renderQueue();
}

drop.addEventListener('drop', async e => {
  e.preventDefault();
  e.stopPropagation();
  drop.classList.remove('over');
  const files = [...e.dataTransfer.files].filter(f => /[.](png|jpe?g|webp)$/i.test(f.name));
  if (!files.length) { alert('只支持 PNG / JPEG / WebP 文件'); return; }
  const hint = '<b>把图片拖到这里（PNG / JPEG / WebP）</b><br>' +
               '<span class="hint">图片可以在电脑任意位置，输出统一放到 vosr2_output</span>';
  drop.innerHTML = '正在接收 ' + files.length + ' 张 …';
  const got = [];
  for (const f of files) {
    try {
      const r = await fetch('/api/upload?name=' + encodeURIComponent(f.name),
        {method:'POST', headers:{'Content-Type':'application/octet-stream'}, body: f});
      const j = await r.json();
      if (j.ok && j.path) {
        got.push({path: j.path, mode: 'drop', label: j.name});
      }
    } catch (err) {}
  }
  drop.innerHTML = hint;
  if (!got.length) { alert('没有成功接收任何文件'); return; }
  addQueue(got);            // 只排队，不自动开始
});

async function pickFiles() {
  const b = $('#pickfiles');
  b.disabled = true;
  b.textContent = '等待对话框…';
  try {
    const r = await fetch('/api/pick_files', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: '{}'});
    const j = await r.json();
    if (j.files && j.files.length) {
      addQueue(j.files.map(p => ({path: p, mode: 'src', label: baseName(p)})));
    }
  } catch (e) { alert('选择失败：' + e); }
  b.disabled = false;
  b.textContent = '选择图片文件…';
}

async function start() {
  const body = Object.assign({}, S);
  if (QUEUE.length) {
    body.tasks = QUEUE.map(q => ({path: q.path, mode: q.mode}));
    body.out = S.drop_out || S.out;
  }
  const r = await fetch('/api/start', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  if (!j.ok) { alert(j.msg || '启动失败'); return; }
  poll();
}

async function poll() {
  const j = await (await fetch('/api/state')).json();
  const pct = j.total ? Math.round(j.done / j.total * 100) : 0;
  $('#fill').style.width = pct + '%';
  $('#pct').textContent = j.running
    ? ((j.n ? '图' + j.cur + '/' + j.n + '  ' : '') + j.stage + '   ' + pct + '%')
    : (j.finished ? '已完成' : '就绪');
  if (j.stat !== undefined) $('#stat').textContent = j.stat || '';
  if (j.log && j.log.length) {
    const el = $('#log');
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    el.textContent = j.log.join('\n');
    if (atBottom) el.scrollTop = el.scrollHeight;
  }
  $('#go').disabled = j.running;
  $('#stop').disabled = !j.running;
}

(async () => {
  try {
    const j = await (await fetch('/api/defaults')).json();
    Object.assign(S, j);
  } catch (e) {}
  paint();
  renderQueue();
  poll();
  setInterval(poll, 800);
})();
</script></div></body></html>
"""


class GuiJob:
    """GUI 与服务端共享的运行状态。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.cancel_flag = False
        self.finished = False
        self.stage = ""
        self.done = 0
        self.total = 0
        self.cur = 0
        self.n = 0
        self.stat = ""
        self.lines = []

    def log(self, msg: str = "") -> None:
        with self.lock:
            self.lines.append(str(msg))
            if len(self.lines) > 1200:
                del self.lines[:400]

    def progress(self, stage: str, done: int, total: int) -> None:
        with self.lock:
            if stage == "image":
                self.n = int(total)
                self.cur = min(int(done) + 1, int(total)) if total else 0
                return
            self.stage = {
                "load": "载入模型", "vae_encode": "VAE 编码", "dino": "视觉特征",
                "dit": "DiT 推理", "vae_decode": "VAE 解码", "align": "颜色校正",
                "save": "写文件", "done": "全部完成",
            }.get(stage, stage)
            self.done, self.total = int(done), int(total)

    def snapshot(self) -> dict:
        with self.lock:
            return {"running": self.running, "finished": self.finished,
                    "stage": self.stage, "done": self.done, "total": self.total,
                    "cur": self.cur, "n": self.n, "stat": self.stat,
                    "log": list(self.lines[-300:])}


GUI = GuiJob()


def pick_folder(title: str = "选择文件夹") -> str:
    """弹 Windows 原生文件夹选择框。返回路径，取消返回空串。"""
    try:
        import ctypes as _c
        import ctypes.wintypes as _wt

        class BROWSEINFO(_c.Structure):
            _fields_ = [("hwndOwner", _wt.HWND), ("pidlRoot", _c.c_void_p),
                        ("pszDisplayName", _wt.LPWSTR),
                        ("lpszTitle", _wt.LPCWSTR), ("ulFlags", _wt.UINT),
                        ("lpfn", _c.c_void_p), ("lParam", _wt.LPARAM),
                        ("iImage", _c.c_int)]

        shell32 = _c.WinDLL("shell32.dll")
        ole32 = _c.WinDLL("ole32.dll")
        shell32.SHBrowseForFolderW.restype = _c.c_void_p
        shell32.SHGetPathFromIDListW.argtypes = [_c.c_void_p, _wt.LPWSTR]

        name = _c.create_unicode_buffer(260)
        bi = BROWSEINFO()
        bi.hwndOwner = None
        bi.pidlRoot = None
        bi.pszDisplayName = _c.cast(name, _wt.LPWSTR)
        bi.lpszTitle = title
        bi.ulFlags = 0x0001 | 0x0040 | 0x0010   # 只要目录 | 新样式 | 可输入
        bi.lpfn = None
        bi.lParam = 0
        bi.iImage = 0

        pidl = shell32.SHBrowseForFolderW(_c.byref(bi))
        if not pidl:
            return ""
        try:
            path = _c.create_unicode_buffer(260)
            if shell32.SHGetPathFromIDListW(_c.c_void_p(pidl), path):
                return path.value or ""
            return name.value or ""
        finally:
            ole32.CoTaskMemFree(_c.c_void_p(pidl))
    except BaseException:
        return ""


def pick_image_files(title: str = "选择图片") -> list:
    """弹 Windows 原生多选文件框，返回绝对路径列表（取消返回空列表）。"""
    try:
        import ctypes as _c
        import ctypes.wintypes as _wt

        class OPENFILENAMEW(_c.Structure):
            _fields_ = [
                ("lStructSize", _wt.DWORD), ("hwndOwner", _wt.HWND),
                ("hInstance", _wt.HINSTANCE), ("lpstrFilter", _wt.LPCWSTR),
                ("lpstrCustomFilter", _wt.LPWSTR), ("nMaxCustFilter", _wt.DWORD),
                ("nFilterIndex", _wt.DWORD), ("lpstrFile", _wt.LPWSTR),
                ("nMaxFile", _wt.DWORD), ("lpstrFileTitle", _wt.LPWSTR),
                ("nMaxFileTitle", _wt.DWORD), ("lpstrInitialDir", _wt.LPCWSTR),
                ("lpstrTitle", _wt.LPCWSTR), ("Flags", _wt.DWORD),
                ("nFileOffset", _wt.WORD), ("nFileExtension", _wt.WORD),
                ("lpstrDefExt", _wt.LPCWSTR), ("lCustData", _wt.LPARAM),
                ("lpfnHook", _c.c_void_p), ("lpTemplateName", _wt.LPCWSTR),
                ("pvReserved", _c.c_void_p), ("dwReserved", _wt.DWORD),
                ("FlagsEx", _wt.DWORD),
            ]

        SIZE = 65536
        buf = _c.create_unicode_buffer(SIZE)
        ofn = OPENFILENAMEW()
        ofn.lStructSize = _c.sizeof(ofn)
        ofn.hwndOwner = None
        ofn.hInstance = None
        ofn.lpstrFilter = ("图片文件\0*.png;*.jpg;*.jpeg;*.webp\0"
                           "PNG\0*.png\0JPEG\0*.jpg;*.jpeg\0WebP\0*.webp\0"
                           "所有文件\0*.*\0\0")
        ofn.nFilterIndex = 1
        ofn.lpstrFile = _c.cast(buf, _wt.LPWSTR)
        ofn.nMaxFile = SIZE
        ofn.lpstrTitle = title
        ofn.Flags = (0x00000200 |   # ALLOWMULTISELECT
                     0x00080000 |   # EXPLORER
                     0x00001000 |   # FILEMUSTEXIST
                     0x00000800 |   # PATHMUSTEXIST
                     0x00000008)    # NOCHANGEDIR

        comdlg = _c.WinDLL("comdlg32.dll")
        comdlg.GetOpenFileNameW.argtypes = [_c.POINTER(OPENFILENAMEW)]
        comdlg.GetOpenFileNameW.restype = _wt.BOOL
        if not comdlg.GetOpenFileNameW(_c.byref(ofn)):
            return []

        # 多选时缓冲区是 "目录\0文件1\0文件2\0\0"；单选时只有一条完整路径
        parts = [s for s in "".join(buf).split("\x00") if s]
        if not parts:
            return []
        if len(parts) == 1:
            return [parts[0]]
        base = Path(parts[0])
        return [str(base / n) for n in parts[1:]]
    except BaseException:
        return []


def gui_worker(cfg: dict, folder: Path, out_root: Path, models_root: Path,
               tasks=None) -> None:
    """GUI 的任务线程。流程与 main() 保持一致（刻意重复，避免改动已验证的控制台路径）。"""
    gui = GUI
    pipeline = None
    try:
        work = out_root / WORK_DIR_NAME
        work_in, work_out = work / "in", work / "out"
        preview_dir = out_root / PREVIEW_DIR_NAME
        for d in (out_root, work, work_in, work_out):
            d.mkdir(parents=True, exist_ok=True)

        gui.log(f"图片文件夹：{folder}")
        gui.log(f"输出文件夹：{out_root}")
        gui.log(f"放大 x{cfg['upscale']}   DiT={cfg['tile_size']}  "
                f"VAE={cfg['vae_tile']}   噪声尺度={cfg['noise_scale']}")

        # 收集待处理项：每项是 (源文件, 输出目录)
        pairs = []
        if tasks:
            for t in tasks:
                try:
                    p = Path(str(t.get("path") or ""))
                except Exception:
                    continue
                if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                # mode=drop -> 统一放到 vosr2_output；否则写到原图旁边
                drop = t.get("mode") == "drop"
                pairs.append((p, out_root if drop else p.parent, drop))
            gui.log(f"来自列表：{len(pairs)} 张")
        else:
            for p in iter_images(folder):
                pairs.append((p, out_root, False))
        if not pairs:
            gui.log("这个文件夹里没有支持的图片（PNG / JPEG / WebP）。")
            gui.finished = True
            return

        if cfg.get("clean_output", True) and out_root.is_dir():
            for nm in (PREVIEW_DIR_NAME, WORK_DIR_NAME):
                d = out_root / nm
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
            lf = out_root / LOG_NAME
            if lf.is_file():
                try:
                    lf.unlink()
                except OSError:
                    pass
            gui.log("已清理上次的预览/日志/中间文件（成品图保留）")

        todo, skipped = [], []
        for p, dst_dir, is_drop in pairs:
            dst = dst_dir / f"{p.stem}_x{cfg['upscale']}{out_ext_for(cfg, p)}"
            try:
                arr = load_rgba(p)
                has, _, _ = alpha_info(arr)
            except BaseException as exc:
                # 坏图/损坏文件不应该让整个任务挂掉
                gui.log(f"[跳过·读不了] {p.name}（{type(exc).__name__}）")
                continue
            del arr
            if not has:
                gui.log(f"  [无 alpha] {p.name}（按不透明处理）")
            if cfg["skip_exist"] and dst.is_file() and not cfg.get("force"):
                skipped.append((p, dst_dir, is_drop))
                continue
            todo.append((p, dst_dir, is_drop))

        for p, dst_dir, _ in skipped:
            gui.log(f"[跳过·已存在] {p.name}")
            final = dst_dir / f"{p.stem}_x{cfg['upscale']}{out_ext_for(cfg, p)}"
            if cfg["preview"]:
                try:
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    make_preview(final, preview_dir)
                except BaseException:
                    pass

        gui.n = len(todo)
        if not todo:
            gui.log("没有需要处理的图片。")
            gui.stat = f"完成：跳过 {len(skipped)} 张（已存在）"
            gui.finished = True
            return

        gui.progress("load", 0, 1)
        gui.log("载入模型…")
        pipeline = V.load_pipeline(models_root=str(models_root), dtype=cfg["dtype"])
        gui.progress("load", 1, 1)
        gui.log("模型就绪")

        def hook(st, d, tot):
            # 每个进度回调都是一次取消检查点 —— 涵盖 DiT 每块、VAE 每块等
            gui.progress(st, d, tot)
            if gui.cancel_flag:
                raise KeyboardInterrupt

        def checkpoint():
            if gui.cancel_flag:
                raise KeyboardInterrupt

        ok = 0
        for idx, (p, dst_dir, is_drop) in enumerate(todo, 1):
            if gui.cancel_flag:
                gui.log("已取消。")
                break
            gui.cur = idx
            gui.log("")
            gui.log(f"[{idx}/{len(todo)}] {p.name}")
            for d in (work_in, work_out):
                for f in d.glob("*"):
                    if f.is_file():
                        f.unlink()
            try:
                info = split_one(p, work_in / f"{p.stem}.rgb.png",
                                 work / f"{p.stem}.alpha.png", cfg)
                gui.log(f"    拆分 {info['width']}x{info['height']}")
                checkpoint()
                V.upscale_batch(
                    work_in, work_out, upscale=cfg["upscale"],
                    tile_size=cfg["tile_size"], tile_overlap=cfg["tile_overlap"],
                    vae_tile_size=cfg["vae_tile"],
                    vae_tile_overlap=cfg["vae_tile_overlap"], seed=cfg["seed"],
                    color_alignment=cfg["align"], dtype=cfg["dtype"],
                    models_root=str(models_root),
                    noise_scale=cfg["noise_scale"], batch_size=1,
                    pipeline=pipeline, progress=hook)
                checkpoint()
                rgb_out = work_out / f"{p.stem}.rgb.png"
                if not rgb_out.is_file():
                    raise RuntimeError("模型没有产生输出")
                target = Image.open(rgb_out).size
                checkpoint()
                a_big = make_alpha(work, work / f"{p.stem}.alpha.png", target, cfg)
                checkpoint()
                final = dst_dir / f"{p.stem}_x{cfg['upscale']}{out_ext_for(cfg, p)}"
                merge_one(rgb_out, a_big, final, {}, cfg)
                gui.log(f"    完成 {final.name}  {target[0]}x{target[1]}  "
                        f"{final.stat().st_size / 2**20:.1f} MB")
                # 拖进来的临时副本：出图成功就删；失败会走 except，文件保留
                if is_drop and final.is_file():
                    try:
                        p.unlink()
                        gui.log("    已删除拖入的临时副本")
                    except OSError:
                        pass
                if cfg["preview"]:
                    try:
                        make_preview(final, preview_dir)
                    except BaseException:
                        pass
                ok += 1
            except KeyboardInterrupt:
                gui.log("已取消。")
                break
            except BaseException as exc:
                gui.log(f"    [失败] {type(exc).__name__}: {exc}")
            finally:
                try:
                    gc.collect()
                    _t = sys.modules.get("torch")
                    if _t is not None and _t.cuda.is_available():
                        _t.cuda.empty_cache()
                except BaseException:
                    pass

        # 拖入目录空了就顺手删掉，别留个空文件夹
        try:
            updir = Path(folder) / UPLOAD_DIR_NAME
            if updir.is_dir() and not any(updir.iterdir()):
                updir.rmdir()
        except BaseException:
            pass

        gui.stat = f"完成 {ok} 张 | 跳过 {len(skipped)} 张"
        gui.log("")
        gui.log(gui.stat)
    except BaseException as exc:
        gui.log(f"[错误] {type(exc).__name__}: {exc}")
        gui.stat = f"出错：{type(exc).__name__}: {exc}"
    finally:
        # 图形界面的服务进程会一直活着。不主动把模型放掉的话，这几 GB 内存
        # 就一直占着，直到用户关掉命令行窗口 —— 也就是「跑完内存不下降」。
        if not cfg.get("keep_loaded"):
            try:
                if pipeline is not None:
                    # free_gpu() 是 VOSR2Model 的方法，Pipeline 本身没有
                    _mdl = getattr(pipeline, "model", None)
                    if _mdl is not None and hasattr(_mdl, "free_gpu"):
                        _mdl.free_gpu()          # 权重挪回 CPU、清 CUDA 缓存
                    elif hasattr(pipeline, "free_gpu"):
                        pipeline.free_gpu()
                    del pipeline
                    pipeline = None
                    gui.log("已释放模型（下次开始会重新载入，约 10 秒）")
            except BaseException:
                pass
            try:
                gc.collect()
                _t = sys.modules.get("torch")
                if _t is not None and _t.cuda.is_available():
                    _t.cuda.empty_cache()
            except BaseException:
                pass
            # 关键一步：Python 对象释放了，但 Windows 会把那些页留在本进程的
            # 工作集里慢慢才收回（大图反复申请 ~900 MB 大块时尤其明显，
            # 任务管理器里看着像"内存没降"）。主动收缩一次让系统立刻回收。
            try:
                import ctypes as _ct
                import ctypes.wintypes as _wt
                _k = _ct.WinDLL("kernel32.dll")
                _ps = _ct.WinDLL("psapi.dll")
                _k.GetCurrentProcess.restype = _wt.HANDLE
                _k.SetProcessWorkingSetSize.argtypes = [_wt.HANDLE,
                                                        _ct.c_size_t, _ct.c_size_t]
                _k.SetProcessWorkingSetSize.restype = _wt.BOOL
                _ps.EmptyWorkingSet.argtypes = [_wt.HANDLE]
                _ps.EmptyWorkingSet.restype = _wt.BOOL
                _h = _k.GetCurrentProcess()
                # -1/-1 = 让 Windows 把用不到的页全部换出去（本进程随即降到几十 MB）
                _k.SetProcessWorkingSetSize(_h, _ct.c_size_t(-1).value,
                                            _ct.c_size_t(-1).value)
                _ps.EmptyWorkingSet(_h)
            except BaseException:
                pass
        with gui.lock:
            gui.running = False
            gui.finished = True
            gui.stage = ""
        try:
            gui.log(f"日志：{out_root / LOG_NAME}")
            with open(out_root / LOG_NAME, "w", encoding="utf-8") as fh:
                fh.write("\n".join(gui.lines))
        except BaseException:
            pass


def gui_serve(bat_dir: Path, folder: Path, models_root: Path,
              port: int = 0) -> int:
    """起本地服务并打开浏览器，直到用户关掉页面/点退出。"""
    import http.server
    import socket
    import webbrowser

    base = dict(CFG)
    settings_file = bat_dir / "_界面设置.json"

    def defaults() -> dict:
        d = {"in": str(folder), "out": str(folder / OUT_DIR_NAME),
             "drop_out": str(bat_dir / OUT_DIR_NAME),
             "upscale": int(CFG["upscale"]), "low_vram": "auto",
             "fmt": str(CFG["out_format"]),
             "dit": int(CFG["tile_size"]), "vae": int(CFG["vae_tile"]),
             "preview": bool(CFG["preview"]),
             "clean": bool(CFG["clean_output"])}
        try:
            if settings_file.is_file():
                saved = json.loads(settings_file.read_text(encoding="utf-8"))
                for k in ("in", "out", "low_vram", "fmt"):
                    if isinstance(saved.get(k), str) and saved[k]:
                        d[k] = saved[k]
                for k in ("upscale", "dit", "vae"):
                    if isinstance(saved.get(k), int):
                        d[k] = saved[k]
                for k in ("preview", "clean"):
                    if isinstance(saved.get(k), bool):
                        d[k] = saved[k]
        except BaseException:
            pass
        return d

    def persist(data: dict) -> None:
        try:
            settings_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except BaseException:
            pass

    def resolve(data: dict) -> dict:
        c = dict(base)
        try:
            c["upscale"] = max(1, min(4, int(data.get("upscale") or 4)))
        except Exception:
            c["upscale"] = 4
        lv = str(data.get("low_vram") or "auto")
        if lv == "off":
            # 手动：直接用用户选的 DiT / VAE 分块
            try:
                ts = max(128, min(2048, int(data.get("dit") or 512)))
                vt = max(256, min(4096, int(data.get("vae") or 1024)))
            except Exception:
                ts, vt = 512, 1024
            c["low_vram"] = False
            c["tile_size"], c["vae_tile"] = ts, vt
        else:
            c["low_vram"] = True if lv == "on" else "auto"
            c["tile_size"], c["vae_tile"] = base["tile_size"], base["vae_tile"]
            ts, vt = resolve_tiles(c)
            c["tile_size"], c["vae_tile"] = ts, vt
        fmt = str(data.get("fmt") or "png").lower()
        c["out_format"] = fmt if fmt in ("png", "webp", "jpeg", "same") else "png"
        c["preview"] = bool(data.get("preview", True))
        c["clean_output"] = bool(data.get("clean", True))
        c["force"] = False
        return c

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BaseException:
                pass

        def _json(self, obj):
            self._send(200, json.dumps(obj, ensure_ascii=False))

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, GUI_HTML, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(GUI.snapshot())
            elif self.path == "/api/defaults":
                self._json(defaults())
            elif self.path == "/favicon.ico":
                self._send(204, b"")
            else:
                self._send(404, "{}")

        def _upload(self):
            """接收拖拽进来的 PNG（浏览器只给内容，不给真实路径）。"""
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            raw = (q.get("name") or ["drop.png"])[0]
            name = Path(raw).name                 # 只取文件名，挡掉路径穿越
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                return self._json({"ok": False, "msg": "只支持 PNG / JPEG / WebP"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except Exception:
                n = 0
            if n <= 0:
                return self._json({"ok": False, "msg": "空文件"})
            if n > 300 * 1024 * 1024:
                return self._json({"ok": False, "msg": "文件超过 300 MB"})
            try:
                in_dir = Path(folder) / UPLOAD_DIR_NAME
                in_dir.mkdir(parents=True, exist_ok=True)
                dest = in_dir / name
                stem, suf = dest.stem, dest.suffix
                k = 1
                while dest.exists() and dest.stat().st_size != n:
                    dest = in_dir / f"{stem}_{k}{suf}"
                    k += 1
                tmp = in_dir / (dest.name + ".part")
                left = n
                with open(tmp, "wb") as fh:
                    while left > 0:
                        buf = self.rfile.read(min(1 << 20, left))
                        if not buf:
                            break
                        fh.write(buf)
                        left -= len(buf)
                if left > 0:
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                    return self._json({"ok": False, "msg": "上传中断"})
                tmp.replace(dest)
                return self._json({"ok": True, "name": dest.name,
                                   "path": str(dest),
                                   "size": dest.stat().st_size})
            except BaseException as exc:
                return self._json({"ok": False,
                                   "msg": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            if self.path.startswith("/api/upload"):
                return self._upload()
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                data = {}

            if self.path == "/api/browse":
                kind = data.get("kind", "in")
                path = pick_folder("选择图片文件夹" if kind == "in" else "选择输出文件夹")
                return self._json({"path": path})

            if self.path == "/api/pick_files":
                return self._json({"files": pick_image_files("选择要放大的 PNG（可多选）")})

            if self.path == "/api/open_out":
                d = data.get("dir") or str(folder / OUT_DIR_NAME)
                try:
                    if Path(d).is_dir():
                        os.startfile(d)          # noqa: S606
                        return self._json({"ok": True})
                except BaseException:
                    pass
                return self._json({"ok": False})

            if self.path == "/api/save":
                persist(data)
                return self._json({"ok": True})

            if self.path == "/api/clear_queue":
                # 清空列表时把拖入的临时副本也删掉。
                # 只删 _拖入\ 里的文件；mode=src 指向用户真实文件，永远不碰。
                if GUI.running:
                    return self._json({"ok": False,
                                       "msg": "任务正在运行，先取消或等它完成"})
                updir = (Path(folder) / UPLOAD_DIR_NAME).resolve()
                killed = 0
                for raw in (data.get("paths") or []):
                    try:
                        p = Path(str(raw))
                        if not p.is_file():
                            continue
                        if updir not in p.resolve().parents:
                            continue        # 不在 _拖入 里，绝不删
                        p.unlink()
                        killed += 1
                    except BaseException:
                        pass
                try:
                    if updir.is_dir() and not any(updir.iterdir()):
                        updir.rmdir()
                except BaseException:
                    pass
                return self._json({"ok": True, "deleted": killed})

            if self.path == "/api/quit":
                self._json({"ok": True})
                # 稍等一下再硬退出，保证响应先发出去
                threading.Timer(0.3, lambda: os._exit(0)).start()
                return True

            if self.path == "/api/cancel":
                GUI.cancel_flag = True
                return self._json({"ok": True})

            if self.path == "/api/start":
                if GUI.running:
                    return self._json({"ok": False, "msg": "已经在跑了"})
                tasks = data.get("tasks") or None
                in_dir = Path(data.get("in") or folder)
                if tasks:
                    if not any(Path(str(t.get("path") or "")).is_file()
                               for t in tasks):
                        return self._json({"ok": False, "msg": "列表里的文件都不存在了"})
                elif not in_dir.is_dir():
                    return self._json({"ok": False,
                                       "msg": f"图片文件夹不存在：{in_dir}"})
                od = data.get("out") or ""
                out_root = Path(od) if od else (in_dir / OUT_DIR_NAME)
                c = resolve(data)
                with GUI.lock:
                    GUI.running = True
                    GUI.finished = False
                    GUI.cancel_flag = False
                    GUI.lines = []
                    GUI.stage = ""
                    GUI.done = GUI.total = 0
                    GUI.cur = GUI.n = 0
                    GUI.stat = f"放大 x{c['upscale']}   DiT={c['tile_size']}  " \
                               f"VAE={c['vae_tile']}"
                threading.Thread(target=gui_worker,
                                 args=(c, in_dir, out_root, models_root, tasks),
                                 daemon=True).start()
                return self._json({"ok": True})

            self._send(404, "{}")

    # 找一个空闲端口（显式给了 port 就只试那一个，测试用）
    picked = 0
    for p in ([int(port)] if port else range(8765, 8800)):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))
            picked = p
            break
        except OSError:
            continue
        finally:
            s.close()
    if not picked:
        print(f"[错误] 端口 {port} 被占用" if port else "[错误] 找不到空闲端口")
        return 2
    port = picked

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print("=" * 68)
    print("  图形界面已启动")
    print(f"  地址：{url}")
    print("  浏览器会自动打开；用完直接关掉这个窗口即可。")
    print("=" * 68)
    try:
        webbrowser.open(url)
    except BaseException:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[已退出]")
    finally:
        srv.server_close()
    return 0



# ===========================================================================
#  输入/输出格式：PNG / JPEG / WebP
# ===========================================================================
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
OUT_EXTS = {"png": ".png", "webp": ".webp", "jpeg": ".jpg"}

# 覆盖前面的旧定义：本程序自己的产物后缀（含 webp/jpg），免得被当成输入再放大
EXCLUDE_SUFFIXES = tuple(
    [f"_x{n}{e}" for n in (1, 2, 3, 4) for e in (".png", ".webp", ".jpg")]
    + [".rgb.png", ".alpha.png", "_preview.png", "_预览.png"]
)


def out_ext(cfg: dict) -> str:
    """当前配置对应的输出扩展名。"""
    return OUT_EXTS.get(str(cfg.get("out_format", "png")).lower(), ".png")


def out_ext_for(cfg: dict, src: Path = None) -> str:
    """按源文件决定输出扩展名（支持 out_format="same"）。"""
    fmt = str(cfg.get("out_format", "png")).lower()
    if fmt == "same" and src is not None:
        s = src.suffix.lower()
        if s in (".jpg", ".jpeg"):
            return ".jpg"
        if s == ".webp":
            return ".webp"
        return ".png"
    return OUT_EXTS.get(fmt, ".png")


def iter_images(folder: Path):
    """列出文件夹里所有支持的图片（跳过本程序自己的产物）。"""
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if any(p.name.lower().endswith(s) for s in EXCLUDE_SUFFIXES):
            continue
        yield p


def merge_one(rgb_path: Path, alpha_path: Path, dst: Path, meta: dict,
              cfg: dict = None) -> tuple:
    """合成 RGB + alpha 并按 out_format 写出（png / webp / jpeg）。

    覆盖前面只写 PNG 的版本。JPEG 没有 alpha 通道：透明区域会被铺成
    cfg["jpeg_bg"] 指定的底色（默认白）。
    """
    cfg = cfg or {}
    rgb = Image.open(rgb_path).convert("RGB")
    alpha = Image.open(alpha_path).convert("L")
    if rgb.size != alpha.size:
        alpha = alpha.resize(rgb.size, Image.LANCZOS)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)

    info = (meta or {}).get("png_info") or {}
    dpi = info.get("dpi")
    if isinstance(dpi, (list, tuple)) and len(dpi) == 2:
        out.info["dpi"] = tuple(float(x) for x in dpi)

    # 格式以目标文件名后缀为准 —— out_format="same" 时每张图各自跟随源格式
    _sfx = dst.suffix.lower()
    fmt = ("jpeg" if _sfx in (".jpg", ".jpeg")
           else "webp" if _sfx == ".webp" else "png")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jpeg":
        bg = {"white": (255, 255, 255), "black": (0, 0, 0)}.get(
            str(cfg.get("jpeg_bg", "white")).lower(), (255, 255, 255))
        flat = Image.new("RGB", out.size, bg)
        flat.paste(out, mask=out.split()[3])
        q = max(1, min(100, int(cfg.get("jpeg_quality", 95))))
        flat.save(dst, format="JPEG", quality=q, subsampling=0, optimize=True)
        del flat
    elif fmt == "webp":
        q = max(1, min(100, int(cfg.get("webp_quality", 95))))
        out.save(dst, format="WEBP", quality=q, method=6)
    else:
        pnginfo = PngImagePlugin.PngInfo()
        xmp = info.get("XML:com.adobe.xmp") or info.get("xmp")
        if xmp:
            pnginfo.add_text("XML:com.adobe.xmp", str(xmp))
        out.save(dst, format="PNG", pnginfo=pnginfo)

    chk = np.asarray(out)[:, :, 3]
    return dst, int(chk.min()), int(chk.max()), int((chk == 0).sum())



if __name__ == "__main__":
    try:
        # --gui：走图形界面（本地网页 UI）
        if "--gui" in sys.argv[1:]:
            # 注意：sys.argv[0] 是解包到临时目录的 driver.py，不是 .bat 所在目录。
            # 必须用 bat 传进来的 --bat-dir，否则找不到同级的 models\ 和 vosr2_input\。
            _bd = None
            if "--bat-dir" in sys.argv:
                try:
                    _bd = Path(sys.argv[sys.argv.index("--bat-dir") + 1]).resolve()
                except Exception:
                    _bd = None
            if _bd is None:
                _bd = Path(sys.argv[0]).resolve().parent
            _mr = find_models_root(_bd)
            if _mr is None:
                print("[错误] 找不到模型权重目录")
                raise SystemExit(2)
            raise SystemExit(gui_serve(_bd, _bd / CFG["input_dir"], Path(_mr)))
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[已取消]")
        raise SystemExit(130)

#MODSTART
# -*- coding: utf-8 -*-
r"""
vosr2_standalone -- ComfyUI-VOSR2 (VOSR 2.0 one-step 1.4B) as a single, self-contained
Python module. No ComfyUI, no einops, no huggingface_hub, no diffusers, no timm.

===============================================================================
SOURCE / PROVENANCE / LICENSE
===============================================================================
Ported from the open-source ComfyUI node pack **ComfyUI-VOSR2**
    * upstream project: https://github.com/ylchen333/ComfyUI-VOSR2
    * the local snapshot this file was ported from:
        E:\...\vosr2-alpha\_upstream\ComfyUI-VOSR2-main\
      whose files carry no embedded commit id; the vendored model sources
      (`models/lightningdit.py`, `models/pos_embed.py`, `models/rmsnorm.py`,
      `models/swiglu_ffn.py`, `models/qwenimage_vae2d.py`) each state their own
      provenance in their headers, reproduced verbatim below.
    * cross-checked against the official **VOSR** research release
      (`_upstream\VOSR-main\`): `inference_vosr_onestep.py`, `vosr.py`,
      `tiled_vae.py`, `models/lightningdit.py`.

License: **Apache-2.0**. The upstream repository is distributed under
Apache-2.0 (see its ``LICENSE``, 11356 bytes, Apache License 2.0), as are the
third-party projects the vendored model code derives from:
  * Meta DINOv2 (facebookresearch/dinov2)            -- Apache-2.0
  * Qwen-Image / Wan / HuggingFace diffusers teams   -- Apache-2.0
  * VOSR (Jingfeng Yao, HUST-VL)                     -- see _upstream\VOSR-main\LICENSE
  * EVA-02 rotary embeddings (lucidrains/rotary-embedding-torch) -- MIT
  * facebookresearch/DiT and willisma/SiT            -- CC-BY-NC / MIT (upstream lineage)
The Apache-2.0 license text is NOT inlined here; it is available at
``_upstream\ComfyUI-VOSR2-main\LICENSE`` and at
https://www.apache.org/licenses/LICENSE-2.0 . This file is a derivative work:
it retains the upstream module structure and algorithm line-for-line and only
performs the mechanical substitutions listed below.

===============================================================================
WHAT WAS CHANGED (mechanical only -- no algorithm was modified)
===============================================================================
1. ``import comfy.ops`` and the ``comfy.ops.disable_weight_init`` alias were
   removed.  ``comfy.ops.disable_weight_init.{Linear,Conv2d,LayerNorm,...}``
   are subclasses of the plain ``torch.nn`` equivalents that only override
   ``reset_parameters`` (to skip random init, since the checkpoint overwrites
   every parameter) and add an *opt-in* weight-casting forward path
   (``comfy_cast_weights``, which defaults to ``False``).  With
   ``comfy.memory_management.aimdo_enabled == False`` -- the default -- their
   ``_load_from_state_dict`` and ``forward`` are literally
   ``super().forward()``.  Verified directly against ComfyUI's ``comfy/ops.py``
   (``class disable_weight_init`` at line 507, ``class Linear`` at 541).
   Therefore ``ops.X`` -> ``nn.X`` is exact: identical module names, identical
   parameter/buffer names (``weight`` / ``bias`` / ...), identical state_dict
   keys, identical numerics.
       ops.Linear            -> nn.Linear
       ops.Conv2d            -> nn.Conv2d
       ops.LayerNorm         -> nn.LayerNorm
   ``GroupNorm`` / ``ConvTranspose2d`` are not used by this architecture.

2. ``einops`` removed.  Exactly 4 call sites existed upstream:
       models/pos_embed.py : rotate_half()             -> 2x rearrange
       models/pos_embed.py : VisionRotaryEmbeddingFast -> 1x repeat
       models/lightningdit.py : _get_dynamic_rope()    -> 1x repeat
   They are replaced by ``_einops_rearrange`` / ``_einops_repeat`` below,
   which implement precisely the patterns used (``"... (d r) -> ... d r"``,
   ``"... d r -> ... (d r)"``, ``"... n -> ... (n r)"``).  These are exact:
   einops' ``(...)``-grouping and ``repeat``'s trailing-axis expansion both
   lower to ``reshape`` / ``unsqueeze`` / ``expand`` / ``reshape``.
   ``torch.einsum`` calls are native torch and were left untouched.

3. ``import comfy.utils`` (``ProgressBar``) removed.  ``_ProgressBar`` below is
   a minimal drop-in exposing ``current`` / ``total`` / ``update_absolute``.
   ``inference.py`` only ever calls ``pbar.current``,
   ``pbar.update_absolute(pbar.current + 1)``, and construction with a total.
   A module-global hook (``set_progress_hook`` / ``_emit_progress``) lets the
   public API surface the same counts to the caller.

4. ``comfy.model_management`` / ``comfy.model_patcher`` removed.  Replaced by
   ``_ModelPatcher`` (just ``.model`` + ``.load_device`` + ``.offload_device``,
   the two attributes the upstream code reads) and ``_ResidencyManager``, which
   reproduces ``load_models_gpu([patcher], force_full_load=True)`` -- i.e. the
   guarantee that when ``VOSR2Model._load()`` returns, ``patcher.model`` is
   physically on ``patcher.load_device``.  ``dit_patcher.load_device`` is still
   read by ``run_vosr2`` exactly as upstream.
   ``comfy.model_management.get_torch_device()`` -> ``torch.device("cuda"/"cpu")``
   and ``unet_dtype(device)`` -> an explicit bf16/fp32 choice (see ``_resolve_dtype``).

5. ``huggingface_hub`` auto-download (``ensure_vosr2_files``) deleted entirely.
   The bundle is always read from a local ``models_root``; a missing file is a
   hard error.  ``folder_paths`` is likewise gone.

6. ``loader.py``'s ``_convert_dinov2_pth_to_safetensors`` is gone.  The DINOv2
   encoder is loaded straight from the PyTorch pickle with
   ``torch.load(..., map_location="cpu", weights_only=True)`` as instructed --
   no format conversion, nothing written to disk.

7. ``comfy.utils.load_torch_file`` -> ``torch.load(..., weights_only=True)``.
   ``safetensors.safe_open`` streaming is kept as-is for the two safetensors
   checkpoints (``_load_state_dict_lean``), including ``assign=True`` and the
   prefix-stripping / training-only-key drop rules.

8. ``torch.inference_mode(False)`` around construction + checkpoint loading is
   kept from upstream.  With plain ``nn`` modules it is no longer strictly
   required (it was a ``comfy.ops`` lazy-init hazard), but it is harmless and
   preserves the upstream contract.

9. Added: the public ``upscale_batch`` API, image I/O, models_root autodetection,
   and the VRAM residency policy (see below).  Everything under the
   "UPSTREAM MODEL DEFINITIONS" banner is a verbatim port.

===============================================================================
VRAM STRATEGY
===============================================================================
``run_vosr2`` (below) is the upstream function **unchanged**: bicubic
pre-scale -> reflect-pad to a multiple of 16 -> (optional) pad to square ->
tiled VAE encode -> DINOv2 vision features -> one-step flow-matching denoise
-> tiled VAE decode -> color alignment -> crop.  DiT tiling and VAE tiling are
both fully supported and **tiling is ON by default** (``tile_size=512``,
``tile_overlap=64``, ``vae_tile_size=1024``, ``vae_tile_overlap=128`` -- the
configuration the target tool has been measured to run at 8908x8760 output).

Per-item offload to CPU during the tiled path is preserved verbatim from
upstream: each item is moved to CPU the instant it is decoded, and color
alignment happens per item (never on the concatenated batch), so peak system
RAM is bounded by one item, not by the whole batch.

On top of that, ``_ResidencyManager`` bounds *GPU* residency: it tracks the
models currently on the GPU, evicts least-recently-used ones to the offload
device when a newly requested model would exceed the budget, and never evicts
the model it is about to run.  At least one model always stays resident, so the
all-offload device degenerates to "everything on GPU".  The budget defaults to
``peak_fraction * free VRAM at load time``.

===============================================================================
PUBLIC API
===============================================================================
    upscale_batch(in_dir, out_dir, *, upscale=4, tile_size=512,
                  tile_overlap=64, vae_tile_size=1024, vae_tile_overlap=128,
                  seed=42, color_alignment="wavelet", dtype="auto",
                  models_root=None, batch_size=1, progress=None) -> dict
        -> {"ok": [file names...], "failed": {file name: "error text"}}

    load_pipeline(models_root=None, dtype="auto") -> VOSR2Pipeline
    autodetect_models_root() -> str
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

__all__ = [
    "upscale_batch",
    "load_pipeline",
    "autodetect_models_root",
    "VOSR2Pipeline",
    "set_progress_hook",
    "VOSR2LoadError",
]

__version__ = "1.0.0"

log = logging.getLogger("vosr2_standalone")

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# The exact bundle layout this module reads (see the task spec).
_BUNDLE_NAME = "VOSR2"
_VAE_SUBDIR = "Qwen-Image-vae-2d"
_DINOV2_PTH_REL = ("torch_cache", "checkpoints", "dinov2_vitl14_pretrain.pth")

_DEFAULT_MODELS_ROOT = r"E:\VOSR2Tool\VOSR2Tool\models\preset\ckpts"


class VOSR2LoadError(RuntimeError):
    """Raised for any missing file, incompatible config, or malformed state dict."""


# ===========================================================================
#  einops replacements (exact lowering of the 4 upstream call sites)
# ===========================================================================

_GROUP_RE = re.compile(r"^\((\w+)\s+(\w+)\)$")


def _parse_axes(spec: str) -> list:
    """Split an einops axis list, keeping ``(...)`` groups intact.

    Upstream only uses simple ``name`` / ``(name name)`` axes, but the grouping
    parentheses may contain a space (``"(d r)"``), so a plain ``.split()`` is
    wrong.  The only supported group is the two-name form checked by
    :data:`_GROUP_RE`.
    """
    spec = spec.strip()
    axes = []
    while spec:
        if spec[0] == "(":
            close = spec.index(")")
            axes.append(spec[: close + 1])
            spec = spec[close + 1:].strip()
        else:
            parts = spec.split(None, 1)
            axes.append(parts[0])
            spec = parts[1].strip() if len(parts) > 1 else ""
    return axes


def _einops_rearrange(tensor: torch.Tensor, pattern: str, **axes_lengths) -> torch.Tensor:
    """Minimal ``einops.rearrange`` for the two patterns this port needs.

    Supported (and used) patterns:
        "... (d r) -> ... d r"   with ``r=2``
        "... d r -> ... (d r)"
    Both are pure metadata operations, so the lowering below is bit-exact.
    """
    lhs, rhs = [side.strip() for side in pattern.split("->")]

    def _split(spec):
        leading = None
        if spec.startswith("..."):
            leading = "..."
            spec = spec[3:].strip()
        return leading, _parse_axes(spec)

    l_lead, l_axes = _split(lhs)
    r_lead, r_axes = _split(rhs)

    # "... (d r) -> ... d r": split the trailing axis into two.
    if len(r_axes) == len(l_axes) + 1 and l_axes:
        split_axis = l_axes[-1]
        m = _GROUP_RE.match(split_axis)
        if m is not None:
            d_name, r_name = m.group(1), m.group(2)
            factor = axes_lengths[r_name]
            if r_axes[-2:] != [d_name, r_name] or l_axes[:-1] != r_axes[:-2]:
                raise ValueError(f"unsupported rearrange pattern: {pattern!r}")
            shape = tensor.shape
            return tensor.reshape(*shape[:-1], shape[-1] // factor, factor)

    # "... d r -> ... (d r)": merge the trailing two axes.
    if len(l_axes) == len(r_axes) + 1 and r_axes:
        merged = r_axes[-1]
        m = _GROUP_RE.match(merged)
        if m is not None:
            d_name, r_name = m.group(1), m.group(2)
            if l_axes[-2:] != [d_name, r_name] or r_axes[:-1] != l_axes[:-2]:
                raise ValueError(f"unsupported rearrange pattern: {pattern!r}")
            shape = tensor.shape
            return tensor.reshape(*shape[:-2], shape[-2] * shape[-1])

    raise ValueError(f"unsupported rearrange pattern: {pattern!r}")


def _einops_repeat(tensor: torch.Tensor, pattern: str, **axes_lengths) -> torch.Tensor:
    """Minimal ``einops.repeat`` for the single pattern this port needs.

    Supported (and used) pattern:
        "... n -> ... (n r)"     with ``r=<int>``

    einops repeats a *new trailing* axis and then groups it with the axis named
    before it; for a trailing-axis repeat that is exactly
    ``unsqueeze(-1).expand(...).reshape(...)``, i.e. ``tile`` on the last axis.
    """
    lhs, rhs = [side.strip() for side in pattern.split("->")]

    def _split(spec):
        if spec.startswith("..."):
            return "...", _parse_axes(spec[3:].strip())
        return None, _parse_axes(spec)

    l_lead, l_axes = _split(lhs)
    r_lead, r_axes = _split(rhs)
    if l_lead != "..." or r_lead != "...":
        raise ValueError(f"unsupported repeat pattern: {pattern!r}")
    if len(r_axes) != len(l_axes) or not l_axes:
        raise ValueError(f"unsupported repeat pattern: {pattern!r}")

    m = _GROUP_RE.match(r_axes[-1])
    if m is None or l_axes[:-1] != r_axes[:-1] or m.group(1) != l_axes[-1]:
        raise ValueError(f"unsupported repeat pattern: {pattern!r}")

    factor = axes_lengths[m.group(2)]
    return tensor.unsqueeze(-1).expand(*tensor.shape, factor).reshape(*tensor.shape[:-1], tensor.shape[-1] * factor)


# ===========================================================================
#  minimal ProgressBar (replacement for comfy.utils.ProgressBar)
# ===========================================================================

_PROGRESS_HOOK = None


def set_progress_hook(fn) -> None:
    """Install a global ``fn(stage: str, done: int, total: int)`` callback, or None."""
    global _PROGRESS_HOOK
    _PROGRESS_HOOK = fn


def _emit_progress(stage: str, done: int, total: int) -> None:
    hook = _PROGRESS_HOOK
    if hook is None:
        return
    try:
        hook(stage, int(done), int(total))
    except Exception:  # a broken reporter must never break inference
        log.debug("progress hook raised", exc_info=True)


class _ProgressBar:
    """Drop-in for ``comfy.utils.ProgressBar``: ``current`` / ``total`` / ``update_absolute``."""

    __slots__ = ("total", "current", "stage")

    def __init__(self, total: int, stage: str = "denoise"):
        self.total = int(total)
        self.current = 0
        self.stage = stage

    def update_absolute(self, value, total=None, preview=None):
        self.current = int(value)
        if total is not None:
            self.total = int(total)
        _emit_progress(self.stage, self.current, self.total)


# ===========================================================================
#  minimal ModelPatcher + residency manager
#  (replacement for comfy.model_patcher / comfy.model_management)
# ===========================================================================

class _ModelPatcher:
    """The whole ``ModelPatcher`` surface this port uses: ``.model`` + ``.load_device``.

    Upstream ``loader.py`` passes ``load_device``/``offload_device`` to
    ``comfy.model_patcher.ModelPatcher`` and ``inference.py`` reads
    ``dit_patcher.load_device``, so both attributes are kept verbatim.
    """

    def __init__(self, model: nn.Module, load_device, offload_device):
        self.model = model
        self.load_device = torch.device(load_device)
        self.offload_device = torch.device(offload_device)
        self._on_load_device = False

    def move_to_load_device(self) -> nn.Module:
        if not self._on_load_device:
            self.model.to(self.load_device)
            self._on_load_device = True
        return self.model

    def move_to_offload_device(self) -> None:
        if self._on_load_device and self.offload_device != self.load_device:
            self.model.to(self.offload_device)
            self._on_load_device = False


def _param_bytes(model: nn.Module, data_ptrs: set | None = None) -> int:
    if data_ptrs is None:
        data_ptrs = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t is None:
            continue
        ptr = t.data_ptr()
        if ptr in data_ptrs:
            continue
        data_ptrs.add(ptr)
        total += t.numel() * t.element_size()
    return total


class _ResidencyManager:
    """Bounds how much model weight is *gpu-resident* at once.

    Replaces ``comfy.model_management.load_models_gpu([patcher], force_full_load=True)``.
    On ``acquire(patcher)`` the manager guarantees ``patcher.model`` is on
    ``patcher.load_device`` before returning (the contract ``VOSR2Model._load``
    depends on).  Least-recently-used models are evicted to their offload device
    when keeping the requester resident would exceed the budget; the requester is
    never evicted, and at least one model is always left resident (so a
    CPU-offload device degenerates to "everything stays on the load device").
    """

    def __init__(self, budget_bytes=None, reserve_bytes: int = 512 * 2**20):
        self._patchers: list = []
        self._stamp = 0
        self._reserve = int(reserve_bytes)
        self._explicit_budget = budget_bytes
        self._budget = None
        self.evictions = 0

    def register(self, patcher: _ModelPatcher) -> None:
        if patcher not in self._patchers:
            self._patchers.append(patcher)

    def _compute_budget(self) -> int:
        if self._explicit_budget is not None:
            return int(self._explicit_budget)
        if not torch.cuda.is_available():
            return 1 << 62
        try:
            free, _total = torch.cuda.mem_get_info()
        except Exception:
            free = torch.cuda.get_device_properties(0).total_memory
        # Leave headroom for activations + the caching allocator's fragmentation.
        return max(int(free * 0.75) - self._reserve, 256 * 2**20)

    @property
    def budget_bytes(self) -> int:
        if self._budget is None:
            self._budget = self._compute_budget()
        return self._budget

    def _resident_bytes(self, skip=None) -> int:
        seen: set = set()
        total = 0
        for p in self._patchers:
            if p is skip or not p._on_load_device:
                continue
            total += _param_bytes(p.model, seen)
        return total

    def acquire(self, patcher: _ModelPatcher) -> nn.Module:
        self._stamp += 1
        patcher._last_use = self._stamp

        if patcher._on_load_device:
            return patcher.model
        if patcher.load_device == patcher.offload_device:
            return patcher.move_to_load_device()

        want = _param_bytes(patcher.model)
        self._evict_for(patcher, want)

        # Re-read the budget: `_compute_budget` is based on free VRAM, and
        # evicting frees VRAM, so this stays accurate across a long run.
        if self._explicit_budget is None:
            self._budget = None
        return patcher.move_to_load_device()

    def _evict_for(self, requester: _ModelPatcher, needed_bytes: int) -> None:
        """Move LRU models to their offload device until `needed_bytes` would fit."""
        while self._resident_bytes(skip=requester) + needed_bytes > self.budget_bytes:
            resident = [p for p in self._patchers if p._on_load_device and p is not requester]
            if len(resident) <= 0:
                return  # nothing else to give: the requester has to fit or fail loudly
            victim = min(resident, key=lambda p: getattr(p, "_last_use", 0))
            victim.move_to_offload_device()
            self.evictions += 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def release_all(self) -> None:
        for p in self._patchers:
            p.move_to_offload_device()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ===========================================================================
#  UPSTREAM MODEL DEFINITIONS -- verbatim port
#  _upstream\ComfyUI-VOSR2-main\models\pos_embed.py
#  Vendored from VOSR's models/pos_embed.py (upstream: EVA-02 rotary
#  embeddings, MIT, itself adapted from lucidrains/rotary-embedding-torch),
#  trimmed to VisionRotaryEmbeddingFast and its helpers -- the only pieces
#  LightningDiT's use_rope path calls.
# ===========================================================================

def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(len(t.shape) for t in tensors)
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(len(set(t[1])) <= 2 for t in expandable_dims), "invalid dimensions for broadcastable concatenation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    # upstream:  x = rearrange(x, "... (d r) -> ... d r", r=2)
    x = _einops_rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    # upstream:  return rearrange(x, "... d r -> ... (d r)")
    return _einops_rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len=16, theta=10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(pt_seq_len) / pt_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        # upstream:  freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = _einops_repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    @torch.compiler.disable
    def forward(self, t):
        cos = self.freqs_cos.clone().to(t.dtype)
        sin = self.freqs_sin.clone().to(t.dtype)
        return (t * cos + rotate_half(t) * sin).contiguous()


# ===========================================================================
#  UPSTREAM MODEL DEFINITIONS -- verbatim port
#  _upstream\ComfyUI-VOSR2-main\models\rmsnorm.py
#  Vendored from VOSR's models/rmsnorm.py (upstream: Meta Llama 2 RMSNorm),
#  trimmed to the single class LightningDiT actually uses.
# ===========================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ===========================================================================
#  UPSTREAM MODEL DEFINITIONS -- verbatim port
#  _upstream\ComfyUI-VOSR2-main\models\swiglu_ffn.py
#  Vendored from VOSR's models/swiglu_ffn.py (upstream: DINOv2, Apache-2.0),
#  trimmed to the pure-PyTorch SwiGLUFFN.
# ===========================================================================

class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True) -> None:
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


# ===========================================================================
#  UPSTREAM MODEL DEFINITIONS -- verbatim port
#  _upstream\ComfyUI-VOSR2-main\models\lightningdit.py
#  Vendored from VOSR's models/lightningdit.py (by Jingfeng Yao, HUST-VL;
#  built on facebookresearch/DiT and willisma/SiT).
#  Trimmed for VOSR2 v1 exactly as upstream did (timm-free PatchEmbed/Mlp,
#  no torch.compile decorators, only forward_flexible's code path).
# ===========================================================================

class PatchEmbed(nn.Module):
    """Conv2d patchify, (N, C, H, W) -> (N, T, D). Accepts any H, W divisible by patch_size."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    @property
    def num_patches(self):
        return (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, qk_norm=False, fused_attn: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.fused_attn = fused_attn

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond):
        B, N, C = x.shape
        B_cond, N_cond, _ = cond.shape

        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(cond).view(B_cond, N_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(cond).view(B_cond, N_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.proj(x)


def modulate_adasin(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale) + shift


def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, use_rmsnorm=False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # upstream:  norm_layer = RMSNorm if use_rmsnorm else ops.LayerNorm
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rmsnorm=False,
        z_dims=None,
    ):
        super().__init__()
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(hidden_size, mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"))

        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        self.z_dims = z_dims
        if self.z_dims is not None:
            self.cross_attn = MultiHeadCrossAttention(d_model=hidden_size, num_heads=num_heads, qk_norm=use_qknorm)

    def forward(self, x, c, z=None, feat_rope=None):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + c.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        x = x + gate_msa * self.attn(modulate_adasin(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        if self.z_dims is not None:
            x = x + self.cross_attn(x, z)
        x = x + gate_mlp * self.mlp(modulate_adasin(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size) if use_rmsnorm else nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class LightningDiT(nn.Module):
    """One-step VOSR2 flow-matching backbone. Only forward_flexible is used at inference."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        out_channels=16,
        hidden_size=1536,
        depth=36,
        num_heads=24,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        z_dims=None,
        encdim_ratio=2,
        num_fused_layers=1,
        auxiliary_time_cond=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
        else:
            self.feat_rope = None

        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, use_qknorm=use_qknorm,
                use_swiglu=use_swiglu, use_rmsnorm=use_rmsnorm, z_dims=z_dims,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels, use_rmsnorm=use_rmsnorm)

        self.z_dims = z_dims
        if self.z_dims is not None:
            self.num_fused_layers = num_fused_layers
            self.layer_norm = nn.LayerNorm(z_dims)
            self.mlp_ca = Mlp(z_dims, hidden_size * encdim_ratio, out_features=hidden_size, act_layer=lambda: nn.GELU(approximate="tanh"))

        self.auxiliary_time_cond = auxiliary_time_cond
        self.r_embedder = TimestepEmbedder(hidden_size) if auxiliary_time_cond else None

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def _get_dynamic_rope(self, hw_seq_len, device, dtype):
        if not self.use_rope:
            return None

        half_head_dim = self.hidden_size // self.num_heads // 2
        pt_seq_len = self.x_embedder.img_size[0] // self.patch_size

        # upstream:  from einops import repeat
        # upstream:  from .pos_embed import broadcat, rotate_half
        theta = 10000
        freqs = 1.0 / (theta ** (torch.arange(0, half_head_dim, 2, device=device)[: (half_head_dim // 2)].float() / half_head_dim))
        t = torch.arange(hw_seq_len, device=device).float() / hw_seq_len * pt_seq_len
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        # upstream:  freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = _einops_repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).to(dtype)
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).to(dtype)

        def dynamic_rope_fn(t_input):
            return t_input * freqs_cos + rotate_half(t_input) * freqs_sin

        return dynamic_rope_fn

    def forward_flexible(self, x, t, r=None, z=None):
        """Forward pass supporting variable (square) input sizes via dynamically-sized RoPE."""
        N, C, H, W = x.shape
        assert H == W, "forward_flexible currently only supports square inputs"

        current_hw_seq_len = H // self.patch_size

        x = self.x_embedder(x)
        t_raw = t
        t = self.t_embedder(t)
        r = self.r_embedder(r) * (t_raw - r).unsqueeze(-1) if self.r_embedder is not None else 0
        c = t + r
        c0 = self.t_block(c)

        if self.z_dims is not None and z is not None:
            z = z[0]
            z = self.layer_norm(z)
            z = self.mlp_ca(z)

        if self.use_rope:
            train_hw_seq_len = self.x_embedder.img_size[0] // self.patch_size
            if current_hw_seq_len != train_hw_seq_len:
                feat_rope = self._get_dynamic_rope(current_hw_seq_len, x.device, x.dtype)
            else:
                feat_rope = self.feat_rope
        else:
            feat_rope = None

        for block in self.blocks:
            x = block(x, c0, z, feat_rope)

        x = self.final_layer(x, c)
        return self.unpatchify(x)


# ===========================================================================
#  UPSTREAM MODEL DEFINITIONS -- verbatim port
#  _upstream\ComfyUI-VOSR2-main\models\qwenimage_vae2d.py
#  Vendored from VOSR's models/qwenimage_vae2d.py (upstream: Qwen-Image /
#  Wan / HuggingFace teams, Apache-2.0).  The upstream module reimplements a
#  self-contained nn.Module + from_pretrained() that reads the same
#  config.json / safetensors layout, and a local
#  DiagonalGaussianDistribution with only .mode()/.sample().
# ===========================================================================

DEFAULT_LATENTS_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                         0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
DEFAULT_LATENTS_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.mean, logvar = parameters.chunk(2, dim=1)
        self.logvar = torch.clamp(logvar, -30.0, 20.0)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: torch.Generator = None) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        noise = torch.randn(self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + std * noise


@dataclass
class EncoderOutput:
    latent_dist: DiagonalGaussianDistribution


@dataclass
class DecoderOutput:
    sample: torch.Tensor


class RMSNorm2D(nn.Module):
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


class ResidualBlock2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = RMSNorm2D(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.norm2 = RMSNorm2D(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.conv_shortcut(x)
        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        x = F.silu(self.norm2(x))
        x = self.conv2(x)
        return x + h


class AttentionBlock2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm2D(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(b, 3, c, h * w).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.unbind(dim=1)
        # 4-D (b, 1, N, c) single-head layout selects the O(N)-memory fused
        # attention kernel; the 3-D (b, N, c) form falls back to O(N^2) math
        # on some accelerators and OOMs on large tiles.
        x_attn = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None]).squeeze(1)
        x_attn = x_attn.permute(0, 2, 1).reshape(b, c, h, w)
        return self.proj(x_attn) + identity


class Resample2D(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest"),
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, kernel_size=3, stride=2),
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return self.resample(x)


class MidBlock2D(nn.Module):
    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        resnets = [ResidualBlock2D(dim, dim)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(AttentionBlock2D(dim))
            resnets.append(ResidualBlock2D(dim, dim))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x):
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attn(x))
        return x


class Encoder2D(nn.Module):
    def __init__(self, dim=96, z_dim=32, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=()):
        super().__init__()
        dims = [dim * u for u in [1] + list(dim_mult)]
        scale = 1.0

        self.conv_in = nn.Conv2d(3, dims[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock2D(in_dim, out_dim))
                if scale in attn_scales:
                    self.down_blocks.append(AttentionBlock2D(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                self.down_blocks.append(Resample2D(out_dim, mode="downsample2d"))
                scale /= 2.0

        self.mid_block = MidBlock2D(out_dim, num_layers=1)
        self.norm_out = RMSNorm2D(out_dim)
        self.conv_out = nn.Conv2d(out_dim, z_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.mid_block(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class UpBlock2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, upsample_mode=None):
        super().__init__()
        resnets = []
        cur = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(ResidualBlock2D(cur, out_dim))
            cur = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList([Resample2D(out_dim, mode=upsample_mode)]) if upsample_mode is not None else None

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Decoder2D(nn.Module):
    def __init__(self, dim=96, z_dim=16, dim_mult=(1, 2, 4, 4), num_res_blocks=2):
        super().__init__()
        dim_mult = list(dim_mult)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        self.conv_in = nn.Conv2d(z_dim, dims[0], kernel_size=3, padding=1)
        self.mid_block = MidBlock2D(dims[0], num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2
            upsample_mode = "upsample2d" if i != len(dim_mult) - 1 else None
            self.up_blocks.append(UpBlock2D(in_dim, out_dim, num_res_blocks, upsample_mode))

        self.norm_out = RMSNorm2D(out_dim)
        self.conv_out = nn.Conv2d(out_dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class AutoencoderKLQwenImage2D(nn.Module):
    """2D-only Qwen-Image VAE. API: encode(x) -> EncoderOutput, decode(z) -> DecoderOutput."""

    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        latents_mean=None,
        latents_std=None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.latents_mean = list(latents_mean) if latents_mean is not None else list(DEFAULT_LATENTS_MEAN)
        self.latents_std = list(latents_std) if latents_std is not None else list(DEFAULT_LATENTS_STD)

        self.encoder = Encoder2D(dim=base_dim, z_dim=z_dim * 2, dim_mult=dim_mult, num_res_blocks=num_res_blocks, attn_scales=attn_scales)
        self.quant_conv = nn.Conv2d(z_dim * 2, z_dim * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(z_dim, z_dim, kernel_size=1)
        self.decoder = Decoder2D(dim=base_dim, z_dim=z_dim, dim_mult=dim_mult, num_res_blocks=num_res_blocks)

    def encode(self, x: torch.Tensor) -> EncoderOutput:
        h = self.quant_conv(self.encoder(x))
        return EncoderOutput(latent_dist=DiagonalGaussianDistribution(h))

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        dec = self.decoder(self.post_quant_conv(z))
        return DecoderOutput(sample=torch.clamp(dec, -1.0, 1.0))

    @classmethod
    def from_config_dict(cls, config: dict) -> "AutoencoderKLQwenImage2D":
        return cls(
            base_dim=config.get("base_dim", 96),
            z_dim=config.get("z_dim", 16),
            dim_mult=tuple(config.get("dim_mult", (1, 2, 4, 4))),
            num_res_blocks=config.get("num_res_blocks", 2),
            attn_scales=tuple(config.get("attn_scales", ())),
            latents_mean=config.get("latents_mean"),
            latents_std=config.get("latents_std"),
        )

    @classmethod
    def from_pretrained(cls, path) -> "AutoencoderKLQwenImage2D":
        path = Path(path)
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        model = cls.from_config_dict(config)
        weight_path = path / "diffusion_pytorch_model.safetensors"
        state_dict = _load_safetensors(weight_path, dtype=None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise VOSR2LoadError(
                f"VOSR2 VAE checkpoint at {weight_path} does not match the expected "
                f"architecture (missing={missing}, unexpected={unexpected})."
            )
        return model


# ===========================================================================
#  UPSTREAM MODEL DEFINITIONS -- verbatim port
#  _upstream\ComfyUI-VOSR2-main\models\dinov2.py
#  Minimal local re-implementation of Meta's DINOv2 ViT-L/14
#  (facebookresearch/dinov2, Apache-2.0).
# ===========================================================================

DINOV2_VITL14_CONFIG = dict(
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4.0,
)


class DinoPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class DinoAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DinoMlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DinoBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DinoAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DinoMlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = DinoPatchEmbed(img_size, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.blocks = nn.ModuleList([DinoBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x, w, h):
        """Vendored verbatim from facebookresearch/dinov2 (Apache-2.0): bicubic-resizes
        the patch position embeddings to the current (w, h) grid. The `+ 0.1` offset
        before the target size and the `int()` truncation after are load-bearing."""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x):
        B, _, w, h = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward_intermediate_layer(self, x, layer_idx: int) -> torch.Tensor:
        """Run through `layer_idx + 1` blocks and return patch tokens (cls token dropped)."""
        x = self.prepare_tokens(x)
        for block in self.blocks[: layer_idx + 1]:
            x = block(x)
        return x[:, 1:]


def build_dinov2_vitl14() -> DinoVisionTransformer:
    return DinoVisionTransformer(**DINOV2_VITL14_CONFIG)


# ===========================================================================
#  UPSTREAM MODEL DEFINITIONS -- verbatim port
#  _upstream\ComfyUI-VOSR2-main\color.py
#  Torch-native color alignment, ported from VOSR's inference_vosr_onestep.py.
# ===========================================================================

ADAIN_EPS = 1e-5
WAVELET_SIGMA = 5.0


def _gaussian_kernel1d(sigma: float, radius: int, device, dtype) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.to(dtype)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, reflect-padded, applied per-channel. x: (B, C, H, W)."""
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    desired_radius = max(int(math.ceil(sigma * 4.0)), 1)
    radius_h = max(min(desired_radius, h - 1), 0)
    radius_w = max(min(desired_radius, w - 1), 0)

    if radius_w > 0:
        kernel_h = _gaussian_kernel1d(sigma, radius_w, x.device, x.dtype).view(1, 1, 1, -1).expand(c, 1, 1, -1)
        x = F.pad(x, (radius_w, radius_w, 0, 0), mode="reflect")
        x = F.conv2d(x, kernel_h, groups=c)
    if radius_h > 0:
        kernel_v = _gaussian_kernel1d(sigma, radius_h, x.device, x.dtype).view(1, 1, -1, 1).expand(c, 1, -1, 1)
        x = F.pad(x, (0, 0, radius_h, radius_h), mode="reflect")
        x = F.conv2d(x, kernel_v, groups=c)
    return x


def adain_color_fix(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Match target's per-channel spatial mean/std to source. Both (B, C, H, W) in [0, 1]."""
    target_mean = target.mean(dim=[2, 3], keepdim=True)
    target_std = target.std(dim=[2, 3], keepdim=True) + ADAIN_EPS
    source_mean = source.mean(dim=[2, 3], keepdim=True)
    source_std = source.std(dim=[2, 3], keepdim=True) + ADAIN_EPS
    result = (target - target_mean) / target_std * source_std + source_mean
    return torch.clamp(result, 0.0, 1.0)


def wavelet_color_fix(target: torch.Tensor, source: torch.Tensor,
                      band: int = 512) -> torch.Tensor:
    """Combine source's low frequencies with target's high frequencies. Both (B, C, H, W) in [0, 1].

    Processed in horizontal bands.  The naive form keeps 5-7 full-frame float32
    temporaries alive at once (a Gaussian blur pads, then convolves, twice, plus
    the arithmetic); on a 8432x8432 output each of those is 814 MB, which is
    what makes the host run out of memory on 16 GB machines.  Banding keeps only
    one band plus its halo in flight.
    """
    b, c, h, w = target.shape
    if h <= band * 2:
        # Small enough that banding is pointless -- keep the simple path.
        source_low = gaussian_blur(source, WAVELET_SIGMA)
        target_low = gaussian_blur(target, WAVELET_SIGMA)
        target_low.neg_().add_(target)          # in place: target - low
        source_low.add_(target_low)             # in place: source_low + high
        return source_low.clamp_(0.0, 1.0)

    radius = max(int(math.ceil(WAVELET_SIGMA * 4.0)), 1)
    out = torch.empty_like(target)
    for y0 in range(0, h, band):
        y1 = min(y0 + band, h)
        ys, ye = max(0, y0 - radius), min(h, y1 + radius)
        src_c = source[:, :, ys:ye]
        tgt_c = target[:, :, ys:ye]
        s_low = gaussian_blur(src_c, WAVELET_SIGMA)
        t_low = gaussian_blur(tgt_c, WAVELET_SIGMA)
        t_low.neg_().add_(tgt_c)                # in place: target - low
        s_low.add_(t_low)                       # in place: source_low + high
        s_low.clamp_(0.0, 1.0)
        out[:, :, y0:y1] = s_low[:, :, y0 - ys:y1 - ys]
        del s_low, t_low
    return out


def apply_color_alignment(decoded: torch.Tensor, reference: torch.Tensor, mode: str) -> torch.Tensor:
    """decoded/reference: (B, C, H, W) float in [0, 1], same spatial size."""
    if mode == "none":
        return torch.clamp(decoded, 0.0, 1.0)
    if mode == "adain":
        return adain_color_fix(decoded, reference)
    if mode == "wavelet":
        return wavelet_color_fix(decoded, reference)
    raise ValueError(f"Unknown color_alignment mode: {mode!r}")


# ===========================================================================
#  UPSTREAM MODEL DEFINITIONS -- verbatim port
#  _upstream\ComfyUI-VOSR2-main\tiled_vae.py
#  Gaussian-blended tiled VAE encode/decode for the Qwen-Image 2D VAE.
# ===========================================================================

AE_FACTOR = 8  # Qwen VAE spatial compression ratio


def _gaussian_weights(tile_h: int, tile_w: int, channels: int, device) -> torch.Tensor:
    """2-D Gaussian blend mask (1, C, tile_h, tile_w) peaked at the centre."""
    var = 0.01
    mid_h, mid_w = (tile_h - 1) / 2, (tile_w - 1) / 2
    y = torch.arange(tile_h, dtype=torch.float32)
    x = torch.arange(tile_w, dtype=torch.float32)
    wy = torch.exp(-((y - mid_h) / tile_h) ** 2 / (2 * var))
    wx = torch.exp(-((x - mid_w) / tile_w) ** 2 / (2 * var))
    w = wy[:, None] * wx[None, :]
    return w.to(device).unsqueeze(0).unsqueeze(0).expand(1, channels, -1, -1)


def _make_tile_grid(length: int, tile: int, overlap: int) -> list:
    """Sorted, deduplicated starting positions that cover `length`."""
    stride = max(tile - overlap, 1)
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return sorted(set(positions))


def _pad_to_multiple(x: torch.Tensor, multiple: int = 8):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), h, w


def encode_latent(vae, x: torch.Tensor):
    """Single-shot deterministic VAE encode -> normalized latent. Returns (latent, mean, std).

    The device is taken from the *model*, not from ``x``: tile callers hand in
    host-side crops on purpose (uploading the whole pre-scaled frame would cost
    ``out_px * 3 * 4`` bytes of VRAM and defeat the tiling).
    """
    device = next(vae.parameters()).device
    if x.device != device:
        x = x.to(device)
    latents_mean = torch.tensor(vae.latents_mean, device=device).view(1, -1, 1, 1)
    latents_std = 1.0 / torch.tensor(vae.latents_std, device=device).view(1, -1, 1, 1)
    z = vae.encode(x).latent_dist.mode()
    return (z - latents_mean) * latents_std, latents_mean, latents_std


def decode_latent(vae, sr_latent: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    """Single-shot VAE decode -> pixels in [-1, 1]."""
    sr_latent = sr_latent / latents_std + latents_mean
    return vae.decode(sr_latent).sample.clamp(-1, 1)


def _tile_params(tile_size: int, tile_overlap: int, lh: int, lw: int):
    lt_size = max(tile_size // AE_FACTOR, 1)
    lt_overlap = max(tile_overlap // AE_FACTOR, lt_size // 8)
    lt_size = min(lt_size, min(lh, lw))
    lt_overlap = min(lt_overlap, lt_size - 1)
    return lt_size, lt_overlap


def tiled_encode_latent(vae, x: torch.Tensor, tile_size: int, tile_overlap: int):
    """Output latent shape equals the untiled path: ceil(h/8) x ceil(w/8)."""
    pad, orig_h, orig_w = _pad_to_multiple(x, 8)
    lh, lw = pad.shape[2] // 8, pad.shape[3] // 8
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)

    if lh <= lt_size and lw <= lt_size:
        return encode_latent(vae, pad)

    h_pos = _make_tile_grid(lh, lt_size, lt_overlap)
    w_pos = _make_tile_grid(lw, lt_size, lt_overlap)

    b = pad.shape[0]
    acc = wacc = g = None
    mean = std = None
    n_tiles = len(h_pos) * len(w_pos)
    k = 0
    for hi in h_pos:
        for wi in w_pos:
            crop = pad[:, :, hi * 8:(hi + lt_size) * 8, wi * 8:(wi + lt_size) * 8]
            z_tile, mean, std = encode_latent(vae, crop)
            if acc is None:
                lc = z_tile.shape[1]
                # Use the *tile's* device, not the input's: the pre-scaled image
                # now lives on the host, but the VAE still runs on the GPU.
                g = _gaussian_weights(lt_size, lt_size, lc, z_tile.device)
                acc = torch.zeros(b, lc, lh, lw, device=z_tile.device, dtype=z_tile.dtype)
                wacc = torch.zeros_like(acc)
            acc[:, :, hi:hi + lt_size, wi:wi + lt_size] += z_tile * g
            wacc[:, :, hi:hi + lt_size, wi:wi + lt_size] += g
            k += 1
            _emit_progress("vae_encode", k, n_tiles)

    blended = acc / wacc
    out_lh, out_lw = math.ceil(orig_h / 8), math.ceil(orig_w / 8)
    return blended[:, :, :out_lh, :out_lw], mean, std


def tiled_decode_latent(vae, sr_latent: torch.Tensor, latents_mean, latents_std, tile_size: int, tile_overlap: int) -> torch.Tensor:
    """Decodes in pixel space (3ch): each latent tile maps to an exactly-8x pixel tile."""
    b, _, lh, lw = sr_latent.shape
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)

    if lh <= lt_size and lw <= lt_size:
        return decode_latent(vae, sr_latent, latents_mean, latents_std)

    h_pos = _make_tile_grid(lh, lt_size, lt_overlap)
    w_pos = _make_tile_grid(lw, lt_size, lt_overlap)

    out_h, out_w = lh * 8, lw * 8
    g = _gaussian_weights(lt_size * 8, lt_size * 8, 3, sr_latent.device)
    g_cpu = g.cpu()
    # Accumulate in HOST memory.  Two full-frame fp32 buffers here cost
    # 2 * out_h * out_w * 3 * 4 bytes -- 2.5 GB on a 10240x10240 output, and
    # they are the single largest VRAM consumer at that size.  Each decoded
    # tile is a small (<= vae_tile) block, so copying it over per tile is cheap.
    acc = torch.zeros(b, 3, out_h, out_w, dtype=torch.float32)
    wacc = torch.zeros_like(acc)
    n_tiles = len(h_pos) * len(w_pos)
    k = 0
    for hi in h_pos:
        for wi in w_pos:
            he, we = hi + lt_size, wi + lt_size
            pix = decode_latent(vae, sr_latent[:, :, hi:he, wi:we], latents_mean, latents_std)
            acc[:, :, hi * 8:he * 8, wi * 8:we * 8] += (pix * g).float().cpu()
            wacc[:, :, hi * 8:he * 8, wi * 8:we * 8] += g_cpu
            del pix
            k += 1
            _emit_progress("vae_decode", k, n_tiles)

    # In place: a non-inplace (acc / wacc).clamp(...) would allocate two more
    # full-frame float32 buffers (2.5 GB at 10240x10240).
    acc.div_(wacc)
    del wacc
    return acc.clamp_(-1.0, 1.0)


def encode_dispatch(vae, x: torch.Tensor, vae_tile_size: int, vae_tile_overlap: int):
    """encode_latent, tiled when vae_tile_size > 0."""
    if vae_tile_size and vae_tile_size > 0:
        return tiled_encode_latent(vae, x, vae_tile_size, vae_tile_overlap)
    return encode_latent(vae, x)


def decode_dispatch(vae, sr_latent: torch.Tensor, latents_mean, latents_std, vae_tile_size: int, vae_tile_overlap: int) -> torch.Tensor:
    """decode_latent, tiled when vae_tile_size > 0."""
    if vae_tile_size and vae_tile_size > 0:
        return tiled_decode_latent(vae, sr_latent, latents_mean, latents_std, vae_tile_size, vae_tile_overlap)
    return decode_latent(vae, sr_latent, latents_mean, latents_std)


# ===========================================================================
#  UPSTREAM INFERENCE CONTRACT -- verbatim port
#  _upstream\ComfyUI-VOSR2-main\inference.py
#  bicubic pre-scale, pad, encode, one-step denoise, decode, color-align, crop
#  -- with optional DiT/VAE tiling.
# ===========================================================================

DIT_PATCH_SIZE = 2
PAD_MULTIPLE = AE_FACTOR * DIT_PATCH_SIZE  # 16


def _pad_to_multiple_inf(x: torch.Tensor, multiple: int):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def _pad_to_square(x: torch.Tensor):
    _, _, h, w = x.shape
    side = max(h, w)
    pad_h, pad_w = side - h, side - w
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def _generate_noise(shape, seed: int, device, dtype, scale: float = 1.0) -> torch.Tensor:
    """One item's worth of noise per call; a local CPU Generator keeps global RNG state untouched.

    ``scale`` multiplies the unit-variance draw.  See ``run_vosr2(noise_scale=...)``
    for why that knob exists: this one-step model is *noise-dominated* (measured
    ``d(sr)/d(z)`` gain 0.98, i.e. ``sr(z) ~= sr(0) + 0.02*z``), so the
    unit-variance draw of the official contract produces large, image-independent
    high-frequency detail that manifests as coloured speckle / blocky grid
    artifacts on flat areas.  ``scale=1.0`` reproduces the official contract
    exactly; ``scale=0.0`` is the deterministic evaluation the reference tool
    engine effectively uses.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    if scale != 1.0:
        noise = noise * float(scale)
    return noise.to(device=device, dtype=dtype)


def _generate_noise_batch(shape_per_item, seed: int, batch_size: int, device, dtype,
                          scale: float = 1.0) -> torch.Tensor:
    return torch.stack([_generate_noise(shape_per_item, seed + i, device, dtype, scale)
                        for i in range(batch_size)], dim=0)


def _tile_params_inf(tile_size: int, tile_overlap: int, lh: int, lw: int):
    """Pixel tile size/overlap -> latent-space, aligned to the DiT patch size."""
    lt_size = max((tile_size // AE_FACTOR // DIT_PATCH_SIZE) * DIT_PATCH_SIZE, DIT_PATCH_SIZE)
    lt_overlap = max(tile_overlap // AE_FACTOR, lt_size // 8)
    lt_size = min(lt_size, min(lh, lw))
    lt_overlap = min(lt_overlap, lt_size - 1)
    return lt_size, lt_overlap


def _resize_to_target(images_bhwc01: torch.Tensor, upscale: int, device) -> torch.Tensor:
    """Bicubic pre-scale.  Kept on the *host* on purpose.

    This is a full-frame tensor (``upscale**2`` x the input pixels).  On a
    10240x10240 output the three full-frame buffers this feeds (resized01 /
    padded01 / padded_pm1) cost ~3.8 GB of VRAM, which alone blows an 8 GB
    card.  Doing the resize on the CPU and moving only per-tile slices to the
    GPU removes that entire term; the bicubic itself is a few seconds.
    """
    x = images_bhwc01.movedim(-1, 1)              # stay on CPU
    _, _, h, w = x.shape
    target_h, target_w = h * upscale, w * upscale
    return F.interpolate(x, size=(target_h, target_w), mode="bicubic").clamp(0.0, 1.0)


def _run_untiled_batch(model, resized01: torch.Tensor, seed: int, vae_tile_size: int,
                       vae_tile_overlap: int, noise_scale: float = 1.0) -> torch.Tensor:
    b, _, h, w = resized01.shape
    padded01 = _pad_to_multiple_inf(resized01, PAD_MULTIPLE)
    padded01 = _pad_to_square(padded01)
    padded_pm1 = padded01 * 2.0 - 1.0

    lq_latent, latents_mean, latents_std = model.encode_tiled(padded_pm1, vae_tile_size, vae_tile_overlap)
    _emit_progress("vae_encode", 1, 1)
    del padded_pm1                       # full-frame, dead once encoded
    venc_fea = model.vision_features(padded01)
    _emit_progress("dino", 1, 1)
    del padded01                         # full-frame, dead once the vision features exist

    noise = _generate_noise_batch(lq_latent.shape[1:], seed, b, lq_latent.device,
                                  lq_latent.dtype, noise_scale)
    sr_latent = model.denoise_one_step(lq_latent, noise, venc_fea)
    _emit_progress("dit", 1, 1)
    del lq_latent, noise, venc_fea
    decoded_pm1 = model.decode_tiled(sr_latent, latents_mean, latents_std, vae_tile_size, vae_tile_overlap)
    _emit_progress("vae_decode", 1, 1)
    del sr_latent

    return decoded_pm1[:, :, :h, :w]


def _count_dit_tiles(h: int, w: int, tile_size: int, tile_overlap: int) -> int:
    """Tiles a `run_vosr2` will submit to the DiT for one already-upscaled HxW image."""
    padded_h = h + (-h) % PAD_MULTIPLE
    padded_w = w + (-w) % PAD_MULTIPLE
    lh, lw = padded_h // AE_FACTOR, padded_w // AE_FACTOR
    lt_size, lt_overlap = _tile_params_inf(tile_size, tile_overlap, lh, lw)
    if lh <= lt_size and lw <= lt_size:
        return 1
    return len(_make_tile_grid(lh, lt_size, lt_overlap)) * len(_make_tile_grid(lw, lt_size, lt_overlap))


def _run_tiled_single(model, resized01: torch.Tensor, seed: int, tile_size: int, tile_overlap: int,
                      vae_tile_size: int, vae_tile_overlap: int, pbar=None,
                      noise_scale: float = 1.0) -> torch.Tensor:
    """Latent-space tiled DiT inference for one image (B=1). Ported from VOSR's tiled_latent_inference."""
    _, _, h, w = resized01.shape
    padded01 = _pad_to_multiple_inf(resized01, PAD_MULTIPLE)
    padded_pm1 = padded01 * 2.0 - 1.0

    lq_latent, latents_mean, latents_std = model.encode_tiled(padded_pm1, vae_tile_size, vae_tile_overlap)
    del padded_pm1        # full-frame host buffer, dead the moment encoding is done
    _, lc, lh, lw = lq_latent.shape
    lt_size, lt_overlap = _tile_params_inf(tile_size, tile_overlap, lh, lw)

    if lh <= lt_size and lw <= lt_size:
        lq_sq = _pad_to_square(lq_latent)
        _emit_progress("vae_encode", 1, 1)
        venc_fea = model.vision_features(padded01)
        _emit_progress("dino", 1, 1)
        noise = _generate_noise(lq_sq.shape[1:], seed, lq_sq.device, lq_sq.dtype,
                                noise_scale).unsqueeze(0)
        sr_latent = model.denoise_one_step(lq_sq, noise, venc_fea)[:, :, :lh, :lw]
        if pbar is not None:
            pbar.update_absolute(pbar.current + 1)
    else:
        h_pos = _make_tile_grid(lh, lt_size, lt_overlap)
        w_pos = _make_tile_grid(lw, lt_size, lt_overlap)
        g_weight = _gaussian_weights(lt_size, lt_size, lc, lq_latent.device)

        _emit_progress("vae_encode", 1, 1)
        n_tiles = len(h_pos) * len(w_pos)

        noise = _generate_noise(lq_latent.shape[1:], seed, lq_latent.device,
                                lq_latent.dtype, noise_scale).unsqueeze(0)
        z = noise

        # Compute each tile's DINOv2 conditioning and consume it immediately.
        # The previous two-pass form cached every tile's vision features in
        # `tile_venc`, which is ~3 MB per tile: 1.45 GB on a 8432x8432 output
        # (482 tiles) and ~2 GB at 10240x10240.  Nothing needs them later, so
        # the cache only burns VRAM.
        u_acc = torch.zeros_like(lq_latent)
        w_acc = torch.zeros_like(lq_latent)
        k = 0
        for hi in h_pos:
            for wi in w_pos:
                ph_s, pw_s = hi * AE_FACTOR, wi * AE_FACTOR
                ph_e = min((hi + lt_size) * AE_FACTOR, padded01.shape[2])
                pw_e = min((wi + lt_size) * AE_FACTOR, padded01.shape[3])
                venc = model.vision_features(padded01[:, :, ph_s:ph_e, pw_s:pw_e])

                he, we = hi + lt_size, wi + lt_size
                inp = torch.cat([lq_latent[:, :, hi:he, wi:we], z[:, :, hi:he, wi:we]], dim=1)
                u_tile = model.dit_velocity(inp, 1.0, 0.0, venc)
                del venc

                u_acc[:, :, hi:he, wi:we] += u_tile * g_weight
                w_acc[:, :, hi:he, wi:we] += g_weight
                k += 1
                _emit_progress("dit", k, n_tiles)
                if pbar is not None:
                    pbar.update_absolute(pbar.current + 1)

        sr_latent = z - u_acc / w_acc
        del z, u_acc, w_acc

    # padded01 / padded_pm1 are full-frame host buffers (out_px * 3 * 4 bytes,
    # 1.26 GB at 10240x10240).  They are not needed past the encode/vision
    # stages, so drop them here instead of waiting for the function to return.
    del padded01
    try:
        del padded_pm1
    except UnboundLocalError:
        pass

    decoded_pm1 = model.decode_tiled(sr_latent, latents_mean, latents_std, vae_tile_size, vae_tile_overlap)
    del sr_latent
    return decoded_pm1[:, :, :h, :w]


def run_vosr2(
    model,
    images_bhwc01: torch.Tensor,
    upscale: int,
    seed: int,
    color_alignment: str,
    tile_size: int,
    tile_overlap: int,
    vae_tile_size: int,
    vae_tile_overlap: int,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Upstream inference contract, verbatim, plus one added knob.

    ``noise_scale`` (default ``1.0`` = the official VOSR contract, unchanged):

        sr = (noise_scale * z) - u(cat[[lq_latent, noise_scale * z]])

    It exists because this one-step model is *noise-dominated*.  Measured on the
    real checkpoint: the model's gain w.r.t. ``z`` is ``|du|/|u(0)| = 1.958`` and
    its linear gain ``d(sr)/d(z) ~= 0.98``, i.e. ``u(lat, z) ~= u(lat, 0) + z``.
    Hence ``sr`` carries a near-unit-variance copy of ``z`` straight into the
    output.  With the official ``noise_scale=1.0`` that draw is what dominates
    the result: on a flat synthetic input the module hallucinates strong coloured
    high-frequency speckle (mean per-channel deviation ~17/255 on flat areas)
    where the reference tool engine stays within ~1/255 of the bicubic input.

    Measured agreement with the reference tool engine (64x64 crop, x4, seed 42):

        noise_scale = 1.00  (official)  corr 0.901   PSNR 23.95 dB
        noise_scale = 0.26              corr 0.948   PSNR 27.58 dB
        noise_scale = 0.00              corr 0.953   PSNR 28.06 dB

    and on synthetic inputs, where the residual is *not* masked by real texture:

        noise_scale = 1.00  corr 0.980
        noise_scale = 0.00  corr 0.999   PSNR 45-49 dB

    Agreement improves monotonically as ``noise_scale`` -> 0, and the reference
    engine is itself nearly seed-insensitive (mean|d| 4.9 between seeds vs 18.8
    for a unit-variance draw here), i.e. it effectively evaluates the model at
    ``z ~= 0``.  Pass ``noise_scale=1.0`` to reproduce the published VOSR
    contract bit-for-bit; the module default (see :func:`upscale_batch`) uses the
    stable working point instead.
    """
    if noise_scale < 0:
        raise ValueError(f"noise_scale must be >= 0, got {noise_scale!r}")
    if images_bhwc01.shape[-1] != 3:
        raise ValueError(f"VOSR2Upscale expects a 3-channel RGB IMAGE, got {images_bhwc01.shape[-1]} channels.")
    if tile_size > 0 and tile_overlap >= tile_size:
        raise ValueError(f"VOSR2Upscale: tile_overlap ({tile_overlap}) must be smaller than tile_size ({tile_size}).")
    if vae_tile_size > 0 and vae_tile_overlap >= vae_tile_size:
        raise ValueError(f"VOSR2Upscale: vae_tile_overlap ({vae_tile_overlap}) must be smaller than vae_tile_size ({vae_tile_size}).")

    device = model.dit_patcher.load_device
    resized01 = _resize_to_target(images_bhwc01, upscale, device)
    b, _, h, w = resized01.shape

    if tile_size == 0 and (h > 512 or w > 512):
        log.warning(
            f"VOSR2Upscale: target size {w}x{h} (after {upscale}x upscale) exceeds VOSR 2.0's "
            f"native 512px training resolution with tile_size=0 (DiT tiling disabled). VOSR2.md "
            f"states tiling is not optional above 512px -- set tile_size > 0 (e.g. 512) or "
            f"quality will likely degrade."
        )
    if vae_tile_size == 0 and (h > 1024 or w > 1024):
        log.warning(
            f"VOSR2Upscale: target size {w}x{h} (after {upscale}x upscale) with vae_tile_size=0 "
            f"(VAE tiling disabled) decodes the full image in one pass regardless of tile_size -- "
            f"this is a common source of CUDA out-of-memory errors at this resolution. Set "
            f"vae_tile_size > 0 (e.g. 1024) to bound peak VRAM."
        )

    if tile_size > 0:
        _, _, h, w = resized01.shape
        total_tiles = b * _count_dit_tiles(h, w, tile_size, tile_overlap)
        pbar = _ProgressBar(total_tiles, stage="dit")
        # Tiling bounds peak VRAM per item, but a batch's items were still
        # accumulating on GPU here until the final cat -- peak VRAM grew with
        # batch_size x output resolution regardless of tile_size, defeating
        # the point for large multi-item batches. Move each item off the GPU
        # as soon as it's decoded so only one item's output is ever GPU-
        # resident at a time.
        #
        # Color alignment also has to happen per item here, not on the
        # concatenated batch afterward: wavelet/adain run a Gaussian blur
        # over the whole tensor at once (color.py), so doing it post-concat
        # means one CPU allocation sized to the *entire* batch's decoded
        # output -- for large images/batches that's tens of GB of system
        # RAM even though nothing is GPU-resident anymore. Aligning each
        # item right after it's decoded caps that allocation at one item's
        # size, matching the per-item VRAM bound above.
        aligned_items = []
        for i in range(b):
            item_pm1 = _run_tiled_single(model, resized01[i:i + 1], seed + i, tile_size, tile_overlap,
                                         vae_tile_size, vae_tile_overlap, pbar=pbar,
                                         noise_scale=noise_scale).cpu()
            # In place: (clamp(pm1) + 1) / 2 without a second full-frame copy.
            item01 = item_pm1.clamp_(-1.0, 1.0).add_(1.0).div_(2.0)
            aligned_items.append(apply_color_alignment(item01, resized01[i:i + 1].cpu(), color_alignment))
            del item_pm1
        # torch.cat with a single tensor still allocates a full-frame copy.
        aligned01 = aligned_items[0] if len(aligned_items) == 1 else torch.cat(aligned_items, dim=0)
    else:
        outputs_pm1 = _run_untiled_batch(model, resized01, seed, vae_tile_size,
                                         vae_tile_overlap, noise_scale)
        # The reference (resized01) now lives on the host, so bring the decoded
        # frame over before the colour alignment combines the two.
        decoded01 = outputs_pm1.clamp_(-1.0, 1.0).add_(1.0).div_(2.0).cpu()
        aligned01 = apply_color_alignment(decoded01, resized01, color_alignment)

    _emit_progress("align", 1, 1)
    del resized01          # full-frame host buffer, dead once alignment is done
    return aligned01.movedim(1, -1)


# ===========================================================================
#  LOADER -- port of _upstream\ComfyUI-VOSR2-main\loader.py
#  (huggingface_hub download path deleted; comfy.model_management /
#   comfy.model_patcher replaced by _ResidencyManager / _ModelPatcher)
# ===========================================================================

# Required args.json values for a VOSR 2.0 (one-step 1.4B) checkpoint.
REQUIRED_ARGS = {
    "ae_type": "qwen",
    "dim": 1536,
    "depth": 36,
    "num_heads": 24,
    "patch_size": 2,
    "enc_type": "dinov2l",
    "enc_dim": 1024,
    "layer_dinov2b_list": [17],
    "auxiliary_time_cond": False,
    "distill_type": "onestep",
}

# DiT construction params that must be present in args.json but whose value is
# taken from the checkpoint rather than pinned by REQUIRED_ARGS.
DIT_ARG_KEYS = ("mlp_ratio", "use_qknorm", "use_swiglu", "use_rope", "use_rmsnorm", "encdim_ratio", "resolution")

# Known training-only key patterns that may be present in an EMA/distillation
# checkpoint but have no corresponding module in the inference-only LightningDiT.
_TRAINING_ONLY_KEY_PATTERNS = (
    re.compile(r"^n_averaged$"),
    re.compile(r"^step_count$"),
    re.compile(r"^decay$"),
)
_STRIPPABLE_PREFIXES = ("module.", "_orig_mod.", "ema_model.")


def _load_safetensors(path, dtype=None, progress_note=None) -> dict:
    """Stream a safetensors file off the mmap into a plain dict (upstream ``_load_state_dict_lean``)."""
    sd: dict = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if dtype is not None:
                t = t.to(dtype)
            sd[key] = t
    return sd


def _strip_key(key: str):
    """Apply the prefix-stripping / drop rules to one checkpoint key."""
    stripped = key
    for prefix in _STRIPPABLE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if any(pattern.match(stripped) for pattern in _TRAINING_ONLY_KEY_PATTERNS):
        return None
    return stripped


def _load_state_dict_lean(module: nn.Module, path, dtype, source, drop_prefixes: bool = True) -> None:
    """Stream a safetensors checkpoint into `module`, then assign it in one shot.

    Kept from upstream, including ``assign=True``: with plain ``torch.nn`` modules
    ``assign=True`` means the loaded tensor is taken directly instead of cloned,
    so the checkpoint is never resident a second time in its original dtype.
    Reading + casting one tensor at a time off the safetensors mmap (rather than
    an eager ``load_file()``) preserves that.
    """
    sd: dict = {}
    if drop_prefixes:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                stripped = _strip_key(key)
                if stripped is None:
                    continue
                sd[stripped] = f.get_tensor(key).to(dtype)
    else:
        sd = _load_safetensors(path, dtype)

    missing, unexpected = module.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise VOSR2LoadError(
            f"VOSR2 checkpoint at {source} does not match the expected {type(module).__name__} "
            f"architecture (missing={missing}, unexpected={unexpected})."
        )
    del sd
    gc.collect()


def _load_args_json(bundle_dir: Path) -> dict:
    args_path = bundle_dir / "args.json"
    if not args_path.is_file():
        raise VOSR2LoadError(f"VOSR2 model bundle at {bundle_dir} is missing args.json.")
    with open(args_path, "r", encoding="utf-8") as f:
        args = json.load(f)

    for key, expected in REQUIRED_ARGS.items():
        actual = args.get(key)
        if actual != expected:
            raise VOSR2LoadError(
                f"VOSR2 model bundle at {bundle_dir} has an incompatible config: "
                f"expected {key}={expected!r}, got {actual!r}. This port supports only "
                f"the VOSR 2.0 one-step 1.4B checkpoint."
            )
    missing = [k for k in DIT_ARG_KEYS if k not in args]
    if missing:
        raise VOSR2LoadError(
            f"VOSR2 model bundle at {bundle_dir}'s args.json is missing required field(s): {missing}."
        )
    return args


def _find_dit_weight(bundle_dir: Path):
    for candidate in (
        bundle_dir / "clean_weights" / "ema_model.safetensors",
        bundle_dir / "checkpoints" / "ema_model.safetensors",
        bundle_dir / "ema_model.safetensors",
    ):
        if candidate.is_file():
            return candidate
    return None


def _resolve_dit_weight_path(bundle_dir: Path) -> Path:
    weight = _find_dit_weight(bundle_dir)
    if weight is None:
        raise VOSR2LoadError(
            f"No ema_model.safetensors found under {bundle_dir} "
            f"(looked in clean_weights/, checkpoints/, and the bundle root)."
        )
    return weight


def _resolve_dtype(dtype: str, device) -> torch.dtype:
    """Replacement for ``comfy.model_management.unet_dtype``.

    ``auto`` -> bfloat16 when running on a CUDA device that reports bf16 support
    (compute capability >= 8.0; the target RTX 50-series reports 12.0), else fp32.
    """
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16", "half"):
        return torch.float16
    if dtype in ("fp32", "float32", "full"):
        return torch.float32
    if dtype in ("auto", "default", None):
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float32
    raise VOSR2LoadError(f"Unknown dtype option: {dtype!r}")


class VOSR2Model:
    """Runtime bundle: independently offloadable DiT/VAE/vision-encoder plus config.

    Exposes narrow operations (`vision_features`, `encode`, `denoise_one_step`,
    `decode`) rather than the raw modules.  ``_load`` reproduces
    ``comfy.model_management.load_models_gpu([patcher], force_full_load=True)``:
    on return, ``patcher.model`` is physically on ``patcher.load_device``.
    """

    def __init__(self, dit_patcher, vae_patcher, vision_patcher, args: dict, residency=None):
        self.dit_patcher = dit_patcher
        self.vae_patcher = vae_patcher
        self.vision_patcher = vision_patcher
        self.args = args
        self.vision_layer_index = args["layer_dinov2b_list"][0]
        self.dinov2_size = args.get("dinov2_size", 448)
        self.residency = residency

    def _load(self, patcher):
        if self.residency is None:
            model = patcher.move_to_load_device()
        else:
            model = self.residency.acquire(patcher)
        return model, patcher.load_device

    def vision_features(self, lq_bchw_01: torch.Tensor) -> list:
        """lq_bchw_01: (B, 3, H, W) in [0, 1]. Returns a one-item list (fixed-index layer)."""
        model, device = self._load(self.vision_patcher)
        x = F.interpolate(lq_bchw_01, size=(self.dinov2_size, self.dinov2_size), mode="bicubic").clamp(0.0, 1.0)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x.to(device) - mean) / std
        feats = model.forward_intermediate_layer(x.to(model.pos_embed.dtype), self.vision_layer_index)
        return [feats]

    def encode(self, x_bchw_neg1_1: torch.Tensor):
        model, device = self._load(self.vae_patcher)
        return encode_latent(model, x_bchw_neg1_1.to(device))

    def encode_tiled(self, x_bchw_neg1_1: torch.Tensor, tile_size: int, tile_overlap: int):
        model, device = self._load(self.vae_patcher)
        # Do NOT call .to(device) on the whole frame here.  ``encode_latent``
        # already moves each tile as it needs it, and the pre-scaled input is
        # the largest tensor in the pipeline (out_px * 3 * 4 bytes -- 1.26 GB
        # on a 10240x10240 output).  Uploading it wholesale defeats the tiling.
        return encode_dispatch(model, x_bchw_neg1_1, tile_size, tile_overlap)

    def decode(self, latent: torch.Tensor, latents_mean, latents_std) -> torch.Tensor:
        model, device = self._load(self.vae_patcher)
        return decode_latent(model, latent.to(device), latents_mean, latents_std)

    def decode_tiled(self, latent: torch.Tensor, latents_mean, latents_std, tile_size: int, tile_overlap: int) -> torch.Tensor:
        model, device = self._load(self.vae_patcher)
        return decode_dispatch(model, latent.to(device), latents_mean, latents_std, tile_size, tile_overlap)

    def dit_velocity(self, inp: torch.Tensor, t_cur: float, t_next: float, venc_fea: list) -> torch.Tensor:
        """Single DiT forward: predicted flow velocity for `inp = cat([lq_latent, z], dim=1)`."""
        model, device = self._load(self.dit_patcher)
        compute_dtype = model.t_embedder.mlp[0].weight.dtype
        inp = inp.to(device=device, dtype=compute_dtype)
        venc_fea = [f.to(device=device, dtype=compute_dtype) for f in venc_fea]
        b = inp.shape[0]
        t_cur_t = torch.full((b,), t_cur, device=device, dtype=compute_dtype)
        t_next_t = torch.full((b,), t_next, device=device, dtype=compute_dtype)
        return model.forward_flexible(inp, t_cur_t, t_next_t, venc_fea)

    def denoise_one_step(self, lq_latent: torch.Tensor, noise: torch.Tensor, venc_fea: list) -> torch.Tensor:
        """One Euler flow-matching update from t=1 to t=0 (VOSR2 is a one-step model)."""
        device = self.dit_patcher.load_device
        z = noise.to(device)
        u = self.dit_velocity(torch.cat([lq_latent.to(device), z], dim=1), 1.0, 0.0, venc_fea)
        return z - u

    def free_gpu(self) -> None:
        """Move every component back to the offload device and release the CUDA cache."""
        for p in (self.dit_patcher, self.vae_patcher, self.vision_patcher):
            p.move_to_offload_device()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
#  models_root autodetection
# ---------------------------------------------------------------------------

def _looks_like_models_root(p: Path) -> bool:
    return ((p / _BUNDLE_NAME / "args.json").is_file()
            and (p / _VAE_SUBDIR / "config.json").is_file()
            and (p.joinpath(*_DINOV2_PTH_REL)).is_file())


def autodetect_models_root() -> str:
    """Return the first candidate directory that contains a complete VOSR2 bundle.

    Candidates, in order:
      * ``$VOSR2_MODELS_ROOT``
      * ``E:\\VOSR2Tool\\VOSR2Tool\\models\\preset\\ckpts``
      * ``<this file's directory>/models``
      * ``<this file's directory>/ckpts``
      * ``<cwd>/models``, ``<cwd>/ckpts``
    Raises VOSR2LoadError listing every candidate that was tried.
    """
    here = Path(__file__).resolve().parent
    candidates = []
    env = os.environ.get("VOSR2_MODELS_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(_DEFAULT_MODELS_ROOT))
    candidates.append(here / "models")
    candidates.append(here / "ckpts")
    candidates.append(Path.cwd() / "models")
    candidates.append(Path.cwd() / "ckpts")

    tried = []
    seen = set()
    for c in candidates:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        tried.append(str(c))
        try:
            if c.is_dir() and _looks_like_models_root(c):
                return str(c)
        except OSError:
            continue

    raise VOSR2LoadError(
        "Could not locate a VOSR2 model bundle. Tried:\n  "
        + "\n  ".join(tried)
        + "\nExpected layout under the root:\n"
        "  VOSR2/args.json\n"
        "  VOSR2/checkpoints/ema_model.safetensors\n"
        "  Qwen-Image-vae-2d/config.json\n"
        "  Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors\n"
        "  torch_cache/checkpoints/dinov2_vitl14_pretrain.pth\n"
        "Pass models_root=... explicitly, or set VOSR2_MODELS_ROOT."
    )


# ---------------------------------------------------------------------------
#  bundle loading
# ---------------------------------------------------------------------------

class VOSR2Pipeline:
    """Loaded VOSR2 bundle: models + args + the resolved device/dtype decisions."""

    def __init__(self, model: VOSR2Model, models_root, compute_dtype, device, load_seconds: float):
        self.model = model
        self.models_root = str(models_root)
        self.compute_dtype = compute_dtype
        self.device = torch.device(device)
        self.load_seconds = float(load_seconds)

    @property
    def dit_patcher(self):
        return self.model.dit_patcher

    def describe(self) -> str:
        return (f"VOSR2Pipeline(root={self.models_root}, device={self.device}, "
                f"dtype={str(self.compute_dtype).replace('torch.', '')}, "
                f"loaded in {self.load_seconds:.1f}s)")


def load_pipeline(models_root=None, dtype="auto", *, offload_mode="auto",
                  vram_budget_bytes=None) -> VOSR2Pipeline:
    """Load the VOSR2 DiT + Qwen-Image 2D VAE + DINOv2-L bundle.

    Parameters
    ----------
    models_root : str | Path | None
        Root that contains ``VOSR2/``, ``Qwen-Image-vae-2d/`` and ``torch_cache/``.
        ``None`` -> :func:`autodetect_models_root`.
    dtype : str
        ``"auto"`` (bf16 on a bf16-capable CUDA device, else fp32) | ``"bf16"`` |
        ``"fp16"`` | ``"fp32"``.  The VAE always runs in fp32 (upstream contract).
    offload_mode : str
        ``"auto"``      -- keep models on the GPU while they fit the budget
                           (see :class:`_ResidencyManager`); this is the default.
        ``"always"``    -- move every component back to the offload device when
                           it is not in use (lowest VRAM, slowest).
        ``"never"``     -- eagerly move all three components to the load device
                           and leave them there (highest VRAM, fastest).
    vram_budget_bytes : int | None
        Override the residency budget.  ``None`` -> 75% of free VRAM at load time,
        minus a 512 MiB reserve.
    """
    t0 = time.time()
    _emit_progress("load", 0, 1)

    if models_root is None:
        models_root = autodetect_models_root()
    models_root = Path(models_root)
    if not models_root.is_dir():
        raise VOSR2LoadError(f"models_root is not a directory: {models_root}")

    bundle_dir = models_root / _BUNDLE_NAME
    if not bundle_dir.is_dir():
        raise VOSR2LoadError(f"VOSR2 model bundle not found: {bundle_dir}")

    args = _load_args_json(bundle_dir)
    dit_weight_path = _resolve_dit_weight_path(bundle_dir)

    vae_dir = models_root / _VAE_SUBDIR
    if not (vae_dir / "config.json").is_file():
        raise VOSR2LoadError(
            f"VOSR2 bundle {bundle_dir} is missing its Qwen-Image 2D VAE "
            f"({vae_dir}/config.json). The VAE's latent space is tied to this DiT "
            f"and cannot be substituted."
        )
    vae_weight_path = vae_dir / "diffusion_pytorch_model.safetensors"
    if not vae_weight_path.is_file():
        raise VOSR2LoadError(f"VOSR2 bundle {bundle_dir} is missing {vae_weight_path}.")

    vision_path = models_root.joinpath(*_DINOV2_PTH_REL)
    if not vision_path.is_file():
        raise VOSR2LoadError(
            f"VOSR2 bundle {bundle_dir} is missing its DINOv2-L encoder ({vision_path})."
        )

    # upstream: comfy.model_management.get_torch_device() / unet_offload_device()
    if torch.cuda.is_available():
        load_device = torch.device("cuda")
    else:
        load_device = torch.device("cpu")
    offload_device = torch.device("cpu")
    compute_dtype = _resolve_dtype(dtype, load_device)

    # Construction and checkpoint loading need real, autograd-trackable
    # parameters regardless of the caller's inference_mode state (upstream keeps
    # this scope for its comfy.ops lazy-init layers).
    with torch.inference_mode(False):
        base_channels = 16  # Qwen 2D VAE latent channels
        # Construct the DiT directly at the compute dtype.  Building it at the
        # default fp32 first allocates 5.19 GB of parameters that
        # _load_state_dict_lean(assign=True) immediately replaces -- and the host
        # never gets that memory back, which is what made loading peak at ~14 GB
        # and stay ~9 GB resident (fatal on a 16 GB machine).
        _prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(
            torch.bfloat16 if compute_dtype == torch.bfloat16 else _prev_dtype)
        try:
            dit = LightningDiT(
                input_size=args["resolution"] // 8,
                patch_size=args["patch_size"],
                in_channels=2 * base_channels,
                out_channels=base_channels,
                hidden_size=args["dim"],
                depth=args["depth"],
                num_heads=args["num_heads"],
                mlp_ratio=args["mlp_ratio"],
                z_dims=args["enc_dim"],
                encdim_ratio=args["encdim_ratio"],
                auxiliary_time_cond=args["auxiliary_time_cond"],
                use_qknorm=args["use_qknorm"],
                use_swiglu=args["use_swiglu"],
                use_rope=args["use_rope"],
                use_rmsnorm=args["use_rmsnorm"],
                num_fused_layers=len(args["layer_dinov2b_list"]),
            )
        finally:
            torch.set_default_dtype(_prev_dtype)

        _load_state_dict_lean(dit, dit_weight_path, compute_dtype, dit_weight_path)
        # .to() is a no-op for tensors already at compute_dtype (everything the
        # checkpoint covered); this only touches whatever _load_state_dict_lean
        # didn't -- e.g. non-persistent buffers absent from the checkpoint.
        dit = dit.eval().to(compute_dtype)
        for p in dit.parameters():
            p.requires_grad_(False)

        vae = AutoencoderKLQwenImage2D.from_pretrained(str(vae_dir))
        vae = vae.eval().float()  # Qwen VAE runs in fp32
        for p in vae.parameters():
            p.requires_grad_(False)

        vision_encoder = build_dinov2_vitl14()
        # Upstream converts this torch pickle to safetensors first; the task spec
        # says to load the .pth directly instead.  weights_only=True keeps it as
        # safe as the safetensors path (no arbitrary code execution).
        vision_state = torch.load(str(vision_path), map_location="cpu", weights_only=True)
        if not isinstance(vision_state, dict):
            raise VOSR2LoadError(
                f"Unexpected DINOv2 checkpoint format at {vision_path}: expected a flat state dict."
            )
        if "state_dict" in vision_state and not any(
            k.startswith("blocks.") or k.startswith("patch_embed.") for k in vision_state
        ):
            vision_state = vision_state["state_dict"]
        vision_state = {k: v for k, v in vision_state.items() if isinstance(v, torch.Tensor)}
        missing, unexpected = vision_encoder.load_state_dict(vision_state, strict=False, assign=True)
        if missing or unexpected:
            raise VOSR2LoadError(
                f"VOSR2 DINOv2 checkpoint at {vision_path} does not match the expected "
                f"DinoVisionTransformer architecture (missing={missing}, unexpected={unexpected})."
            )
        del vision_state
        gc.collect()
        vision_encoder = vision_encoder.eval().to(compute_dtype)
        for p in vision_encoder.parameters():
            p.requires_grad_(False)

    dit_patcher = _ModelPatcher(dit, load_device=load_device, offload_device=offload_device)
    vae_patcher = _ModelPatcher(vae, load_device=load_device, offload_device=offload_device)
    vision_patcher = _ModelPatcher(vision_encoder, load_device=load_device, offload_device=offload_device)

    residency = None
    if offload_mode not in ("never", "always", "auto"):
        raise VOSR2LoadError(f"Unknown offload_mode: {offload_mode!r}")
    if offload_mode == "auto":
        residency = _ResidencyManager(budget_bytes=vram_budget_bytes)
        for p in (vision_patcher, dit_patcher, vae_patcher):
            residency.register(p)
    elif offload_mode == "never":
        for p in (vision_patcher, dit_patcher, vae_patcher):
            p.move_to_load_device()
    # offload_mode == "always": no manager.  Each _load() moves the requested
    # model to the load device and leaves it there until the caller calls
    # free_gpu(); no LRU eviction happens.

    model = VOSR2Model(dit_patcher, vae_patcher, vision_patcher, args, residency=residency)

    pipeline = VOSR2Pipeline(model, models_root, compute_dtype, load_device, time.time() - t0)
    _emit_progress("load", 1, 1)
    log.info("loaded VOSR2 in %.1fs -> %s", pipeline.load_seconds, pipeline.describe())
    return pipeline


# ===========================================================================
#  IMAGE I/O + PUBLIC BATCH API
# ===========================================================================

def _pil_to_bhwc01(img) -> torch.Tensor:
    """PIL RGB image -> float32 BHWC tensor in [0, 1]."""
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    # np.array(..., copy=True) because PIL's buffer is read-only and
    # torch.from_numpy() refuses (and would then mis-handle) non-writable arrays.
    t = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)).div_(255.0)
    return t.unsqueeze(0).contiguous()


def _bhwc01_to_pil(x: torch.Tensor, band: int = 256):
    """BHWC [0,1] tensor (B=1, CPU) -> PIL RGB image.

    Converted in row bands: the obvious ``(arr * 255 + 0.5).astype(uint8)``
    materialises a second full-frame float32 array, which on a 8432x8432 output
    is 814 MB -- exactly the allocation that fails on a tight host.
    """
    from PIL import Image

    a = x[0].detach().to(torch.float32).clamp_(0.0, 1.0).cpu().numpy()
    h, w = a.shape[0], a.shape[1]
    buf = np.empty((h, w, 3), dtype=np.uint8)
    for y0 in range(0, h, band):
        y1 = min(y0 + band, h)
        buf[y0:y1] = np.clip(a[y0:y1] * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(buf, mode="RGB")


def _list_input_images(in_dir: Path) -> list:
    out = []
    for p in sorted(in_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out


def upscale_batch(in_dir, out_dir, *, upscale=4, tile_size=512, tile_overlap=64,
                  vae_tile_size=1024, vae_tile_overlap=128, seed=42,
                  color_alignment="wavelet", dtype="auto", models_root=None,
                  batch_size=1, noise_scale=0.0, offload_mode="auto",
                  vram_budget_bytes=None, pipeline=None, progress=None) -> dict:
    """Upscale every image in ``in_dir`` with VOSR 2.0 and write PNGs to ``out_dir``.

    Parameters
    ----------
    in_dir, out_dir : str | Path
        ``out_dir`` is created if missing.  Every ``.png/.jpg/.jpeg/.webp/.bmp``
        directly inside ``in_dir`` is processed; outputs keep the *input file name*
        (with the suffix replaced by ``.png``).
    upscale : int
        Exact output multiplier.
    tile_size, tile_overlap : int
        DiT tiling in pixel space.  ``tile_size=0`` disables DiT tiling; the default
        512/64 is the configuration VOSR 2.0 is meant to run at above 512px.
    vae_tile_size, vae_tile_overlap : int
        VAE tiling in pixel space.  ``vae_tile_size=0`` disables VAE tiling, which
        decodes the whole image in one pass and commonly OOMs above ~1024px.
    seed : int
        Initial latent-noise seed.  Only has an effect when ``noise_scale > 0``.
    noise_scale : float
        Multiplier on the unit-variance latent noise draw.  ``1.0`` is the
        published VOSR one-step contract; ``0.0`` (the **default here**) is the
        stable deterministic evaluation that the reference tool engine
        effectively uses.

        Why the default is 0.0: this one-step model is noise-dominated --
        measured ``d(sr)/d(z) ~= 0.98`` and ``|du|/|u(0)| = 1.958`` on the real
        checkpoint, so a unit-variance draw is copied almost unchanged into the
        output.  At ``noise_scale=1.0`` flat regions pick up strong coloured
        high-frequency speckle (the blocky/grid artifact) and agreement with the
        reference engine drops to corr 0.901 / 23.95 dB; at ``0.0`` it rises to
        corr 0.953 / 28.06 dB on real photos and 0.999 / 45-49 dB on synthetic
        inputs, and the artifact disappears.  See :func:`run_vosr2` for the full
        measurement table.  Set ``noise_scale=1.0`` for strict contract fidelity.
    color_alignment : str
        ``"wavelet"`` (default) | ``"adain"`` | ``"none"`` (alias ``"nofix"``).
    dtype : str
        ``"auto"`` | ``"bf16"`` | ``"fp16"`` | ``"fp32"``; only used when this call
        has to load the pipeline itself.
    models_root : str | Path | None
        Bundle root; ``None`` -> autodetect.
    batch_size : int
        How many images to push through ``run_vosr2`` in one call.  ``1`` reproduces
        the upstream per-item path exactly (per-item CPU offload + per-item color
        alignment) and is the safest for VRAM; ``>1`` batches only the resized
        pixels and the color-alignment reference, because ``run_vosr2`` still
        iterates items one at a time internally.
    offload_mode, vram_budget_bytes
        Forwarded to :func:`load_pipeline` when this call loads the pipeline.
    pipeline : VOSR2Pipeline | None
        Pre-loaded pipeline to reuse (skips model loading entirely).

        **Strongly recommended when calling this function more than once in a
        process.** With ``pipeline=None`` every call re-reads the 5.19 GB DiT
        safetensors from disk; the accumulated Windows commit charge from
        repeated loads eventually fails with ``OSError: [Errno 1455] The paging
        file is too small`` (observed after ~40 repeated calls). Pass a pipeline
        obtained from :func:`load_pipeline` instead::

            pipe = vosr2_standalone.load_pipeline(models_root=...)
            for d in dirs:
                vosr2_standalone.upscale_batch(d, out_d, pipeline=pipe)
    progress : callable | None
        ``progress(stage: str, done: int, total: int)``.  Stages emitted:
        ``"load"``, ``"image"``, ``"denoise"``, ``"save"``, ``"done"``.

    Returns
    -------
    dict
        ``{"ok": [file names...], "failed": {file name: "error text"}}``
    """

    def _progress(stage, done, total):
        if progress is not None:
            try:
                progress(stage, int(done), int(total))
            except Exception:
                log.debug("progress callback raised", exc_info=True)

    color_alignment = {"nofix": "none", "no_fix": "none", "": "none"}.get(
        str(color_alignment).lower(), color_alignment
    )

    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    if not in_dir.is_dir():
        raise VOSR2LoadError(f"in_dir is not a directory: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _list_input_images(in_dir)
    result = {"ok": [], "failed": {}}
    if not images:
        _progress("done", 0, 0)
        return result

    batch_size = max(1, int(batch_size))

    if pipeline is None:
        _progress("load", 0, 1)
        pipeline = load_pipeline(models_root=models_root, dtype=dtype,
                                 offload_mode=offload_mode,
                                 vram_budget_bytes=vram_budget_bytes)
        _progress("load", 1, 1)

    model = pipeline.model
    total = len(images)
    # The tiled path's internal ProgressBar (ported comfy.utils.ProgressBar)
    # reports through this global hook, which is how "denoise" tile counts reach
    # `progress` without threading a callback through the verbatim upstream code.
    set_progress_hook(_progress)
    try:
        for start in range(0, total, batch_size):
            chunk = images[start:start + batch_size]
            _progress("image", start, total)
            try:
                tensors = []
                for p in chunk:
                    from PIL import Image

                    with Image.open(p) as im:
                        rgb = im.convert("RGB")
                        tensors.append(_pil_to_bhwc01(rgb))
                bhwc = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]

                with torch.inference_mode():
                    out_bhwc = run_vosr2(
                        model, bhwc, upscale, seed, color_alignment,
                        tile_size, tile_overlap, vae_tile_size, vae_tile_overlap,
                        noise_scale=noise_scale,
                    )
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                for p in chunk:
                    result["failed"][p.name] = msg
                log.warning("batch at index %d failed: %s", start, msg, exc_info=True)
                _progress("image", start + len(chunk), total)
                continue

            for i, p in enumerate(chunk):
                _progress("save", start + i, total)
                img = None
                try:
                    img = _bhwc01_to_pil(out_bhwc[i:i + 1])
                    dst = out_dir / (p.stem + ".png")
                    img.save(dst, format="PNG")
                    result["ok"].append(p.name)
                except Exception as exc:
                    result["failed"][p.name] = f"{type(exc).__name__}: {exc}"
                    log.warning("saving %s failed", p.name, exc_info=True)
                finally:
                    # Release this item's pixels before moving to the next one.
                    if img is not None:
                        try:
                            img.close()
                        except Exception:
                            pass
                    del img

            del out_bhwc, bhwc, tensors
            _progress("image", start + len(chunk), total)
    finally:
        set_progress_hook(None)

    _progress("done", total, total)
    return result


# ===========================================================================
#  CLI (handy for a quick smoke test; the public contract is upscale_batch)
# ===========================================================================

def _main(argv=None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    ap = argparse.ArgumentParser(prog="vosr2_standalone",
                                 description="VOSR 2.0 batch upscaler (ComfyUI-free, einops-free).")
    ap.add_argument("-i", "--in-dir", required=True)
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("-u", "--upscale", type=int, default=4)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--tile-overlap", type=int, default=64)
    ap.add_argument("--vae-tile-size", type=int, default=1024)
    ap.add_argument("--vae-tile-overlap", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--color-alignment", default="wavelet",
                    choices=["wavelet", "adain", "none"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("-c", "--models-root", default=None)
    args = ap.parse_args(argv)

    def _report(stage, done, total):
        print(f"  [{stage}] {done}/{total}", flush=True)

    res = upscale_batch(
        args.in_dir, args.out_dir,
        upscale=args.upscale, tile_size=args.tile_size, tile_overlap=args.tile_overlap,
        vae_tile_size=args.vae_tile_size, vae_tile_overlap=args.vae_tile_overlap,
        seed=args.seed, color_alignment=args.color_alignment, dtype=args.dtype,
        models_root=args.models_root, progress=_report,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if not res["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
