@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1

rem ===========================================================================
rem  VOSR 2.0 weights downloader -- standalone build
rem
rem  Downloads the 6.5 GB of model weights the upscaler needs.
rem  Weights are NOT shipped with this project; they belong to their authors.
rem
rem  Usage:
rem      double-click          download into  models\  next to this file
rem      this.bat --check      verify what is already downloaded
rem      this.bat --mirror hf  use Hugging Face instead of ModelScope
rem      this.bat --help       all options
rem
rem  NOTE: keep everything above the #PYSTART marker pure ASCII (see the note
rem  in the upscaler .bat for why).
rem ===========================================================================

set "SELF=%~f0"

set "PYEXE="
set "PYNOIMG="

call :try_py "%~dp0python\python.exe"
call :try_py "%~dp0..\python\python.exe"
call :try_py "%SHRINK_PYTHON%"
call :try_py "%VOSR2_PYTHON%"
if exist "%~dp0python.txt" (
    for /f "usebackq delims=" %%L in ("%~dp0python.txt") do call :try_py "%%L"
)
for /f "delims=" %%P in ('where python 2^>nul') do call :try_py "%%P"
for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do call :try_py "%%P"
call :try_py "C:\Python312\python.exe"
call :try_py "C:\Python311\python.exe"
call :try_py "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :try_py "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if not defined PYEXE if defined PYNOIMG set "PYEXE=%PYNOIMG%"

if not defined PYEXE (
    echo.
    echo  [ERROR] No usable Python found.
    echo.
    echo  Put a  python\  folder next to this .bat, or install Python from
    echo  https://www.python.org and tick "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

echo  [i] Python : %PYEXE%

set "TMPDIR=%TEMP%\vosr2_dl_%RANDOM%%RANDOM%"
mkdir "%TMPDIR%" 2>nul
set "PROG=%TMPDIR%\download_weights.py"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$l = Get-Content -LiteralPath $env:SELF -Encoding UTF8; $i = [Array]::IndexOf($l, '#PYSTART'); if ($i -lt 0) { exit 2 }; $enc = New-Object System.Text.UTF8Encoding($false); [System.IO.File]::WriteAllLines($env:PROG, $l[($i + 1)..($l.Count - 1)], $enc)"

if not exist "%PROG%" (
    echo.
    echo  [ERROR] Failed to extract the embedded Python payload.
    echo          Make sure this file still has CRLF line endings and the
    echo          '#PYSTART' marker line is intact.
    echo.
    pause
    exit /b 1
)

if /i "%~1"=="--dump" (
    copy /y "%PROG%" "%~dp0download_weights.py" >nul
    echo  Dumped to: %~dp0download_weights.py
    rmdir /s /q "%TMPDIR%" >nul 2>&1
    exit /b 0
)

"%PYEXE%" "%PROG%" --bat-dir "%~dp0." %*
set "RC=%ERRORLEVEL%"
rmdir /s /q "%TMPDIR%" >nul 2>&1

echo.
if "%RC%"=="0" (echo  [OK] Done.) else (echo  [FAILED] exit code = %RC%)
pause
exit /b %RC%

:try_py
if "%~1"=="" exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 0
if not defined PYEXE set "PYEXE=%~1"
exit /b 0

#PYSTART
#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""下载 VOSR 2.0 运行所需的模型权重。

为什么要有这个脚本
------------------
权重一共约 6.49 GB，塞进 Git 仓库既超限制又不合规。所以仓库只放代码，
权重由用户自己跑这个脚本下载。

下载什么（只下 VOSR 2.0 推理真正需要的 5 个文件）
--------------------------------------------------
    Qwen-Image-vae-2d/config.json                              754 字节
    Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors    170.2 MB
    VOSR2/args.json                                            816 字节
    VOSR2/checkpoints/ema_model.safetensors                   5.19 GB
    torch_cache/checkpoints/dinov2_vitl14_pretrain.pth        1.13 GB

全部来自 ModelScope 的 LULALULALU/VOSR_CKPT（或 Hugging Face 的 CSWRY/VOSR）。

特性
----
* 断点续传：网断了重跑会从断点继续，不会从头下
* SHA256 校验：下完逐个核对，和官方仓库公布的一致
* 已存在就跳过：大小和哈希都对得上就不重复下
* 进度条：显示已下载量、速度、剩余时间
* 两个镜像：默认 ModelScope（国内快），可切 Hugging Face

用法
----
    python download_weights.py                   # 下到 ./models
    python download_weights.py -o D:\vosr2\models
    python download_weights.py --mirror hf       # 走 Hugging Face
    python download_weights.py --check           # 只校验已有文件，不下载
    python download_weights.py --no-verify       # 跳过哈希（快，但没保障）
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

# ===========================================================================
#  下载源
# ===========================================================================
MS_REPO = "LULALULALU/VOSR_CKPT"
HF_REPO = "CSWRY/VOSR"

URL_TEMPLATES = {
    "modelscope": "https://modelscope.cn/api/v1/models/"
                  + MS_REPO + "/repo?Revision=master&FilePath={path}",
    "hf": "https://huggingface.co/" + HF_REPO + "/resolve/main/{path}?download=true",
}

# (仓库内相对路径, 字节数, SHA256)  —— 与 ModelScope 仓库公布的一致
FILES = [
    ("Qwen-Image-vae-2d/config.json",
     754,
     "86b0706a47e95dc7e469e7895ef1d43a062b2104d0d5fe16c421a5df5d9fd142"),
    ("Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors",
     178418892,
     "1dd2c67f3f29734d48709513ed181d4f351966d76ba95f77a5cff8e1daf547fa"),
    ("VOSR2/args.json",
     816,
     "63b2f423eafae1f684ca860eb8fd0e81aa5fac858397f783d8f337871241c281"),
    ("VOSR2/checkpoints/ema_model.safetensors",
     5576385424,
     "bdcaa81e4c675b6074de643e27daafe73eb8125d4d0264d1bf30a004e9644b71"),
    ("torch_cache/checkpoints/dinov2_vitl14_pretrain.pth",
     1217586395,
     "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"),
]

TOTAL_BYTES = sum(f[1] for f in FILES)
CHUNK = 1 << 20          # 1 MB


# ===========================================================================
#  显示
# ===========================================================================
def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def bar(frac: float, width: int = 22) -> str:
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    filled = int(width * frac + 0.5)
    return "[" + "#" * filled + "." * (width - filled) + f"] {int(frac * 100):3d}%"


_PREV = [0]


def live(msg: str) -> None:
    pad = max(0, _PREV[0] - len(msg))
    sys.stdout.write("\r" + msg + " " * pad)
    sys.stdout.flush()
    _PREV[0] = len(msg)


def live_clear() -> None:
    if _PREV[0]:
        sys.stdout.write("\r" + " " * _PREV[0] + "\r")
        sys.stdout.flush()
        _PREV[0] = 0


# ===========================================================================
#  校验
# ===========================================================================
def sha256_file(path: Path, on_progress=None) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    done = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(8 << 20)
            if not b:
                break
            h.update(b)
            done += len(b)
            if on_progress and size:
                on_progress(done, size)
    return h.hexdigest()


def already_ok(path: Path, size: int, sha: str, verify: bool) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size != size:
        return False
    if not verify:
        return True
    return sha256_file(path) == sha


# ===========================================================================
#  下载
# ===========================================================================
def download_one(url: str, dest: Path, size: int, sha: str,
                 verify: bool, retries: int = 4) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.is_file() else 0
        if have > size:                      # 上次写坏了，重来
            part.unlink()
            have = 0
        if have == size:
            break

        req = urllib.request.Request(url, headers={
            "User-Agent": "vosr2-weights-downloader/1.0",
        })
        if have:
            req.add_header("Range", f"bytes={have}-")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:
                    have = 0                 # 服务器不支持续传，从头来
                mode = "ab" if have else "wb"
                t0 = time.time()
                done = have
                with open(part, mode) as fh:
                    while True:
                        buf = resp.read(CHUNK)
                        if not buf:
                            break
                        fh.write(buf)
                        done += len(buf)
                        el = max(0.001, time.time() - t0)
                        spd = (done - have) / el
                        eta = (size - done) / spd if spd > 1 else 0
                        live(f"     {bar(done / size)}  {human(done)} / {human(size)}"
                             f"  {human(spd)}/s  剩 {int(eta // 60)}分{int(eta % 60):02d}秒")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            live_clear()
            print(f"     ! 第 {attempt} 次中断: {type(exc).__name__}")
            if attempt == retries:
                return False
            time.sleep(2 * attempt)
            continue

        live_clear()
        if part.stat().st_size >= size:
            break

    if part.stat().st_size != size:
        live_clear()
        print(f"     ! 大小不对: 得到 {part.stat().st_size}，应为 {size}")
        return False

    if verify:
        print("     校验 SHA256 ...", end="", flush=True)
        got = sha256_file(part)
        if got != sha:
            print(f"\r     ! 校验失败\n       期望 {sha}\n       实际 {got}")
            part.unlink()
            return False
        print("\r     校验通过            ")

    part.replace(dest)
    return True


# ===========================================================================
#  主流程
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("-o", "--out", default=None, help="输出目录（默认 ./models）")
    ap.add_argument("--mirror", default="modelscope",
                    choices=("modelscope", "hf"), help="下载源")
    ap.add_argument("--check", action="store_true", help="只校验，不下载")
    ap.add_argument("--no-verify", action="store_true", help="跳过 SHA256 校验")
    ap.add_argument("--bat-dir", default=None, help="内部使用")
    args = ap.parse_args()

    base = Path(args.bat_dir).resolve() if args.bat_dir else Path(__file__).resolve().parent
    out = Path(args.out) if args.out else (base / "models")
    verify = not args.no_verify

    print("=" * 74)
    print("  VOSR 2.0 模型权重下载")
    print("=" * 74)
    print(f"  下载源 : {args.mirror}  ({MS_REPO if args.mirror == 'modelscope' else HF_REPO})")
    print(f"  保存到 : {out}")
    print(f"  总大小 : {human(TOTAL_BYTES)}  共 {len(FILES)} 个文件")
    print(f"  校验   : {'SHA256' if verify else '仅比对大小'}")
    print()
    print("  权重许可归各自作者所有，请遵守其使用条款；")
    print("  本项目只做下载，不附带、不修改任何权重。")
    print()

    if args.check:
        bad = 0
        for rel, size, sha in FILES:
            p = out / rel
            ok = already_ok(p, size, sha, verify)
            print(f"  {'OK  ' if ok else '缺失/不符'}  {rel}")
            if not ok:
                bad += 1
        print()
        print("  全部就绪。" if not bad else f"  有 {bad} 个文件需要下载。")
        return 0 if not bad else 1

    tpl = URL_TEMPLATES[args.mirror]
    okn = skipn = 0
    t_all = time.time()

    for i, (rel, size, sha) in enumerate(FILES, 1):
        dest = out / rel
        print(f"  [{i}/{len(FILES)}] {rel}   ({human(size)})")

        if already_ok(dest, size, sha, verify):
            print(f"        已存在且校验通过，跳过")
            skipn += 1
            continue

        url = tpl.format(path=rel)
        if download_one(url, dest, size, sha, verify):
            print(f"        完成 -> {dest}")
            okn += 1
        else:
            live_clear()
            print()
            print(f"  [失败] {rel}")
            print(f"         可以手动下载：")
            print(f"           ModelScope: https://modelscope.cn/models/{MS_REPO}")
            print(f"           HuggingFace: https://huggingface.co/{HF_REPO}")
            print(f"         然后放到：{dest}")
            print()
            return 1
        print()

    live_clear()
    print("  " + "-" * 70)
    print(f"  完成：新下载 {okn} 个，跳过 {skipn} 个，用时 {time.time() - t_all:.0f} 秒")
    print(f"  权重目录：{out}")
    print()
    print("  现在可以运行  一键放大.bat  了。")
    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[已取消]")
        raise SystemExit(130)
