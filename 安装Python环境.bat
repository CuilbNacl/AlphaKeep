@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1

rem ===========================================================================
rem  部署独立 Python 运行环境（供 VOSR2 一键放大使用）
rem
rem  做四件事：
rem    1. 下载官方 Python 3.11 嵌入式包，解压到本目录下的 python\
rem    2. 装上 pip
rem    3. 按你的显卡自动选 CUDA 版本并安装 torch
rem    4. 安装 numpy / Pillow / safetensors，并做一次自检
rem
rem  用法：
rem    双击                     自动检测显卡并选 CUDA 版本
rem    安装Python环境.bat cu126  强制指定 CUDA 版本
rem    安装Python环境.bat cpu    安装 CPU 版
rem    安装Python环境.bat D:\mypy cu128    指定安装目录
rem
rem  不写注册表、不装进系统目录、不需要管理员权限。
rem  卸载 = 直接删掉本目录下的 python\ 文件夹。
rem ===========================================================================

set "SELF=%~f0"
set "BATDIR=%~dp0"
set "TARGET="
set "CUDAARG="

for %%A in (%*) do (
    set "ARG=%%~A"
    setlocal EnableDelayedExpansion
    if /i "!ARG:~0,3!"=="cu1" (
        endlocal & set "CUDAARG=%%~A"
    ) else if /i "!ARG!"=="cpu" (
        endlocal & set "CUDAARG=cpu"
    ) else (
        endlocal & set "TARGET=%%~A"
    )
)
if not defined TARGET set "TARGET=%BATDIR%python"

set "PYDIR=%TARGET%"
set "PYEXE=%PYDIR%\python.exe"
set "PYVER=3.11.9"
set "ZIPNAME=python-%PYVER%-embed-amd64.zip"
set "ZIPURL=https://www.python.org/ftp/python/%PYVER%/%ZIPNAME%"
set "ZIPLOCAL=%PYDIR%\%ZIPNAME%"
set "PTH=%PYDIR%\python311._pth"

echo ============================================================
echo   VOSR2 独立 Python 环境部署
echo ============================================================
echo   安装目录 : %PYDIR%
echo   Python   : %PYVER%  (官方 embedded amd64)
echo   CUDA 参数: %CUDAARG%
echo ============================================================
echo.

if not exist "%PYDIR%" mkdir "%PYDIR%" 2>nul
if not exist "%PYDIR%" (
    echo  [错误] 无法创建目录 %PYDIR%
    pause
    exit /b 1
)

rem ---- 1. 下载官方嵌入式 Python ----
if exist "%PYEXE%" (
    echo  [跳过] Python 已存在：%PYEXE%
    goto :have_python
)
if not exist "%ZIPLOCAL%" (
    echo  [1/4] 下载 %ZIPNAME% ^(约 10 MB^) ...
    where curl.exe >nul 2>&1
    if not errorlevel 1 (
        curl.exe -L --fail --silent --show-error --max-time 600 -o "%ZIPLOCAL%" "%ZIPURL%"
    ) else (
        powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri $env:ZIPURL -OutFile $env:ZIPLOCAL -UseBasicParsing"
    )
)
if not exist "%ZIPLOCAL%" (
    echo.
    echo  [错误] 下载失败。请手动下载：
    echo         %ZIPURL%
    echo         放到：%ZIPLOCAL%
    echo         然后重新双击本脚本。
    echo.
    pause
    exit /b 1
)

echo  [2/4] 解压 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath $env:ZIPLOCAL -DestinationPath $env:PYDIR -Force"
if not exist "%PYEXE%" (
    echo  [错误] 解压后找不到 %PYEXE%
    pause
    exit /b 1
)
del "%ZIPLOCAL%" >nul 2>&1

:have_python
rem ---- 修 python311._pth：让嵌入式 Python 能 import site 和 site-packages ----
if exist "%PTH%" (
    > "%PTH%" echo python311.zip
    >>"%PTH%" echo .
    >>"%PTH%" echo Lib\site-packages
    >>"%PTH%" echo import site
    echo  [i] 已重写 python311._pth
)
if not exist "%PYDIR%\Lib\site-packages" mkdir "%PYDIR%\Lib\site-packages" 2>nul

rem ---- 3. 释放内嵌的 Python 段并执行 ----
set "TMPPY=%TEMP%\vosr2_setup_%RANDOM%%RANDOM%.py"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$l = Get-Content -LiteralPath $env:SELF -Encoding UTF8; $i = [Array]::IndexOf($l, '#PYSTART'); if ($i -lt 0) { exit 2 }; $enc = New-Object System.Text.UTF8Encoding($false); [System.IO.File]::WriteAllLines($env:TMPPY, $l[($i + 1)..($l.Count - 1)], $enc)"
if not exist "%TMPPY%" (
    echo  [错误] 无法释放内嵌脚本。请确认本文件没有被编辑器改坏换行符^(需要 CRLF^)。
    pause
    exit /b 1
)

echo  [3/4] 安装 pip / torch / 依赖（最慢的一步，可能十几分钟）...
echo.
"%PYEXE%" "%TMPPY%" --python-dir "%PYDIR%" --cuda "%CUDAARG%"
set "RC=%ERRORLEVEL%"
del "%TMPPY%" >nul 2>&1
if not "%RC%"=="0" (
    echo.
    echo  [错误] 依赖安装失败，退出码 %RC%
    pause
    exit /b %RC%
)

rem ---- 4. 自检 ----
echo.
echo  [4/4] 自检 ...
echo.
"%PYEXE%" -c "import sys,torch,numpy,PIL,safetensors;print('python  :',sys.version.split()[0]);print('torch   :',torch.__version__);print('cuda可用:',torch.cuda.is_available());print('GPU     :',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无(CPU模式)');print('numpy   :',numpy.__version__);print('Pillow  :',PIL.__version__)"

echo.
echo ============================================================
echo  完成。Python 解释器：%PYEXE%
echo.
echo  下一步：在「一键放大.bat」同目录建一个 vosr2_env.txt，
echo  里面写这一行：
echo      %PYEXE%
echo ============================================================
pause
exit /b 0


#PYSTART
# -*- coding: utf-8 -*-
r"""
VOSR2 独立环境部署 —— 内嵌 Python 主体

职责：
  1. 确保 pip 可用（下载官方 get-pip.py）
  2. 检测显卡与驱动版本，推荐并安装对应 CUDA 构建的 torch
  3. 安装 numpy / Pillow / safetensors
被 安装Python环境.bat 调用，也可以单独运行：
  python.exe setup_env_payload.py --python-dir <目录> --cuda cu128
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GETPIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# 驱动版本 -> 可用的 CUDA 构建。表是保守值，宁可低一档也不要装完跑不起来。
# 依据 NVIDIA 的 CUDA 与驱动对应关系 + PyTorch 官方支持的 wheel 列表。
DRIVER_TABLE = [
    (570, "cu128"),   # CUDA 12.8 需要 Windows 驱动 >= 570.65
    (560, "cu126"),   # CUDA 12.6 需要 >= 560.76
    (551, "cu124"),   # CUDA 12.4 需要 >= 551.61
    (531, "cu121"),   # CUDA 12.1 需要 >= 531.14
    (522, "cu118"),   # CUDA 11.8 需要 >= 522.06
]
ALL_VARIANTS = ("cu128", "cu126", "cu124", "cu121", "cu118", "cpu")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("    > " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, text=True, **kw)


def detect_gpu() -> tuple[str | None, int | None]:
    """返回 (显卡名, 驱动主版本号)。没有 NVIDIA 卡时返回 (None, None)。"""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        fallback = Path(r"C:\Windows\System32\nvidia-smi.exe")
        smi = str(fallback) if fallback.is_file() else None
    if smi is None:
        return None, None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except BaseException as exc:
        print(f"  nvidia-smi 调用失败: {exc}")
        return None, None
    if not out:
        return None, None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if parts else None
    major = None
    if len(parts) > 1:
        m = re.match(r"(\d+)", parts[1])
        if m:
            major = int(m.group(1))
    return name, major


def recommend_cuda(name: str | None, driver: int | None) -> str:
    if name is None and driver is None:
        return "cpu"
    # Blackwell（RTX 50 系）只有 cu128 及以上的 wheel 才有 sm_120
    if name and re.search(r"RTX\s*50\d\d|B[0-9]{3}|Blackwell", name, re.I):
        return "cu128"
    if driver is None:
        return "cpu"
    for min_driver, variant in DRIVER_TABLE:
        if driver >= min_driver:
            return variant
    return "cu118"


def ensure_pip(pyexe: str) -> None:
    print("\n[1/3] 检查 pip ...", flush=True)
    probe = subprocess.run([pyexe, "-m", "pip", "--version"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        print("  pip 已存在:", probe.stdout.strip())
        return
    print("  未检测到 pip，下载官方 get-pip.py ...", flush=True)
    tmp = Path(tempfile.gettempdir()) / "vosr2_get-pip.py"
    try:
        data = urllib.request.urlopen(GETPIP_URL, timeout=120).read()
        tmp.write_bytes(data)
        print(f"  已下载 {len(data) / 1024:.0f} KB")
    except BaseException as exc:
        raise SystemExit(f"[错误] 无法下载 get-pip.py: {exc}")
    r = run([pyexe, str(tmp)], capture_output=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit("[错误] get-pip.py 执行失败")
    print("  pip 安装完成")


def pip_install(pyexe: str, pkgs: list[str], index_url: str | None = None) -> None:
    cmd = [pyexe, "-m", "pip", "install", "--no-warn-script-location",
           "--disable-pip-version-check", *pkgs]
    if index_url:
        cmd += ["--index-url", index_url]
    r = run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"[错误] pip 安装失败（退出码 {r.returncode}）：{' '.join(pkgs)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-dir", required=True)
    ap.add_argument("--cuda", default="")
    args = ap.parse_args()

    pydir = Path(args.python_dir).resolve()
    pyexe = str(pydir / "python.exe")
    if not Path(pyexe).is_file():
        print(f"[错误] 找不到 {pyexe}")
        return 2
    (pydir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  目标解释器: {pyexe}")
    print("=" * 60)

    ensure_pip(pyexe)

    print("\n[2/3] 选择 torch 的 CUDA 构建 ...", flush=True)
    name, driver = detect_gpu()
    if name:
        print(f"  检测到显卡: {name}    驱动主版本: {driver}")
    else:
        print("  未检测到 NVIDIA 显卡")

    variant = (args.cuda or "").strip().lower() or recommend_cuda(name, driver)
    if variant not in ALL_VARIANTS:
        print(f"  [警告] 未知的 CUDA 参数 '{variant}'，改回自动选择")
        variant = recommend_cuda(name, driver)
    print(f"  将安装: torch ({variant})")

    index = ("https://download.pytorch.org/whl/cpu" if variant == "cpu"
             else f"https://download.pytorch.org/whl/{variant}")
    print(f"  索引源: {index}")
    print("  提示：torch 大约 2.5-3 GB，第一次会很慢，请耐心等待。\n")

    pip_install(pyexe, ["torch", "torchvision"], index_url=index)
    pip_install(pyexe, ["numpy", "pillow", "safetensors"])

    print("\n[3/3] 校验 ...", flush=True)
    r = subprocess.run(
        [pyexe, "-c",
         "import torch;print('torch', torch.__version__);"
         "print('cuda', torch.cuda.is_available());"
         "print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"],
        text=True)
    if r.returncode != 0:
        print("  [警告] 校验失败，但依赖已装完，请把上面的输出留存")
        return 1

    if name and not variant == "cpu":
        chk = subprocess.run(
            [pyexe, "-c", "import torch;print(torch.cuda.is_available())"],
            capture_output=True, text=True).stdout.strip()
        if chk != "True":
            print("\n  [警告] torch 装好了但 cuda 不可用。")
            print("         可能原因：驱动太旧、显卡架构不被该 wheel 支持。")
            print("         可以改用：安装Python环境.bat cu126  （或 cu121 / cpu）")

    print("\n部署完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
