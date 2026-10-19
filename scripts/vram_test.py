"""检测 CUDA 实际可用显存: 报告 + 真实分配测试。"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

print("torch      :", torch.__version__)
print("cuda 可用  :", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("没有可用 CUDA 设备")

print("设备       :", torch.cuda.get_device_name(0))
print("算力       :", torch.cuda.get_device_capability(0))

free, total = torch.cuda.mem_get_info()
print(f"mem_get_info: free={free / 2**20:.0f} MiB   total={total / 2**20:.0f} MiB")

for gib in (1.0, 2.0, 4.0):
    n = int(gib * 2**30 // 4)
    try:
        t = torch.zeros(n, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        print(f"  分配 {gib:.1f} GiB -> 成功")
        del t
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        print(f"  分配 {gib:.1f} GiB -> 失败: {type(exc).__name__}: {str(exc)[:200]}")
        break

free2, _ = torch.cuda.mem_get_info()
print(f"测试后 free={free2 / 2**20:.0f} MiB")
