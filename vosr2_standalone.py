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
