#!/usr/bin/env python3
"""
PyTorch/CUDA port of train_gpt_mlx.py.

Goals:
- Keep the original script structure readable for newcomers.
- Preserve the same high-level model, optimizer split, data pipeline, validation,
  and int8+zlib export logic where practical.
- Replace MLX-specific pieces with PyTorch/CUDA equivalents.

Notes:
- This is a faithful porting effort, not a byte-for-byte behavioral clone.
- On CPU it can still run, but the intended target is CUDA.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import random
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# DEVICE + DTYPE
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16


# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

class Hyperparameters:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop.
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    cuda_max_microbatch_tokens: int = int(os.environ.get("CUDA_MAX_MICROBATCH_TOKENS", 8_192))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model.
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Runtime.
    enable_torch_compile: bool = bool(int(os.environ.get("ENABLE_TORCH_COMPILE", "1")))
    compile_mode: str = os.environ.get("TORCH_COMPILE_MODE", "default")
    allow_tf32: bool = bool(int(os.environ.get("ALLOW_TF32", "1")))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    # Quantization.
    awq_quant: bool = bool(int(os.environ.get("AWQ_QUANT", "1")))
    awq_calib_batches: int = int(os.environ.get("AWQ_CALIB_BATCHES", "4"))

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)).to(dtype=x.dtype)


def zeropower_newtonschulz5(g: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / (torch.sqrt(torch.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(dtype=g.dtype)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================

class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = torch.from_numpy(chunk[:-1].reshape(-1, seq_len)).long().to(DEVICE, non_blocking=True)
        y = torch.from_numpy(chunk[1:].reshape(-1, seq_len)).long().to(DEVICE, non_blocking=True)
        return x, y


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

def _awq_mean_abs_np(x: torch.Tensor) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    mean_abs = flat.abs().mean(dim=0)
    return mean_abs.detach().float().cpu().numpy().astype(np.float32, copy=False)


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype))


class RMSNormNoWeight(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]
        t = x.shape[-2]
        freqs = torch.outer(torch.arange(t, device=x.device, dtype=torch.float32), self.inv_freq)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = nn.Parameter(torch.ones((num_heads,), dtype=torch.float32) * qk_gain_init)
        self.rope = RoPE(self.head_dim, base=rope_base)
        self.scale = self.head_dim ** -0.5
        self.kv_repeat = self.num_heads // self.num_kv_heads

    def forward(
        self,
        x: torch.Tensor,
        layer_prefix: str = "",
        awq_accum: dict[str, list[np.ndarray]] | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        if awq_accum is not None:
            awq_accum.setdefault(f"{layer_prefix}.attn.qkv_in", []).append(_awq_mean_abs_np(x))
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(rms_norm(q).to(dtype=COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).to(dtype=COMPUTE_DTYPE))
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=self.scale)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        if awq_accum is not None:
            awq_accum.setdefault(f"{layer_prefix}.attn.proj_in", []).append(_awq_mean_abs_np(y))
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def forward(
        self,
        x: torch.Tensor,
        layer_prefix: str = "",
        awq_accum: dict[str, list[np.ndarray]] | None = None,
    ) -> torch.Tensor:
        if awq_accum is not None:
            awq_accum.setdefault(f"{layer_prefix}.mlp.fc_in", []).append(_awq_mean_abs_np(x))
        h = F.relu(self.fc(x))
        if awq_accum is not None:
            awq_accum.setdefault(f"{layer_prefix}.mlp.proj_in", []).append(_awq_mean_abs_np(h))
        return self.proj(h * h)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones((dim,), dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones((dim,), dtype=torch.float32))
        resid = np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32)))
        self.resid_mix = nn.Parameter(torch.from_numpy(resid))

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        layer_idx: int | None = None,
        awq_accum: dict[str, list[np.ndarray]] | None = None,
    ) -> torch.Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        prefix = f"blocks.{layer_idx}" if layer_idx is not None else ""
        attn_out = self.attn(self.attn_norm(x), layer_prefix=prefix, awq_accum=awq_accum)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x), layer_prefix=prefix, awq_accum=awq_accum
        )
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
        tied_embed_init_std: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones((self.num_skip_weights, dim), dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNormNoWeight()

        for b in self.blocks:
            nn.init.zeros_(b.attn.proj.weight)
            nn.init.zeros_(b.mlp.proj.weight)
        with torch.no_grad():
            self.tok_emb.weight.copy_(torch.randn_like(self.tok_emb.weight, dtype=torch.float32) * tied_embed_init_std)

    def softcap(self, logits: torch.Tensor) -> torch.Tensor:
        c = self.logit_softcap
        return c * torch.tanh(logits / c)

    def forward(
        self,
        input_ids: torch.Tensor,
        awq_accum: dict[str, list[np.ndarray]] | None = None,
    ) -> torch.Tensor:
        x = rms_norm(self.tok_emb(input_ids).to(dtype=COMPUTE_DTYPE))
        x0 = x
        skips: list[torch.Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, layer_idx=i, awq_accum=awq_accum)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            bi = self.num_encoder_layers + i
            x = self.blocks[bi](x, x0, layer_idx=bi, awq_accum=awq_accum)
        h = self.final_norm(x)
        if awq_accum is not None:
            awq_accum.setdefault("tok_emb.lm_head_in", []).append(_awq_mean_abs_np(h))
        return h

    def loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        weight = self.tok_emb.weight.to(dtype=x.dtype)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ weight.T
            logits = self.softcap(logits_proj)
            return F.cross_entropy(logits.float(), y, reduction="mean")

        loss_sum = x.new_tensor(0.0, dtype=torch.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ weight.T
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + F.cross_entropy(logits.float(), y[s:e], reduction="sum")
        return loss_sum / float(n)


# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================

class Muon:
    def __init__(self, named_params: dict[str, nn.Parameter], keys: list[str], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: torch.zeros_like(named_params[k].data) for k in keys}

    @torch.no_grad()
    def step(self, named_params: dict[str, nn.Parameter], step: int, lr_mul: float) -> None:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        for k in self.keys:
            p = named_params[k]
            if p.grad is None:
                continue
            g = p.grad
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            p.data.add_((-lr * scale) * g_ortho.to(dtype=p.dtype))


class SplitOptimizers:
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        named_params = dict(model.named_parameters())
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k for k, p in named_params.items()
            if k.startswith("blocks.") and p.ndim == 2 and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in named_params.items()
            if k == "skip_weights" or (k.startswith("blocks.") and (p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
        ]
        self.muon = Muon(named_params, self.matrix_keys, args)
        self.embed_opt = torch.optim.Adam([
            named_params[self.embed_key]
        ], lr=args.tied_embed_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)
        self.scalar_opt = torch.optim.Adam([
            named_params[k] for k in self.scalar_keys
        ], lr=args.scalar_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)

    def zero_grad(self) -> None:
        self.embed_opt.zero_grad(set_to_none=True)
        self.scalar_opt.zero_grad(set_to_none=True)

    def set_lrs(self, lr_mul: float) -> None:
        for group in self.embed_opt.param_groups:
            group["lr"] = self.args.tied_embed_lr * lr_mul
        for group in self.scalar_opt.param_groups:
            group["lr"] = self.args.scalar_lr * lr_mul

    @torch.no_grad()
    def step(self, model: GPT, step: int, lr_mul: float) -> None:
        named_params = dict(model.named_parameters())
        self.set_lrs(lr_mul)
        self.muon.step(named_params, step=step, lr_mul=lr_mul)
        self.embed_opt.step()
        self.scalar_opt.step()


# ==============================================================================
# QUANTIZATION (INT8 + ZLIB)
# ==============================================================================

MX_DTYPE_FROM_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def _np_float32(arr: torch.Tensor) -> np.ndarray:
    return arr.detach().float().cpu().numpy().astype(np.float32, copy=False)


def keep_float_array(name: str, arr: torch.Tensor, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(arr.detach().to(dtype=torch.float16).cpu().numpy().astype(INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(arr.detach().cpu().numpy().copy())


def quantize_float_array(arr: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int8(
    flat_state: dict[str, torch.Tensor],
    awq_weight_scales: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    awq_in_scales: dict[str, np.ndarray] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    awq_weight_scales = awq_weight_scales or {}
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.numel() * arr.element_size())
        if not torch.is_floating_point(arr):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(arr.detach().cpu().numpy())
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        if int(arr.numel()) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        to_quant = arr
        if name in awq_weight_scales and arr.ndim == 2:
            s_in = np.asarray(awq_weight_scales[name], dtype=np.float32)
            if s_in.shape[0] == arr.shape[1]:
                w = _np_float32(arr) * s_in[None, :]
                to_quant = torch.from_numpy(w)
                awq_in_scales[name] = np.ascontiguousarray(s_in.astype(np.float16, copy=False))
        q, s = quantize_float_array(to_quant)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    fmt = "int8_awq_per_row_v1" if awq_in_scales else "int8_clean_per_row_v1"
    obj: dict[str, object] = {
        "__quant_format__": fmt,
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    if awq_in_scales:
        obj["awq_in_scales"] = awq_in_scales
    return obj, stats


def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    awq_in = quant_obj.get("awq_in_scales", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        if name in awq_in:
            s_in = np.asarray(awq_in[name], dtype=np.float32)
            out_arr = out_arr / s_in[None, :]
        out[name] = torch.from_numpy(out_arr).to(dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = torch.from_numpy(out_arr).to(dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = torch.from_numpy(out_arr)
    return out


def _finalize_awq_accum(awq_accum: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, lst in awq_accum.items():
        stacked = np.stack(lst, axis=0)
        s = np.mean(stacked, axis=0).astype(np.float32)
        s = np.maximum(s, 1e-8)
        s = s / np.mean(s)
        out[k] = s
    return out


def _expand_awq_to_weight_names(args: Hyperparameters, awq_mean: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i in range(args.num_layers):
        p = f"blocks.{i}.attn"
        s_qkv = awq_mean[f"{p}.qkv_in"]
        for suf in ("c_q.weight", "c_k.weight", "c_v.weight"):
            out[f"{p}.{suf}"] = s_qkv
        out[f"{p}.proj.weight"] = awq_mean[f"{p}.proj_in"]
        mp = f"blocks.{i}.mlp"
        out[f"{mp}.fc.weight"] = awq_mean[f"{mp}.fc_in"]
        out[f"{mp}.proj.weight"] = awq_mean[f"{mp}.proj_in"]
    out["tok_emb.weight"] = awq_mean["tok_emb.lm_head_in"]
    return out


def calibrate_awq_scales(model: GPT, val_tokens: np.ndarray, args: Hyperparameters) -> dict[str, np.ndarray]:
    awq_accum: dict[str, list[np.ndarray]] = {}
    seq_len = args.train_seq_len
    usable = (val_tokens.size - 1) // seq_len
    n = min(usable, max(1, args.awq_calib_batches))
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=COMPUTE_DTYPE, enabled=(DEVICE.type == "cuda")):
        for b in range(n):
            start = b * seq_len
            if start + seq_len + 1 > val_tokens.size:
                break
            chunk = val_tokens[start : start + seq_len + 1]
            input_ids = torch.from_numpy(chunk[:-1].reshape(1, seq_len)).long().to(DEVICE)
            _ = model(input_ids, awq_accum=awq_accum)
    finalized = _finalize_awq_accum(awq_accum)
    return _expand_awq_to_weight_names(args, finalized)


# ==============================================================================
# TOKENIZER / DATASET HELPERS
# ==============================================================================

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


# ==============================================================================
# TRAIN/EVAL HELPERS
# ==============================================================================

def maybe_torch_compile(fn, args: Hyperparameters):
    if not args.enable_torch_compile:
        return fn
    if not hasattr(torch, "compile"):
        return fn
    try:
        return torch.compile(fn, mode=args.compile_mode)
    except Exception:
        return fn


def loss_forward(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return model.loss(x, y)


def backward_microbatch(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    scale: float,
    compiled_loss_fn,
) -> float:
    with torch.autocast(device_type=DEVICE.type, dtype=COMPUTE_DTYPE, enabled=(DEVICE.type == "cuda")):
        loss = compiled_loss_fn(model, x, y)
    (loss * scale).backward()
    return float(loss.detach().item())


def train_step_chunked(
    args: Hyperparameters,
    model: GPT,
    train_loader: TokenLoader,
    compiled_loss_fn,
) -> float:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.cuda_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = 0.0
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        scale = float(y.numel()) / total_tokens
        loss_value += backward_microbatch(model, x, y, scale=scale, compiled_loss_fn=compiled_loss_fn) * scale
    return loss_value


@torch.no_grad()
def eval_val(
    args: Hyperparameters,
    model: GPT,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    model.eval()
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = torch.from_numpy(x_np).long().to(DEVICE)
        y = torch.from_numpy(y_np).long().to(DEVICE)
        chunk_token_count = float(y.numel())
        with torch.autocast(device_type=DEVICE.type, dtype=COMPUTE_DTYPE, enabled=(DEVICE.type == "cuda")):
            batch_loss = model.loss(x, y).float()
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        if log_fn is not None and total_batches > 1 and batch_idx == total_batches:
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    model.train()
    return val_loss, val_bpb


def flat_state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def clip_grad_model(model: nn.Module, max_norm: float) -> None:
    if max_norm <= 0:
        return
    params = [p for p in model.parameters() if p.grad is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running PyTorch {torch.__version__}", console=False)
    log(f"device:{DEVICE}", console=False)
    log("=" * 100, console=False)

    if args.allow_tf32 and DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if not args.tie_embeddings:
        raise NotImplementedError("train_gpt_cuda.py only supports tied embeddings")
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    seed_everything(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    ).to(DEVICE)
    opt = SplitOptimizers(model, args)
    compiled_loss_fn = maybe_torch_compile(loss_forward, args)

    n_params = sum(int(np.prod(tuple(p.shape))) for p in model.parameters())
    log(f"run_id:{args.run_id}")
    log(f"torch_version:{torch.__version__}")
    log(f"device:{DEVICE} compute_dtype:{COMPUTE_DTYPE}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"cuda_max_microbatch_tokens:{args.cuda_max_microbatch_tokens}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(f"torch_compile:{args.enable_torch_compile}")

    model.train()
    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            opt.zero_grad()
            for _ in range(args.grad_accum_steps):
                _ = train_step_chunked(args, model, train_loader, compiled_loss_fn)
            opt.zero_grad()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log,
            )
            if step % 25 == 0 or last_step:
                log(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        opt.zero_grad()
        train_loss_value = 0.0
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            train_loss_value += train_step_chunked(args, model, train_loader, compiled_loss_fn) * grad_scale

        clip_grad_model(model, args.grad_clip_norm)
        opt.step(model, step=step, lr_mul=lr_mul)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step % args.train_log_every == 0 or stop_after_step is not None):
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}",
                console=True,
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    out_path = out_dir / f"{args.run_id}_cuda_model.pt"
    torch.save(model.state_dict(), out_path)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    flat_state = flat_state_dict_cpu(model)
    awq_scales: dict[str, np.ndarray] | None = None
    if args.awq_quant:
        log(f"awq_calib:batches:{args.awq_calib_batches}")
        awq_scales = calibrate_awq_scales(model, val_tokens, args)
        quant_obj, quant_stats = quantize_state_dict_int8(flat_state, awq_weight_scales=awq_scales)
    else:
        quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_cuda_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    qtag = "serialized_model_int8_awq_zlib" if args.awq_quant else "serialized_model_int8_zlib"
    log(
        f"{qtag}:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
    )

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.load_state_dict({k: v.to(DEVICE) for k, v in quant_flat.items()}, strict=True)
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    rt_tag = "final_int8_awq_zlib_roundtrip" if args.awq_quant else "final_int8_zlib_roundtrip"
    log(f"{rt_tag} val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"{rt_tag}_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
