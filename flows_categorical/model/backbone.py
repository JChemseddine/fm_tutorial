# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Discrete Distributions (arXiv:2605.05629)

"""DiT backbone with adaLN modulation, flash attention, and rotary embeddings.

Based on the MDLM codebase by Sahoo et al. (2024):
  https://github.com/kuleshov-group/mdlm

Two wrapper classes:
- ContinuousTransformer: continuous embeddings h_t -> h_prime (vMF, geodesic, VE)
- MaskedTransformer: token indices -> logits (masked diffusion)

Both share the same DDiTBlock architecture. adaLN is always active —
when time_conditioning=False, sigma=0 provides learnable scale/shift/gate.
"""

import math
import typing
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

# Try flash attention; fall back to standard if unavailable
try:
    import flash_attn
    import flash_attn.layers.rotary
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


# ─── JIT-fused helpers ────────────────────────────────────────────────────────

def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale) + shift


# ─── Rotary embeddings ────────────────────────────────────────────────────────

class Rotary(nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # Identity transform on v
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)
        return self.cos_cached, self.sin_cached


def apply_rotary_pos_emb_flash(qkv, cos, sin):
    """Apply rotary via flash_attn (in-place, fast)."""
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_standard(qkv, cos, sin):
    """Apply rotary without flash_attn dependency."""
    # qkv: (B, L, 3, H, D)
    cos_q = cos[0, :, 0, 0, :]  # (L, D)
    sin_q = sin[0, :, 0, 0, :]
    for i in range(3):  # q, k, v
        x = qkv[:, :, i]  # (B, L, H, D)
        c = cos_q[None, :, None, :]  # (1, L, 1, D)
        s = sin_q[None, :, None, :]
        if i < 2:  # only rotate q and k
            qkv[:, :, i] = x * c + rotate_half(x) * s
    return qkv


# ─── LayerNorm ────────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


# ─── Timestep embedder ────────────────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# ─── DDiTBlock (core) ─────────────────────────────────────────────────────────

class DDiTBlock(nn.Module):
    """DiT block with adaLN modulation (from MDLM).

    adaLN provides learnable per-block scale/shift/gate via conditioning vector c.
    When time_conditioning=False, c comes from sigma=0 (constant) — the adaLN
    layers still provide useful learnable affine parameters.
    """

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # Attention
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)

        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            if HAS_FLASH_ATTN:
                qkv = apply_rotary_pos_emb_flash(
                    qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
                )
            else:
                qkv = apply_rotary_pos_emb_standard(
                    qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
                )

        if HAS_FLASH_ATTN:
            qkv_packed = rearrange(qkv, "b s ... -> (b s) ...")
            if seqlens is None:
                cu_seqlens = torch.arange(
                    0, (batch_size + 1) * seq_len, step=seq_len,
                    dtype=torch.int32, device=qkv_packed.device,
                )
            else:
                cu_seqlens = seqlens.cumsum(-1)
            x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv_packed, cu_seqlens, seq_len, 0.0, causal=False
            )
            x = rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)
        else:
            # Standard attention fallback
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, L, H, D)
            q = rearrange(q, "b s h d -> b h s d")
            k = rearrange(k, "b s h d -> b h s d")
            v = rearrange(v, "b s h d -> b h s d")
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            x = rearrange(x, "b h s d -> b s (h d)")

        x = bias_dropout_scale_fn(
            self.attn_out(x), None, gate_msa, x_skip, self.dropout
        )

        # MLP
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout,
        )
        return x


# ─── Final output layer ──────────────────────────────────────────────────────

class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ─── Embedding layer ─────────────────────────────────────────────────────────

class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


# ═══════════════════════════════════════════════════════════════════════════════
#  Wrapper 1: ContinuousTransformer (for vMF, geodesic, VE flow)
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousTransformer(nn.Module):
    """MDLM DiT adapted for spherical flow matching.

    Continuous embedding transformer:
      Input: h_t (B, L, embed_dim) on S^{r-1}
      Output: h_prime (B, L, embed_dim) on S^{r-1}
      Has: W_E, get_W_E(), compute_logits(), velocity_head
    """

    def __init__(self, vocab_size: int, config: Union[Dict, object]):
        super().__init__()
        if isinstance(config, dict):
            get = lambda key: config[key]
        else:
            get = lambda key: getattr(config, key)

        self.vocab_size = vocab_size
        self.hidden_size = get("hidden_size")
        self.embed_dim = get("embed_dim")
        self.sequence_length = get("sequence_length")
        self.learned_tau = (get("learned_tau") if not isinstance(config, dict)
                           else config.get("learned_tau", False))
        self.per_position_bias = (getattr(config, "per_position_bias", False)
                                  if not isinstance(config, dict)
                                  else config.get("per_position_bias", False))

        self.time_conditioning = (getattr(config, "time_conditioning", False)
                                   if not isinstance(config, dict)
                                   else config.get("time_conditioning", False))

        scale_embeddings = (getattr(config, "scale_embeddings", False)
                            if not isinstance(config, dict)
                            else config.get("scale_embeddings", False))
        self.embed_scale = math.sqrt(self.embed_dim) if scale_embeddings else 1.0

        cond_dim = getattr(config, "cond_dim", None) if not isinstance(config, dict) else config.get("cond_dim", None)
        if cond_dim is None:
            cond_dim = 128  # MDLM default
        # Input projection: R^embed_dim -> R^hidden_size (identity when equal)
        if self.embed_dim != self.hidden_size:
            self.input_proj = nn.Linear(self.embed_dim, self.hidden_size)
        else:
            self.input_proj = None

        # Clue mask embedding
        self.clue_embed = nn.Embedding(2, self.hidden_size)

        # Timestep embedder (always created — provides learned conditioning vector)
        self.sigma_map = TimestepEmbedder(cond_dim)

        # Rotary
        self.rotary_emb = Rotary(self.hidden_size // get("n_heads"))

        # DDiT blocks (always with adaLN — provides learnable scale/shift/gate)
        self.blocks = nn.ModuleList([
            DDiTBlock(
                self.hidden_size, get("n_heads"), cond_dim,
                dropout=get("dropout"),
            )
            for _ in range(get("n_blocks"))
        ])

        # Output layer
        self.output_layer = DDitFinalLayer(
            self.hidden_size, self.embed_dim, cond_dim,
        )

        # Embedding matrix W_E (unit-norm columns via get_W_E())
        self.W_E = nn.Parameter(torch.empty(self.embed_dim, vocab_size))
        nn.init.normal_(self.W_E, std=1.0 / math.sqrt(self.embed_dim))

        # Logit biases
        self.logit_bias = nn.Parameter(torch.zeros(vocab_size))
        if self.per_position_bias:
            self.logit_pos_bias = nn.Parameter(torch.zeros(self.sequence_length, vocab_size))
        else:
            self.logit_pos_bias = None

        # Velocity head
        velocity_head_hidden = (
            get("velocity_head_hidden")
            if not isinstance(config, dict)
            else config.get("velocity_head_hidden", 0)
        )
        if velocity_head_hidden > 0:
            self.velocity_head = nn.Sequential(
                nn.Linear(self.hidden_size, velocity_head_hidden),
                nn.Softplus(),
                nn.Linear(velocity_head_hidden, self.embed_dim),
            )
        else:
            self.velocity_head = None

    def get_W_E(self) -> Tensor:
        return F.normalize(self.W_E, dim=0) * self.embed_scale

    def compute_logits(
        self, h_prime: Tensor, W_E=None,
    ) -> Tensor:
        if W_E is None:
            W_E = self.get_W_E()
        logits = torch.matmul(h_prime, W_E)
        logits = logits + self.logit_bias
        if self.logit_pos_bias is not None:
            logits = logits + self.logit_pos_bias
        return logits

    def predict_velocity(self, backbone: Tensor) -> Tensor:
        return self.velocity_head(backbone)

    def forward(
        self,
        h_t: Tensor,
        clue_mask: Optional[Tensor] = None,
        sigma: Optional[Tensor] = None,
        return_backbone: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        B = h_t.shape[0]
        device = h_t.device

        x = self.input_proj(h_t.float()) if self.input_proj is not None else h_t.float()

        if clue_mask is not None:
            x = x + self.clue_embed(clue_mask.long())

        # Time conditioning (sigma=0 when not provided — adaLN still provides learnable params)
        if sigma is None:
            sigma = torch.zeros(B, device=device)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c)
            backbone = x.float()
            out = self.output_layer(x, c).float()

        h_prime = out if self.learned_tau else F.normalize(out, dim=-1)

        if return_backbone:
            D_hat = self.velocity_head(backbone) if self.velocity_head is not None else None
            return h_prime, D_hat
        return h_prime


# ═══════════════════════════════════════════════════════════════════════════════
#  Wrapper 2: MaskedTransformer (for masked diffusion)
# ═══════════════════════════════════════════════════════════════════════════════

class MaskedTransformer(nn.Module):
    """MDLM DiT adapted for masked discrete flow matching.

    Same interface as MaskedTransformer:
      Input: token indices (B, L) with [MASK] tokens
      Output: logits (B, L, V) over vocabulary
    """

    def __init__(self, vocab_size: int, config: Union[Dict, object]):
        super().__init__()
        if isinstance(config, dict):
            get = lambda key: config[key]
        else:
            get = lambda key: getattr(config, key)

        self.vocab_size = vocab_size
        self.hidden_size = get("hidden_size")
        self.embed_dim = get("embed_dim")
        self.sequence_length = get("sequence_length")
        self.mask_token_id = vocab_size
        self.per_position_bias = (getattr(config, "per_position_bias", False)
                                  if not isinstance(config, dict)
                                  else config.get("per_position_bias", False))

        # Token embedding (vocab_size+1 for [MASK])
        self.vocab_embed = EmbeddingLayer(self.hidden_size, vocab_size + 1)

        # Clue mask embedding
        self.clue_embed = nn.Embedding(2, self.hidden_size)

        cond_dim = getattr(config, "cond_dim", None) if not isinstance(config, dict) else config.get("cond_dim", None)
        if cond_dim is None:
            cond_dim = 128

        # Timestep embedder (sigma=0 always — adaLN provides learnable params)
        self.sigma_map = TimestepEmbedder(cond_dim)

        # Rotary
        self.rotary_emb = Rotary(self.hidden_size // get("n_heads"))

        # DDiT blocks (adaLN with sigma=0 — learnable scale/shift/gate)
        self.blocks = nn.ModuleList([
            DDiTBlock(
                self.hidden_size, get("n_heads"), cond_dim,
                dropout=get("dropout"),
            )
            for _ in range(get("n_blocks"))
        ])

        # Output layer
        self.output_layer = DDitFinalLayer(
            self.hidden_size, vocab_size, cond_dim,
        )

        # Logit biases
        self.logit_bias = nn.Parameter(torch.zeros(vocab_size))
        if self.per_position_bias:
            self.logit_pos_bias = nn.Parameter(torch.zeros(self.sequence_length, vocab_size))
        else:
            self.logit_pos_bias = None

    def forward(
        self,
        x_t: Tensor,
        clue_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B = x_t.shape[0]
        device = x_t.device

        x = self.vocab_embed(x_t)

        if clue_mask is not None:
            x = x + self.clue_embed(clue_mask.long())

        sigma = torch.zeros(B, device=device)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c)
            logits = self.output_layer(x, c).float()

        logits = logits + self.logit_bias
        if self.logit_pos_bias is not None:
            logits = logits + self.logit_pos_bias
        return logits