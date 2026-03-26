"""Helpers for analyzing rotary position embeddings (RoPE)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RopeMetadata:
    model_type: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    partial_rotary_factor: float
    rope_theta: float
    max_position_embeddings: int | None

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


def rope_metadata_from_config(config) -> RopeMetadata | None:
    """Build RoPE metadata from a Hugging Face config, or return None."""
    rope_theta = getattr(config, 'rope_theta', None)
    if rope_theta is None:
        return None

    hidden_size = getattr(config, 'hidden_size')
    num_attention_heads = getattr(config, 'num_attention_heads')
    num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads) or num_attention_heads
    head_dim = getattr(config, 'head_dim', None) or (hidden_size // num_attention_heads)

    partial_rotary_factor = getattr(config, 'partial_rotary_factor', None)
    if partial_rotary_factor is None:
        partial_rotary_factor = getattr(config, 'rotary_pct', None)
    if partial_rotary_factor is None:
        partial_rotary_factor = 1.0

    rotary_dim = int(round(head_dim * partial_rotary_factor))
    rotary_dim = max(2, min(head_dim, rotary_dim - (rotary_dim % 2)))

    return RopeMetadata(
        model_type=config.model_type,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        partial_rotary_factor=float(partial_rotary_factor),
        rope_theta=float(rope_theta),
        max_position_embeddings=getattr(config, 'max_position_embeddings', None),
    )


def rope_inv_freq(meta: RopeMetadata) -> np.ndarray:
    """Return the inverse-frequency vector used by standard RoPE."""
    return 1.0 / (meta.rope_theta ** (np.arange(0, meta.rotary_dim, 2, dtype=np.float64) / meta.rotary_dim))


def rope_cos_sin(meta: RopeMetadata, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cos/sin tensors in the same duplicated layout used by HF apply_rotary_pos_emb."""
    positions = np.asarray(positions, dtype=np.float64)
    freqs = np.outer(positions, rope_inv_freq(meta))
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotate the two halves of the last dimension."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, rotary_dim: int) -> np.ndarray:
    """Apply RoPE to the leading rotary_dim of the last axis."""
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return np.concatenate([x_embed, x_pass], axis=-1)


def relative_kernel(meta: RopeMetadata, gaps: np.ndarray) -> np.ndarray:
    """Average operator-induced cosine at each relative offset."""
    gaps = np.asarray(gaps, dtype=np.float64)
    inv_freq = rope_inv_freq(meta)
    return np.cos(np.outer(gaps, inv_freq)).mean(axis=1)
