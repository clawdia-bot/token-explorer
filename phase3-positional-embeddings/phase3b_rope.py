"""
Phase 3B: RoPE geometry and first-layer positional effects.

Questions:
  1. What relative-position kernel does each RoPE model hard-code?
  2. What frequency inventory does each model allocate across rotary pairs?
  3. How much does RoPE change first-layer q/k geometry for exact probe tokens?

Usage: poetry run python phase3-positional-embeddings/phase3b_rope.py [--model MODEL]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.models import add_model_arg, load_model
from common.probes import COMPARISON_NEIGHBOR_PROBES, token_for_concept, validate_model_probes
from common.rope import apply_rope, relative_kernel, rope_cos_sin, rope_inv_freq, rope_metadata_from_config


GAP_CANDIDATES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
POSITION_CANDIDATES = GAP_CANDIDATES
TOPK = 3


parser = argparse.ArgumentParser(description="Phase 3B: RoPE operator analysis")
add_model_arg(parser)
args = parser.parse_args()

m = load_model(args.model)
OUT = Path(__file__).parent / 'rope-results' / m.slug
OUT.mkdir(parents=True, exist_ok=True)


def save_results(payload):
    with open(OUT / 'results.json', 'w') as f:
        json.dump(payload, f, indent=2)


base_results = {
    'model': {
        'name': m.name,
        'slug': m.slug,
        'hf_id': m.hf_id,
        'vocab_size': m.vocab_size,
        'hidden_dim': m.hidden_dim,
        'tied': m.tied,
        'position_type': m.position_type,
        'max_positions': m.max_positions,
        'has_learned_positions': m.has_learned_positions,
    }
}

if m.position_type != 'rope':
    skip_reason = f"{m.name} does not use RoPE; Phase 3B is for rotary position models."
    print(skip_reason)
    save_results({
        **base_results,
        'compatible': False,
        'skip_reason': skip_reason,
    })
    print(f"\nResults saved to {OUT}/")
    print("Run phase3b_rope_charts.py for a summary page.")
    raise SystemExit(0)


config = AutoConfig.from_pretrained(m.hf_id)
rope_meta = rope_metadata_from_config(config)
if rope_meta is None:
    raise ValueError(f"RoPE metadata was expected for {m.slug} but could not be resolved")

print(f"RoPE: theta={rope_meta.rope_theta:.0f}, head_dim={rope_meta.head_dim}, rotary_dim={rope_meta.rotary_dim}")

probe_concepts = tuple(COMPARISON_NEIGHBOR_PROBES)
probe_indices_map = validate_model_probes(m, probe_concepts)
probe_indices = np.array([probe_indices_map[concept_id] for concept_id in probe_concepts], dtype=np.int64)


def resolve_positions(limit: int) -> list[int]:
    positions = sorted({p for p in POSITION_CANDIDATES if p < limit} | {limit - 1})
    return positions


positions = resolve_positions(rope_meta.max_position_embeddings or 2048)
gaps = positions

# ============================================================
# 1. OPERATOR-LEVEL RoPE GEOMETRY
# ============================================================
print("\n" + "=" * 60)
print("1. RoPE OPERATOR GEOMETRY")
print("=" * 60)

inv_freq = rope_inv_freq(rope_meta)
periods = (2.0 * math.pi) / inv_freq
kernel = relative_kernel(rope_meta, np.array(gaps, dtype=np.int64))
gap_stats = {}
for gap, value in zip(gaps, kernel):
    gap_stats[str(int(gap))] = float(value)
    print(f"  gap {gap:5d}: kernel={value:.4f}")

first_negative_gap = None
for gap, value in zip(gaps[1:], kernel[1:]):
    if value < 0:
        first_negative_gap = int(gap)
        break

# ============================================================
# 2. FIRST-LAYER q/k UNDER RoPE
# ============================================================
print("\n" + "=" * 60)
print("2. FIRST-LAYER q/k DRIFT")
print("=" * 60)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()


def get_first_layer_qk(full_model, token_indices: np.ndarray):
    with torch.no_grad():
        emb = full_model.get_input_embeddings().weight[token_indices]
        base = getattr(full_model, full_model.base_model_prefix)

        if config.model_type == 'gpt_neox':
            layer = base.layers[0]
            hidden = layer.input_layernorm(emb)
            qkv = layer.attention.query_key_value(hidden)
            qkv = qkv.view(len(token_indices), rope_meta.num_attention_heads, 3, rope_meta.head_dim)
            q = qkv[:, :, 0, :]
            k = qkv[:, :, 1, :]
            return to_numpy(q), to_numpy(k)

        if config.model_type in ('llama', 'qwen2'):
            layer = base.layers[0]
            hidden = layer.input_layernorm(emb)
            attn = layer.self_attn
            q = attn.q_proj(hidden).view(len(token_indices), rope_meta.num_attention_heads, rope_meta.head_dim)
            k = attn.k_proj(hidden).view(len(token_indices), rope_meta.num_key_value_heads, rope_meta.head_dim)
            if rope_meta.num_key_value_groups > 1:
                k = k.repeat_interleave(rope_meta.num_key_value_groups, dim=1)
            return to_numpy(q), to_numpy(k)

    raise ValueError(f"Unsupported RoPE architecture for Phase 3B: {config.model_type}")


full_model = AutoModelForCausalLM.from_pretrained(m.hf_id, dtype=torch.float32)
q_base, k_base = get_first_layer_qk(full_model, probe_indices)
del full_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

probe_score_matrices = []
query_drift = {}
key_drift = {}
score_pearson = {}
score_spearman = {}
topk_overlap = {}
per_concept_query_drift = {concept_id: {} for concept_id in probe_concepts}
per_concept_key_drift = {concept_id: {} for concept_id in probe_concepts}

baseline_scores = np.einsum('ihd,jhd->ij', q_base, k_base) / (math.sqrt(rope_meta.head_dim) * q_base.shape[1])
baseline_flat = baseline_scores[~np.eye(len(probe_concepts), dtype=bool)]


def cosine_last_dim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-10
    return np.sum(a * b, axis=-1) / denom


def row_topk(matrix: np.ndarray, k: int) -> list[set[int]]:
    top = []
    for i in range(matrix.shape[0]):
        row = matrix[i].copy()
        row[i] = -np.inf
        idx = np.argsort(row)[-k:]
        top.append(set(idx.tolist()))
    return top


baseline_topk = row_topk(baseline_scores, TOPK)

for position in positions:
    cos, sin = rope_cos_sin(rope_meta, np.array([position], dtype=np.int64))
    cos = cos[0][None, None, :]
    sin = sin[0][None, None, :]
    q_rot = apply_rope(q_base, cos, sin, rope_meta.rotary_dim)
    k_rot = apply_rope(k_base, cos, sin, rope_meta.rotary_dim)

    q_cos = cosine_last_dim(q_rot, q_base)
    k_cos = cosine_last_dim(k_rot, k_base)
    scores = np.einsum('ihd,jhd->ij', q_rot, k_base) / (math.sqrt(rope_meta.head_dim) * q_rot.shape[1])
    probe_score_matrices.append(scores)

    q_mean = float(q_cos.mean())
    k_mean = float(k_cos.mean())
    query_drift[str(position)] = q_mean
    key_drift[str(position)] = k_mean
    for concept_id, q_val, k_val in zip(probe_concepts, q_cos.mean(axis=1), k_cos.mean(axis=1)):
        per_concept_query_drift[concept_id][str(position)] = float(q_val)
        per_concept_key_drift[concept_id][str(position)] = float(k_val)

    flat = scores[~np.eye(len(probe_concepts), dtype=bool)]
    score_pearson[str(position)] = float(stats.pearsonr(baseline_flat, flat)[0]) if position != 0 else 1.0
    score_spearman[str(position)] = float(stats.spearmanr(baseline_flat, flat)[0]) if position != 0 else 1.0

    current_topk = row_topk(scores, TOPK)
    overlaps = [len(base & cur) / TOPK for base, cur in zip(baseline_topk, current_topk)]
    topk_overlap[str(position)] = float(np.mean(overlaps))

    print(
        f"  pos {position:5d}: q_drift={q_mean:.3f}, k_drift={k_mean:.3f}, "
        f"score_spearman={score_spearman[str(position)]:.3f}"
    )

probe_score_matrices = np.stack(probe_score_matrices, axis=0)
np.save(OUT / 'probe_score_matrices.npy', probe_score_matrices)

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    **base_results,
    'compatible': True,
    'rope': {
        'theta': float(rope_meta.rope_theta),
        'head_dim': int(rope_meta.head_dim),
        'rotary_dim': int(rope_meta.rotary_dim),
        'num_attention_heads': int(rope_meta.num_attention_heads),
        'num_key_value_heads': int(rope_meta.num_key_value_heads),
        'num_key_value_groups': int(rope_meta.num_key_value_groups),
        'partial_rotary_factor': float(rope_meta.partial_rotary_factor),
        'inv_freq_count': int(len(inv_freq)),
        'periods': {
            'min': float(periods.min()),
            'median': float(np.median(periods)),
            'max': float(periods.max()),
            'samples': [float(x) for x in periods[: min(8, len(periods))]],
        },
    },
    'relative_kernel': {
        'gaps': {str(int(gap)): float(value) for gap, value in zip(gaps, kernel)},
        'first_negative_gap': first_negative_gap,
    },
    'qk_rotation': {
        'positions': positions,
        'probe_count': len(probe_concepts),
        'probe_labels': [token_for_concept(m.slug, concept_id) for concept_id in probe_concepts],
        'mean_query_drift_by_position': query_drift,
        'mean_key_drift_by_position': key_drift,
        'score_pearson_by_position': score_pearson,
        'score_spearman_by_position': score_spearman,
        'top3_overlap_by_position': topk_overlap,
        'per_concept_query_drift': {
            concept_id: {
                'token': token_for_concept(m.slug, concept_id),
                'center_idx': int(probe_indices_map[concept_id]),
                'positions': values,
            }
            for concept_id, values in per_concept_query_drift.items()
        },
        'per_concept_key_drift': {
            concept_id: {
                'token': token_for_concept(m.slug, concept_id),
                'center_idx': int(probe_indices_map[concept_id]),
                'positions': values,
            }
            for concept_id, values in per_concept_key_drift.items()
        },
    },
}

save_results(results)
print(f"\nResults saved to {OUT}/")
print("Run phase3b_rope_charts.py for visualizations.")
