"""
Phase 3: Positional Embeddings
Supports any model in the registry (default: GPT-2).

Questions:
  1. What geometry does the learned positional matrix have?
  2. How separate are token and position subspaces?
  3. How much does adding position perturb the exact Phase 2 token probes?

Usage: poetry run python phase3-positional-embeddings/explore.py [--model MODEL]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.models import add_model_arg, load_model
from common.probes import PHASE2_ANALOGIES, PHASE2_NEIGHBOR_PROBES, token_for_concept, validate_model_probes


GAP_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
INTERACTION_POSITIONS = (0, 1, 10, 50, 100, 200, 500)
HEATMAP_STEP = 8
NN_K = 20


parser = argparse.ArgumentParser(description="Phase 3: Positional embedding geometry")
add_model_arg(parser)
args = parser.parse_args()

m = load_model(args.model)
OUT = Path(__file__).parent / 'results' / m.slug
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

if not m.has_learned_positions or m.pos_emb is None:
    skip_reason = (
        f"{m.name} uses {m.position_type} position handling, so there is no learned absolute "
        "position matrix to analyze in this phase."
    )
    print(skip_reason)
    save_results({
        **base_results,
        'compatible': False,
        'skip_reason': skip_reason,
    })
    print(f"\nResults saved to {OUT}/")
    print("Run charts.py for a summary page.")
    raise SystemExit(0)


pos = m.pos_emb
emb = m.emb
norms_tok = m.norms
normed_emb = m.normed_emb
n_positions = pos.shape[0]
pos_norms = np.linalg.norm(pos, axis=1)
normed_pos = pos / (pos_norms[:, None] + 1e-10)

print(f"Position matrix: {n_positions} positions x {m.hidden_dim} dimensions")

# Shared exact probes
probe_concepts = tuple(dict.fromkeys(
    [concept_id for concept_id, _ in PHASE2_NEIGHBOR_PROBES]
    + [item for _, a, b, c, d in PHASE2_ANALOGIES for item in (a, b, c, d)]
))
concept_indices = validate_model_probes(m, probe_concepts)

# ============================================================
# 1. POSITION NORMS
# ============================================================
print("\n" + "=" * 60)
print("1. POSITION NORMS")
print("=" * 60)

key_positions = sorted({
    0, 1, 2, 4, 8, 16, 32, 64, 128,
    n_positions // 2,
    n_positions - 4,
    n_positions - 2,
    n_positions - 1,
})
key_positions = [p for p in key_positions if 0 <= p < n_positions]

for p in key_positions[:10]:
    print(f"  pos {p:4d}: norm={pos_norms[p]:.4f}")
if key_positions[-1] not in key_positions[:10]:
    print(f"  pos {key_positions[-1]:4d}: norm={pos_norms[key_positions[-1]]:.4f}")

quartile_size = max(n_positions // 4, 1)
quartiles = []
for start in range(0, n_positions, quartile_size):
    end = min(start + quartile_size, n_positions)
    if start >= end:
        continue
    chunk = pos_norms[start:end]
    quartiles.append({
        'start': int(start),
        'end': int(end - 1),
        'mean': float(chunk.mean()),
        'std': float(chunk.std()),
    })
    if len(quartiles) == 4:
        break

# ============================================================
# 2. POSITION SIMILARITY + PCA
# ============================================================
print("\n" + "=" * 60)
print("2. POSITION SIMILARITY + PCA")
print("=" * 60)

gap_stats = {}
gaps = sorted({gap for gap in GAP_CANDIDATES if gap < n_positions} | {n_positions // 2, n_positions - 1})
for gap in gaps:
    cos = np.sum(normed_pos[:-gap] * normed_pos[gap:], axis=1)
    gap_stats[str(gap)] = {
        'count': int(len(cos)),
        'mean': float(cos.mean()),
        'std': float(cos.std()),
        'min': float(cos.min()),
        'max': float(cos.max()),
    }
    print(f"  gap {gap:4d}: cosine={cos.mean():.4f} +/- {cos.std():.4f}")

pos_centered = pos - pos.mean(axis=0)
u_pos, s_pos, vt_pos = np.linalg.svd(pos_centered, full_matrices=False)
pos_var = (s_pos ** 2) / np.maximum((s_pos ** 2).sum(), 1e-10)
pos_cumulative = np.cumsum(pos_var)
pos_coords = u_pos[:, :3] * s_pos[:3]
pc_spearman = {
    f'pc{i + 1}': float(stats.spearmanr(np.arange(n_positions), pos_coords[:, i])[0])
    for i in range(pos_coords.shape[1])
}

var_pos_by_dim = pos.var(axis=0)
top_var_dims = np.argsort(var_pos_by_dim)[-5:][::-1]
periodicity = []
for dim in top_var_dims:
    signal = pos[:, dim] - pos[:, dim].mean()
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.argsort(spectrum[1:])[-3:][::-1] + 1
    periodicity.append({
        'dim': int(dim),
        'variance': float(var_pos_by_dim[dim]),
        'top_periods': [float(n_positions / f) for f in freqs],
        'top_frequencies': [int(f) for f in freqs],
    })

# ============================================================
# 3. TOKEN vs POSITION SUBSPACES
# ============================================================
print("\n" + "=" * 60)
print("3. TOKEN vs POSITION SUBSPACES")
print("=" * 60)

tok_centered = emb - emb.mean(axis=0)
_u_tok, _s_tok, vt_tok = np.linalg.svd(tok_centered, full_matrices=False)
_u_pos2, _s_pos2, vt_pos2 = np.linalg.svd(pos_centered, full_matrices=False)

subspace_alignment = {}
for k in (5, 10, 50):
    if k > vt_pos2.shape[0] or k > vt_tok.shape[0]:
        continue
    alignment = np.linalg.norm(vt_tok[:k] @ vt_pos2[:k].T, 'fro') / np.sqrt(k)
    random_baseline = np.sqrt(k / m.hidden_dim)
    subspace_alignment[str(k)] = {
        'alignment': float(alignment),
        'random_baseline': float(random_baseline),
    }
    print(f"  top-{k:2d}: alignment={alignment:.4f} vs random={random_baseline:.4f}")

var_tok_by_dim = emb.var(axis=0)
var_corr = float(stats.spearmanr(var_tok_by_dim, var_pos_by_dim)[0])
top_tok_dims = np.argsort(var_tok_by_dim)[-10:][::-1]
top_pos_dims = np.argsort(var_pos_by_dim)[-10:][::-1]
top_dim_overlap = sorted(set(top_tok_dims) & set(top_pos_dims))

# ============================================================
# 4. TOKEN-POSITION INTERACTION
# ============================================================
print("\n" + "=" * 60)
print("4. TOKEN-POSITION INTERACTION")
print("=" * 60)

interaction_positions = sorted({p for p in INTERACTION_POSITIONS if p < n_positions} | {n_positions - 1})
combined_vectors = {}
combined_norms = {}


def nearest_neighbors_at_position(position, idx, k=NN_K):
    combined = combined_vectors[position]
    norms = combined_norms[position]
    vec = combined[idx]
    cos = (combined @ vec) / (norms * np.linalg.norm(vec) + 1e-10)
    cos[idx] = -2
    nn = np.argsort(cos)[-k:][::-1]
    return nn, cos


def analogy_at_position(position, ia, ib, ic, k=5):
    combined = combined_vectors[position]
    norms = combined_norms[position]
    vec = combined[ib] - combined[ia] + combined[ic]
    vec_norm = np.linalg.norm(vec)
    cos = (combined @ vec) / (norms * vec_norm + 1e-10)
    cos[[ia, ib, ic]] = -2
    nn = np.argsort(cos)[-k:][::-1]
    return nn, cos


for position in interaction_positions:
    combined = emb + pos[position]
    combined_vectors[position] = combined
    combined_norms[position] = np.linalg.norm(combined, axis=1)

baseline_neighbors = {}
neighbor_stability = {}
token_drift = {}
for concept_id in probe_concepts:
    idx = concept_indices[concept_id]
    nn_idx, nn_cos = nearest_neighbors_at_position(0, idx)
    baseline_neighbors[concept_id] = set(nn_idx.tolist())
    neighbor_stability[concept_id] = {}
    token_drift[concept_id] = {}
    baseline_vec = combined_vectors[0][idx]
    baseline_norm = np.linalg.norm(baseline_vec)
    for position in interaction_positions:
        pos_nn_idx, pos_nn_cos = nearest_neighbors_at_position(position, idx)
        pos_nn_set = set(pos_nn_idx.tolist())
        overlap = len(pos_nn_set & baseline_neighbors[concept_id])
        union = len(pos_nn_set | baseline_neighbors[concept_id])
        vec = combined_vectors[position][idx]
        drift = float(np.dot(baseline_vec, vec) / (baseline_norm * np.linalg.norm(vec) + 1e-10))
        neighbor_stability[concept_id][str(position)] = {
            'overlap': int(overlap),
            'jaccard': float(overlap / union if union else 1.0),
            'top5': [
                {
                    'idx': int(i),
                    'token': repr(m.labels[i]),
                    'cosine': float(pos_nn_cos[i]),
                }
                for i in pos_nn_idx[:5]
            ],
        }
        token_drift[concept_id][str(position)] = drift

analogy_stability = {}
for label, a_concept, b_concept, c_concept, expected_concept in PHASE2_ANALOGIES:
    ia = concept_indices[a_concept]
    ib = concept_indices[b_concept]
    ic = concept_indices[c_concept]
    expected_idx = concept_indices[expected_concept]
    key = f"{a_concept}:{b_concept}::{c_concept}:?"
    analogy_stability[key] = {'label': label, 'positions': {}}
    for position in interaction_positions:
        nn, cos = analogy_at_position(position, ia, ib, ic)
        analogy_stability[key]['positions'][str(position)] = {
            'top1_idx': int(nn[0]),
            'top1_token': repr(m.labels[nn[0]]),
            'top1_cosine': float(cos[nn[0]]),
            'expected_idx': int(expected_idx),
            'expected_token': repr(token_for_concept(m.slug, expected_concept)),
            'expected_rank': int(np.where(nn == expected_idx)[0][0] + 1) if expected_idx in nn else None,
            'hit_top1': bool(nn[0] == expected_idx),
            'hit_top5': bool(expected_idx in nn),
        }

mean_jaccard_by_position = {}
mean_drift_by_position = {}
analogy_top1_by_position = {}
analogy_top5_by_position = {}
for position in interaction_positions:
    pos_key = str(position)
    jaccards = [neighbor_stability[concept_id][pos_key]['jaccard'] for concept_id in probe_concepts]
    drifts = [token_drift[concept_id][pos_key] for concept_id in probe_concepts]
    top1_hits = [analogy_stability[key]['positions'][pos_key]['hit_top1'] for key in analogy_stability]
    top5_hits = [analogy_stability[key]['positions'][pos_key]['hit_top5'] for key in analogy_stability]
    mean_jaccard_by_position[pos_key] = float(np.mean(jaccards))
    mean_drift_by_position[pos_key] = float(np.mean(drifts))
    analogy_top1_by_position[pos_key] = float(np.mean(top1_hits))
    analogy_top5_by_position[pos_key] = float(np.mean(top5_hits))
    print(
        f"  pos {position:4d}: mean_jaccard={mean_jaccard_by_position[pos_key]:.3f}, "
        f"mean_drift={mean_drift_by_position[pos_key]:.3f}, "
        f"analogy_top1={analogy_top1_by_position[pos_key]:.2f}"
    )

# ============================================================
# SAVE RESULTS + ARRAYS
# ============================================================
heatmap_positions = list(range(0, n_positions, HEATMAP_STEP))
if heatmap_positions[-1] != n_positions - 1:
    heatmap_positions.append(n_positions - 1)
heatmap_embs = pos[heatmap_positions]
heatmap_norms = pos_norms[heatmap_positions]
heatmap_normed = heatmap_embs / (heatmap_norms[:, None] + 1e-10)
heatmap_cos = heatmap_normed @ heatmap_normed.T

np.save(OUT / 'position_norms.npy', pos_norms)
np.save(OUT / 'position_pca_coords.npy', pos_coords)
np.save(OUT / 'position_cosine_matrix.npy', heatmap_cos)

results = {
    **base_results,
    'compatible': True,
    'position_norms': {
        'count': int(n_positions),
        'mean': float(pos_norms.mean()),
        'std': float(pos_norms.std()),
        'min': float(pos_norms.min()),
        'max': float(pos_norms.max()),
        'median': float(np.median(pos_norms)),
        'min_position': int(pos_norms.argmin()),
        'max_position': int(pos_norms.argmax()),
        'percentiles': {str(p): float(np.percentile(pos_norms, p)) for p in (1, 5, 10, 25, 75, 90, 95, 99)},
        'quartiles': quartiles,
        'key_positions': {str(p): float(pos_norms[p]) for p in key_positions},
    },
    'position_similarity': {
        'gaps': gap_stats,
        'sampled_heatmap_positions': heatmap_positions,
        'adjacent_mean': gap_stats['1']['mean'] if '1' in gap_stats else None,
        'halfway_mean': gap_stats[str(n_positions // 2)]['mean'] if str(n_positions // 2) in gap_stats else None,
    },
    'pca': {
        'threshold_components': {
            str(t): int(np.searchsorted(pos_cumulative, t) + 1)
            for t in (0.5, 0.8, 0.9, 0.95, 0.99)
        },
        'variance_at_k': {
            str(k): float(pos_cumulative[min(k - 1, len(pos_cumulative) - 1)])
            for k in (1, 2, 3, 5, 10, 20, 50, 100)
            if k <= len(pos_cumulative)
        },
        'pc_spearman_with_position': pc_spearman,
    },
    'periodicity': {
        'top_variance_dimensions': periodicity,
    },
    'token_position_subspace': {
        'variance_spearman': var_corr,
        'top10_variance_overlap': [int(dim) for dim in top_dim_overlap],
        'top10_token_dims': [int(dim) for dim in top_tok_dims],
        'top10_position_dims': [int(dim) for dim in top_pos_dims],
        'principal_subspace_alignment': subspace_alignment,
    },
    'token_position_interaction': {
        'positions': interaction_positions,
        'probe_count': len(probe_concepts),
        'neighbor_k': NN_K,
        'mean_jaccard_by_position': mean_jaccard_by_position,
        'mean_token_drift_by_position': mean_drift_by_position,
        'analogy_top1_by_position': analogy_top1_by_position,
        'analogy_top5_by_position': analogy_top5_by_position,
        'neighbor_stability': {
            concept_id: {
                'token': token_for_concept(m.slug, concept_id),
                'center_idx': int(concept_indices[concept_id]),
                'positions': positions_data,
            }
            for concept_id, positions_data in neighbor_stability.items()
        },
        'token_drift': {
            concept_id: {
                'token': token_for_concept(m.slug, concept_id),
                'center_idx': int(concept_indices[concept_id]),
                'positions': position_scores,
            }
            for concept_id, position_scores in token_drift.items()
        },
        'analogy_stability': analogy_stability,
    },
}

save_results(results)
print(f"\nResults saved to {OUT}/")
print("Run charts.py for visualizations.")
