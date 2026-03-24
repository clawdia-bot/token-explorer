"""
Token Embedding Space Geometry Explorer
Supports any model in the registry (default: GPT-2).

Usage: poetry run python phase1-norms-and-structure/explore.py [--model MODEL]
"""

import argparse
import numpy as np
from scipy import stats
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.models import load_model, add_model_arg
from common.tokenutils import categorize

parser = argparse.ArgumentParser(description="Phase 1: Token embedding space geometry")
add_model_arg(parser)
args = parser.parse_args()

m = load_model(args.model)
emb, tok, tokens, labels, norms = m.emb, m.tokenizer, m.tokens, m.labels, m.norms

print(f"Embedding matrix: {m.vocab_size} tokens x {m.hidden_dim} dimensions ({emb.dtype})")

# Precompute shared quantities
mean_emb = emb.mean(axis=0)
mean_norm = np.linalg.norm(mean_emb)
lengths = np.array([len(t) for t in tokens])

# Output directory
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', m.slug)
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. NORMS AND FREQUENCY
# ============================================================
print("\n" + "="*60)
print("1. NORMS AND FREQUENCY — rarity writes itself into geometry")
print("="*60)

smallest = np.argsort(norms)[:20]
largest = np.argsort(norms)[-20:][::-1]
hist_counts, hist_edges = np.histogram(norms, bins=50)

r_idx, p_idx = stats.pearsonr(np.arange(len(norms)), norms)
r_len, p_len = stats.pearsonr(lengths, norms)

print(f"  BPE index vs norm correlation: r={r_idx:.3f} (p={p_idx:.2e})")
print(f"  Norms: mean={norms.mean():.3f} +/- {norms.std():.3f}, range [{norms.min():.3f}, {norms.max():.3f}]")
print(f"  Smallest: {repr(labels[smallest[0]])} ({norms[smallest[0]]:.3f})")
print(f"  Largest:  {repr(labels[largest[0]])} ({norms[largest[0]]:.3f})")
print(f"  Token length vs norm: r={r_len:.3f} (shorter tokens tend toward higher frequency)")

# ============================================================
# 2. ANISOTROPY
# ============================================================
print("\n" + "="*60)
print("2. ANISOTROPY — do embeddings point the same direction?")
print("="*60)

rng = np.random.RandomState(42)
n_sample = 5000
idx = rng.choice(m.vocab_size, n_sample, replace=False)
sample = emb[idx]
sample_norms = norms[idx]

sample_normed = sample / (sample_norms[:, None] + 1e-10)

cos_matrix = sample_normed @ sample_normed.T
triu_idx = np.triu_indices(n_sample, k=1)
cos_pairs = cos_matrix[triu_idx]

cos_hist, cos_edges = np.histogram(cos_pairs, bins=50)

print(f"  Mean pairwise cosine: {cos_pairs.mean():.3f} +/- {cos_pairs.std():.3f} ({n_sample} random tokens)")

# ============================================================
# 3. EFFECTIVE DIMENSIONALITY (PCA)
# ============================================================
print("\n" + "="*60)
print("3. EFFECTIVE DIMENSIONALITY — how much space is actually used?")
print("="*60)

emb_centered = emb - mean_emb
_, S, _ = np.linalg.svd(emb_centered, full_matrices=False)
explained_var = S**2 / (m.vocab_size - 1)
total_var = explained_var.sum()
explained_ratio = explained_var / total_var
cumulative = np.cumsum(explained_ratio)

pr = (explained_var.sum())**2 / (explained_var**2).sum()

p = explained_var / explained_var.sum()
entropy = -np.sum(p * np.log(p + 1e-30))
eff_dim_entropy = np.exp(entropy)

thresholds = {}
for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
    thresholds[threshold] = int(np.searchsorted(cumulative, threshold) + 1)

print(f"  Participation ratio: {pr:.0f} / {m.hidden_dim} dims")
print(f"  Entropy-based effective dims: {eff_dim_entropy:.0f} / {m.hidden_dim}")
print(f"  Variance thresholds: 50% at {thresholds[0.5]} dims, 90% at {thresholds[0.9]}, 99% at {thresholds[0.99]}")
print(f"  PC1 explains just {explained_ratio[0]*100:.2f}% — no single dimension dominates.")

# ============================================================
# 4. ORIGIN vs CENTROID — is the frequency-norm relationship real?
# ============================================================
print("\n" + "="*60)
print("4. ORIGIN vs CENTROID — is the frequency-norm relationship real?")
print("="*60)

dists_to_mean = np.linalg.norm(emb - mean_emb, axis=1)
dist_hist_counts, dist_hist_edges = np.histogram(dists_to_mean, bins=50)
closest_to_mean = np.argsort(dists_to_mean)[:20]
farthest_from_mean = np.argsort(dists_to_mean)[-1]

cos_to_mean = (emb @ mean_emb) / (norms * mean_norm + 1e-10)

# Same correlations, but measured from centroid instead of origin
r_idx_centered, p_idx_centered = stats.pearsonr(np.arange(len(dists_to_mean)), dists_to_mean)
r_len_centered, p_len_centered = stats.pearsonr(lengths, dists_to_mean)

print(f"  From origin:   BPE index vs norm r={r_idx:.3f}, token length vs norm r={r_len:.3f}")
print(f"  From centroid: BPE index vs dist r={r_idx_centered:.3f}, token length vs dist r={r_len_centered:.3f}")
print(f"  Closest to centroid: {repr(labels[closest_to_mean[0]])} (dist={dists_to_mean[closest_to_mean[0]]:.3f})")
print(f"  Farthest from centroid: {repr(labels[farthest_from_mean])} (dist={dists_to_mean[farthest_from_mean]:.3f})")
min_cos_idx = int(cos_to_mean.argmin())
print(f"  Most dissimilar from mean direction: {repr(labels[min_cos_idx])} (cosine to mean={cos_to_mean.min():.3f})")

ratio = abs(r_idx_centered) / abs(r_idx) if abs(r_idx) > 0 else 0
print(f"  -> Centroid correlation is {ratio:.0%} of origin correlation.")

# ============================================================
# 5. TOKEN CATEGORIES
# ============================================================
print("\n" + "="*60)
print("5. TOKEN CATEGORIES — script and type clustering")
print("="*60)

categories = {}
for i in range(m.vocab_size):
    cat = categorize(tok, i)
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(i)

cat_means = {}
for cat, indices in categories.items():
    cat_means[cat] = emb[indices].mean(axis=0)

sorted_cats = sorted(categories.items(), key=lambda x: -len(x[1]))
for cat, indices in sorted_cats[:5]:
    cat_norms = norms[indices]
    print(f"  {cat}: {len(indices)} tokens, mean norm={cat_norms.mean():.3f}")
if len(sorted_cats) > 5:
    print(f"  ... and {len(sorted_cats) - 5} more categories (see results.json)")

# Compute inter-category cosines
cat_names = sorted(cat_means.keys())
inter_cat_cosines = []
for i, c1 in enumerate(cat_names):
    for c2 in cat_names[i+1:]:
        v1, v2 = cat_means[c1], cat_means[c2]
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
        inter_cat_cosines.append({'cat1': c1, 'cat2': c2, 'cosine': cos})

if inter_cat_cosines:
    most_distant = min(inter_cat_cosines, key=lambda x: x['cosine'])
    print(f"  Most distant categories: {most_distant['cat1']} x {most_distant['cat2']} (cosine={most_distant['cosine']:.3f})")

# ============================================================
# SAVE RESULTS
# ============================================================

# Build per-length stats
per_length = {}
for length in range(1, 15):
    mask = lengths == length
    if mask.sum() > 0:
        per_length[str(length)] = {
            'count': int(mask.sum()),
            'mean_norm': float(norms[mask].mean()),
        }

# Build single-char token list
single_char_tokens = []
for i, t in enumerate(tokens):
    if len(t) == 1:
        single_char_tokens.append({
            'idx': int(i), 'norm': float(norms[i]),
            'token': repr(t), 'ord': ord(t),
        })

# Build binned frequency means
bin_size = max(m.vocab_size // 5, 1)
binned_means = []
for start in range(0, m.vocab_size, bin_size):
    end = min(start + bin_size, m.vocab_size)
    binned_means.append({
        'range': f'{start}-{end}',
        'mean_norm': float(norms[start:end].mean()),
        'std_norm': float(norms[start:end].std()),
    })

# PCA variance_at_k — only include valid k values for this model's hidden_dim
variance_at_k = {}
for k in [1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 768, 1024, 2048]:
    if k <= len(cumulative):
        variance_at_k[str(k)] = float(cumulative[k-1])

results = {
    'model': {
        'name': m.name,
        'slug': m.slug,
        'hf_id': m.hf_id,
        'vocab_size': m.vocab_size,
        'hidden_dim': m.hidden_dim,
        'tied': m.tied,
    },
    'shape': list(emb.shape),
    'norms': {
        'mean': float(norms.mean()),
        'std': float(norms.std()),
        'min': float(norms.min()),
        'max': float(norms.max()),
        'median': float(np.median(norms)),
        'min_token': repr(labels[norms.argmin()]),
        'max_token': repr(labels[norms.argmax()]),
        'percentiles': {str(p): float(np.percentile(norms, p)) for p in [1, 5, 10, 25, 75, 90, 95, 99]},
        'top20': [{'idx': int(i), 'norm': float(norms[i]), 'token': repr(labels[i])} for i in largest],
        'bottom20': [{'idx': int(i), 'norm': float(norms[i]), 'token': repr(labels[i])} for i in smallest],
        'histogram': {'counts': hist_counts.tolist(), 'edges': hist_edges.tolist()},
    },
    'origin': {
        'mean_embedding_norm': float(mean_norm),
        'closest_to_mean_top20': [
            {'idx': int(i), 'dist': float(dists_to_mean[i]), 'norm': float(norms[i]), 'token': repr(labels[i])}
            for i in closest_to_mean
        ],
        'cosine_to_mean': {
            'mean': float(cos_to_mean.mean()),
            'std': float(cos_to_mean.std()),
            'min': float(cos_to_mean.min()),
            'max': float(cos_to_mean.max()),
            'min_token': repr(labels[cos_to_mean.argmin()]),
            'max_token': repr(labels[cos_to_mean.argmax()]),
        },
    },
    'centroid': {
        'r_idx': float(r_idx_centered), 'p_idx': float(p_idx_centered),
        'r_len': float(r_len_centered), 'p_len': float(p_len_centered),
        'closest': {'idx': int(closest_to_mean[0]), 'dist': float(dists_to_mean[closest_to_mean[0]]), 'token': repr(labels[closest_to_mean[0]])},
        'farthest': {'idx': int(farthest_from_mean), 'dist': float(dists_to_mean[farthest_from_mean]), 'token': repr(labels[farthest_from_mean])},
        'histogram': {'counts': dist_hist_counts.tolist(), 'edges': dist_hist_edges.tolist()},
    },
    'pca': {
        'total_variance': float(total_var),
        'participation_ratio': float(pr),
        'entropy_effective_dims': float(eff_dim_entropy),
        'top10_singular_values': S[:10].tolist(),
        'top10_explained': float(cumulative[9]),
        'top50_explained': float(cumulative[min(49, len(cumulative)-1)]),
        'top100_explained': float(cumulative[min(99, len(cumulative)-1)]),
        'variance_at_k': variance_at_k,
        'threshold_components': {str(t): int(np.searchsorted(cumulative, t) + 1) for t in [0.5, 0.8, 0.9, 0.95, 0.99]},
    },
    'anisotropy': {
        'mean_cosine': float(cos_pairs.mean()),
        'std_cosine': float(cos_pairs.std()),
        'min_cosine': float(cos_pairs.min()),
        'max_cosine': float(cos_pairs.max()),
        'median_cosine': float(np.median(cos_pairs)),
        'percentiles': {str(p): float(np.percentile(cos_pairs, p)) for p in [1, 5, 25, 75, 95, 99]},
        'histogram': {'counts': cos_hist.tolist(), 'edges': cos_edges.tolist()},
    },
    'categories': {
        cat: {'count': len(indices), 'mean_norm': float(norms[indices].mean()), 'std_norm': float(norms[indices].std())}
        for cat, indices in categories.items()
    },
    'inter_category_cosines': inter_cat_cosines,
    'token_length': {
        'min': int(lengths.min()), 'max': int(lengths.max()), 'mean': float(lengths.mean()),
        'pearson_r': float(r_len), 'pearson_p': float(p_len),
        'per_length': per_length,
        'single_char_tokens': single_char_tokens,
    },
    'frequency_proxy': {
        'pearson_r': float(r_idx), 'pearson_p': float(p_idx),
        'binned_means': binned_means,
    },
}

with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save data for visualization
np.save(os.path.join(OUT, 'norms.npy'), norms)
np.save(os.path.join(OUT, 'explained_ratio.npy'), explained_ratio)

print(f"\nResults saved to {OUT}/")
print("Run charts.py for finding visualizations, visualize.py for the UMAP interactive plot.")
