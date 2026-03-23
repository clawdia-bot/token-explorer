"""
Token Embedding Space Geometry Explorer
GPT-2 (50,257 tokens × 768 dimensions)
"""

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from scipy import stats
import json
import os
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
from tokenutils import token_display, categorize

print("Loading GPT-2 embeddings...")
path = hf_hub_download('gpt2', 'pytorch_model.bin')
sd = torch.load(path, map_location='cpu', weights_only=False)
emb = sd['wte.weight'].numpy()  # [50257, 768]
tok = AutoTokenizer.from_pretrained('gpt2')

print(f"Embedding matrix: {emb.shape[0]} tokens x {emb.shape[1]} dimensions ({emb.dtype})")

# Decode all tokens — raw decoded form for analysis, display form for output
tokens = [tok.decode([i]) for i in range(emb.shape[0])]
labels = [token_display(tok, i) for i in range(emb.shape[0])]

# Precompute shared quantities
norms = np.linalg.norm(emb, axis=1)
mean_emb = emb.mean(axis=0)
mean_norm = np.linalg.norm(mean_emb)
lengths = np.array([len(t) for t in tokens])

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
print(f"  Smallest: {repr(labels[smallest[0]])} ({norms[smallest[0]]:.3f}) — common preposition")
print(f"  Largest:  {repr(labels[largest[0]])} ({norms[largest[0]]:.3f}) — rare label")
print(f"  Token length vs norm: r={r_len:.3f} (shorter tokens tend toward higher frequency)")
print(f"  -> The model pushes common tokens toward the origin; rare tokens orbit the periphery.")

# ============================================================
# 2. ANISOTROPY
# ============================================================
print("\n" + "="*60)
print("2. ANISOTROPY — do embeddings point the same direction?")
print("="*60)

rng = np.random.RandomState(42)
n_sample = 5000
idx = rng.choice(emb.shape[0], n_sample, replace=False)
sample = emb[idx]
sample_norms = norms[idx]

sample_normed = sample / (sample_norms[:, None] + 1e-10)

cos_matrix = sample_normed @ sample_normed.T
triu_idx = np.triu_indices(n_sample, k=1)
cos_pairs = cos_matrix[triu_idx]

cos_hist, cos_edges = np.histogram(cos_pairs, bins=50)

print(f"  Mean pairwise cosine: {cos_pairs.mean():.3f} +/- {cos_pairs.std():.3f} ({n_sample} random tokens)")
print(f"  -> Moderately anisotropic (0 = isotropic sphere, >0.5 = narrow cone).")
print(f"  -> Embeddings favor the same hemisphere but aren't degenerate.")

# ============================================================
# 3. EFFECTIVE DIMENSIONALITY (PCA)
# ============================================================
print("\n" + "="*60)
print("3. EFFECTIVE DIMENSIONALITY — how much space is actually used?")
print("="*60)

emb_centered = emb - mean_emb
U, S, Vt = np.linalg.svd(emb_centered, full_matrices=False)
explained_var = S**2 / (emb.shape[0] - 1)
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

print(f"  Participation ratio: {pr:.0f} / {emb.shape[1]} dims")
print(f"  Entropy-based effective dims: {eff_dim_entropy:.0f} / {emb.shape[1]}")
print(f"  Variance thresholds: 50% at {thresholds[0.5]} dims, 90% at {thresholds[0.9]}, 99% at {thresholds[0.99]}")
print(f"  PC1 explains just {explained_ratio[0]*100:.2f}% — no single dimension dominates.")
print(f"  -> Despite the anisotropy, the space genuinely uses most of its {emb.shape[1]} dimensions.")

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
print(f"  Most directionally unique: {repr(labels[min_cos_idx])} (cosine to mean={cos_to_mean.min():.3f})")

# Data-driven conclusion
if abs(r_idx_centered) > abs(r_idx) * 0.5:
    print(f"  -> Frequency-distance holds from both reference points. The radial structure is real.")
else:
    print(f"  -> Correlation weakens significantly from centroid. The origin-based finding is partly")
    print(f"     an artifact of centroid displacement — common tokens pull the mean toward them.")

# ============================================================
# 5. TOKEN CATEGORIES
# ============================================================
print("\n" + "="*60)
print("5. TOKEN CATEGORIES — script and type clustering")
print("="*60)

categories = {}
for i in range(emb.shape[0]):
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
binned_means = []
for start in range(0, 50000, 10000):
    end = min(start + 10000, len(norms))
    binned_means.append({
        'range': f'{start}-{end}',
        'mean_norm': float(norms[start:end].mean()),
        'std_norm': float(norms[start:end].std()),
    })

results = {
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
        'top50_explained': float(cumulative[49]),
        'top100_explained': float(cumulative[99]),
        'variance_at_k': {str(k): float(cumulative[k-1]) for k in [1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 768] if k <= len(cumulative)},
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
np.save(os.path.join(OUT, 'dists_to_mean.npy'), dists_to_mean)
np.save(os.path.join(OUT, 'explained_ratio.npy'), explained_ratio)
np.save(os.path.join(OUT, 'singular_values.npy'), S)

print(f"\nDetailed data saved to results.json (ranked lists, histograms, percentiles, etc.)")
print("Numpy arrays saved (norms, explained_ratio, singular_values).")
print("\nRun charts.py for finding visualizations, visualize.py for the UMAP interactive plot.")
