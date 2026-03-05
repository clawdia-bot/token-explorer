"""
Token Embedding Space Geometry Explorer
GPT-2 (50,257 tokens × 768 dimensions)

Clawdia's night expedition — 2026-02-25
"""

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import json
import os

OUT = "/Users/clawdia/.openclaw/workspace/projects/token-explorer"

print("Loading GPT-2 embeddings...")
path = hf_hub_download('gpt2', 'pytorch_model.bin')
sd = torch.load(path, map_location='cpu', weights_only=False)
emb = sd['wte.weight'].numpy()  # [50257, 768]
tok = AutoTokenizer.from_pretrained('gpt2')

print(f"Embedding matrix shape: {emb.shape}")
print(f"dtype: {emb.dtype}")

# Decode all tokens for reference
tokens = [tok.decode([i]) for i in range(emb.shape[0])]

# ============================================================
# 1. NORMS
# ============================================================
print("\n" + "="*60)
print("1. L2 NORMS")
print("="*60)

norms = np.linalg.norm(emb, axis=1)
print(f"Mean norm: {norms.mean():.4f}")
print(f"Std:       {norms.std():.4f}")
print(f"Min:       {norms.min():.4f} (token {norms.argmin()}: {repr(tokens[norms.argmin()])})")
print(f"Max:       {norms.max():.4f} (token {norms.argmax()}: {repr(tokens[norms.argmax()])})")
print(f"Median:    {np.median(norms):.4f}")

# Percentiles
for p in [1, 5, 10, 25, 75, 90, 95, 99]:
    print(f"  P{p:02d}: {np.percentile(norms, p):.4f}")

# Top/bottom 20 by norm
print("\nSmallest norms (20):")
smallest = np.argsort(norms)[:20]
for i in smallest:
    print(f"  [{i:5d}] norm={norms[i]:.4f}  {repr(tokens[i])}")

print("\nLargest norms (20):")
largest = np.argsort(norms)[-20:][::-1]
for i in largest:
    print(f"  [{i:5d}] norm={norms[i]:.4f}  {repr(tokens[i])}")

# Norm histogram data
hist_counts, hist_edges = np.histogram(norms, bins=50)
print("\nNorm histogram (50 bins):")
for c, e in zip(hist_counts, hist_edges):
    print(f"  {e:.3f}: {'#' * (c // 20)} ({c})")

# ============================================================
# 2. THE ORIGIN — what's near it?
# ============================================================
print("\n" + "="*60)
print("2. THE ORIGIN")
print("="*60)

# Tokens closest to the origin (smallest norms, already computed)
print("Closest to origin = smallest norms (see above)")

# But also: what direction does the mean point?
mean_emb = emb.mean(axis=0)
mean_norm = np.linalg.norm(mean_emb)
print(f"\nMean embedding norm: {mean_norm:.4f}")
print(f"  (compare to individual token mean norm: {norms.mean():.4f})")

# What tokens are closest to the mean?
dists_to_mean = np.linalg.norm(emb - mean_emb, axis=1)
closest_to_mean = np.argsort(dists_to_mean)[:20]
print("\nClosest to MEAN embedding (20):")
for i in closest_to_mean:
    print(f"  [{i:5d}] dist={dists_to_mean[i]:.4f} norm={norms[i]:.4f}  {repr(tokens[i])}")

# What's the centroid's cosine similarity to everything?
cos_to_mean = (emb @ mean_emb) / (norms * mean_norm + 1e-10)
print(f"\nCosine similarity to mean:")
print(f"  Mean: {cos_to_mean.mean():.4f}")
print(f"  Std:  {cos_to_mean.std():.4f}")
print(f"  Min:  {cos_to_mean.min():.4f} (token {cos_to_mean.argmin()}: {repr(tokens[cos_to_mean.argmin()])})")
print(f"  Max:  {cos_to_mean.max():.4f} (token {cos_to_mean.argmax()}: {repr(tokens[cos_to_mean.argmax()])})")

# ============================================================
# 3. EFFECTIVE DIMENSIONALITY (PCA)
# ============================================================
print("\n" + "="*60)
print("3. EFFECTIVE DIMENSIONALITY")
print("="*60)

# Center the data
emb_centered = emb - mean_emb
# Covariance matrix via SVD (more numerically stable)
U, S, Vt = np.linalg.svd(emb_centered, full_matrices=False)
explained_var = S**2 / (emb.shape[0] - 1)
total_var = explained_var.sum()
explained_ratio = explained_var / total_var
cumulative = np.cumsum(explained_ratio)

print(f"Total variance: {total_var:.2f}")
print(f"Top singular values: {S[:10]}")
print(f"\nVariance explained by top components:")
for k in [1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 768]:
    if k <= len(cumulative):
        print(f"  Top {k:3d}: {cumulative[k-1]*100:.2f}%")

# Effective dimensionality (participation ratio)
pr = (explained_var.sum())**2 / (explained_var**2).sum()
print(f"\nParticipation ratio (effective dims): {pr:.1f} / {emb.shape[1]}")

# Entropy-based effective dimensionality
p = explained_var / explained_var.sum()
entropy = -np.sum(p * np.log(p + 1e-30))
eff_dim_entropy = np.exp(entropy)
print(f"Entropy-based effective dims: {eff_dim_entropy:.1f}")

# Where does 90%, 95%, 99% variance land?
for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
    k = np.searchsorted(cumulative, threshold) + 1
    print(f"  {threshold*100:.0f}% variance at component {k}")

# ============================================================
# 4. ANISOTROPY
# ============================================================
print("\n" + "="*60)
print("4. ANISOTROPY")
print("="*60)

# Random sample for pairwise cosine similarities (full matrix too expensive)
rng = np.random.RandomState(42)
n_sample = 5000
idx = rng.choice(emb.shape[0], n_sample, replace=False)
sample = emb[idx]
sample_norms = norms[idx]

# Normalize
sample_normed = sample / (sample_norms[:, None] + 1e-10)

# Pairwise cosine similarities
cos_matrix = sample_normed @ sample_normed.T
# Extract upper triangle (exclude diagonal)
triu_idx = np.triu_indices(n_sample, k=1)
cos_pairs = cos_matrix[triu_idx]

print(f"Pairwise cosine similarity ({n_sample} random tokens):")
print(f"  Mean: {cos_pairs.mean():.4f}")
print(f"  Std:  {cos_pairs.std():.4f}")
print(f"  Min:  {cos_pairs.min():.4f}")
print(f"  Max:  {cos_pairs.max():.4f}")
print(f"  Median: {np.median(cos_pairs):.4f}")

# If mean cosine >> 0, embeddings are anisotropic (narrow cone)
# If mean cosine ≈ 0, embeddings are isotropic (well spread)

for p in [1, 5, 25, 75, 95, 99]:
    print(f"  P{p:02d}: {np.percentile(cos_pairs, p):.4f}")

# Cosine histogram
cos_hist, cos_edges = np.histogram(cos_pairs, bins=50)
print("\nCosine similarity histogram:")
for c, e in zip(cos_hist, cos_edges):
    bar = '#' * max(1, c // 5000)
    print(f"  {e:+.3f}: {bar} ({c})")

# ============================================================
# 5. TOKEN CATEGORIES & CLUSTERS
# ============================================================
print("\n" + "="*60)
print("5. TOKEN CATEGORIES")
print("="*60)

# Categorize tokens
categories = {}
for i, t in enumerate(tokens):
    if t.strip() == '':
        cat = 'whitespace'
    elif t.strip().isdigit():
        cat = 'number'
    elif t.strip().isalpha() and len(t.strip()) == 1:
        cat = 'single_letter'
    elif t.startswith('Ġ') and t[1:].isalpha():  # GPT-2 uses Ġ for space-prefixed
        cat = 'word_with_space'
    elif t.startswith('Ġ'):
        cat = 'space_prefixed_other'
    elif t.isalpha() and t == t.lower():
        cat = 'lowercase_fragment'
    elif t.isalpha() and t[0].isupper():
        cat = 'capitalized'
    elif all(c in '.,;:!?-—()[]{}"\'/\\' for c in t.strip()):
        cat = 'punctuation'
    else:
        cat = 'other'
    
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(i)

print("Token categories:")
for cat, indices in sorted(categories.items(), key=lambda x: -len(x[1])):
    cat_norms = norms[indices]
    print(f"  {cat}: {len(indices)} tokens, mean norm={cat_norms.mean():.3f}, std={cat_norms.std():.3f}")

# Mean embedding per category
print("\nInter-category cosine similarities:")
cat_means = {}
for cat, indices in categories.items():
    cat_means[cat] = emb[indices].mean(axis=0)
    
cat_names = sorted(cat_means.keys())
for i, c1 in enumerate(cat_names):
    for c2 in cat_names[i+1:]:
        v1, v2 = cat_means[c1], cat_means[c2]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        if abs(cos) > 0.5:  # only show notable ones
            print(f"  {c1} × {c2}: {cos:.3f}")

# ============================================================
# 6. TOKEN LENGTH ANALYSIS
# ============================================================
print("\n" + "="*60)
print("6. TOKEN LENGTH")
print("="*60)

lengths = np.array([len(t) for t in tokens])
print(f"Token length stats: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.2f}")

for length in range(1, 15):
    mask = lengths == length
    if mask.sum() > 0:
        print(f"  Length {length:2d}: {mask.sum():5d} tokens, mean norm={norms[mask].mean():.3f}")

# Correlation between length and norm
from scipy import stats
r, p_val = stats.pearsonr(lengths, norms)
print(f"\nPearson correlation (length vs norm): r={r:.4f}, p={p_val:.2e}")

# Single-char tokens: what are they?
print("\nAll single-character tokens:")
for i, t in enumerate(tokens):
    if len(t) == 1:
        print(f"  [{i:5d}] norm={norms[i]:.4f} {repr(t)} (ord={ord(t)})")

# ============================================================
# 7. FREQUENCY vs NORM (using token index as rough proxy)
# ============================================================
print("\n" + "="*60)
print("7. FREQUENCY PROXY")
print("="*60)
# GPT-2 tokenizer: lower indices ≈ more frequent (BPE merge order)
# Let's check the correlation
r_idx, p_idx = stats.pearsonr(np.arange(len(norms)), norms)
print(f"Correlation (token index vs norm): r={r_idx:.4f}, p={p_idx:.2e}")

# Bin by index ranges
for start in range(0, 50000, 10000):
    end = min(start + 10000, len(norms))
    print(f"  Tokens {start:5d}-{end:5d}: mean norm={norms[start:end].mean():.3f}, std={norms[start:end].std():.3f}")

# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================
results = {
    'norms': {
        'mean': float(norms.mean()),
        'std': float(norms.std()),
        'min': float(norms.min()),
        'max': float(norms.max()),
        'min_token': repr(tokens[norms.argmin()]),
        'max_token': repr(tokens[norms.argmax()]),
    },
    'pca': {
        'participation_ratio': float(pr),
        'entropy_effective_dims': float(eff_dim_entropy),
        'top10_explained': float(cumulative[9]),
        'top50_explained': float(cumulative[49]),
        'top100_explained': float(cumulative[99]),
    },
    'anisotropy': {
        'mean_cosine': float(cos_pairs.mean()),
        'std_cosine': float(cos_pairs.std()),
    },
    'shape': list(emb.shape),
}

with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results.json")

# Save data for visualization
np.save(os.path.join(OUT, 'norms.npy'), norms)
np.save(os.path.join(OUT, 'explained_ratio.npy'), explained_ratio)
np.save(os.path.join(OUT, 'singular_values.npy'), S)

print("Numpy arrays saved.")
print("\n✅ Phase 1 complete. Run visualize.py next.")
