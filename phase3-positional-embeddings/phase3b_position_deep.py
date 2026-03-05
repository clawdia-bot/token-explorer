"""
Phase 3b: Deep dive into positional embeddings.

Key findings from 3a to investigate:
- Position 0 has norm 0.12 (!!) vs mean 3.39 — why?
- Consecutive positions have cosine 0.997 — extremely smooth
- Only 3 PCs needed for 90% variance (vs 558 for tokens!)
- Token and position embeddings use ZERO overlapping top-variance dims
- PC2 has period-512 as top FFT frequency — sinusoidal?
"""
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from scipy import stats

print("Loading...")
path = hf_hub_download('gpt2', 'pytorch_model.bin')
sd = torch.load(path, map_location='cpu', weights_only=False)
pos = sd['wpe.weight'].numpy()
emb = sd['wte.weight'].numpy()
tok = AutoTokenizer.from_pretrained('gpt2')
tokens = [tok.decode([i]) for i in range(emb.shape[0])]
norms_pos = np.linalg.norm(pos, axis=1)

# ============================================================
# 1. POSITION 0 ANOMALY
# ============================================================
print("\n" + "="*60)
print("1. POSITION 0 — THE ANOMALY")
print("="*60)

print(f"Position 0 norm: {norms_pos[0]:.6f}")
print(f"Position 1 norm: {norms_pos[1]:.6f}")
print(f"Position 2 norm: {norms_pos[2]:.6f}")
print(f"Position 10 norm: {norms_pos[10]:.6f}")
print(f"Position 100 norm: {norms_pos[100]:.6f}")

# Position 0 is nearly the zero vector — what does that mean?
# It means at position 0, the model adds almost nothing to the token embedding.
# The token IS its identity at position 0. Position only starts mattering later.
cos_0_1 = np.dot(pos[0], pos[1]) / (norms_pos[0] * norms_pos[1] + 1e-10)
cos_1_2 = np.dot(pos[1], pos[2]) / (norms_pos[1] * norms_pos[2] + 1e-10)
print(f"\nCosine pos0-pos1: {cos_0_1:.4f}")
print(f"Cosine pos1-pos2: {cos_1_2:.4f}")

# Norm trajectory — is it monotonic?
print(f"\nNorm trajectory (first 20 positions):")
for i in range(20):
    bar = "█" * int(norms_pos[i] * 5)
    print(f"  pos {i:3d}: {norms_pos[i]:.4f} {bar}")

print(f"\nNorm trajectory (around 512):")
for i in [500, 505, 510, 511, 512, 513, 515, 520]:
    print(f"  pos {i:3d}: {norms_pos[i]:.4f}")

print(f"\nNorm trajectory (last 10):")
for i in range(1014, 1024):
    bar = "█" * int(norms_pos[i] * 5)
    print(f"  pos {i:3d}: {norms_pos[i]:.4f} {bar}")

# ============================================================
# 2. PERIODICITY — Are positional embeddings sinusoidal?
# ============================================================
print("\n" + "="*60)
print("2. PERIODICITY ANALYSIS")
print("="*60)

# Look at individual dimensions
from numpy.fft import fft

# Pick a few high-variance position dimensions
var_per_dim = pos.var(axis=0)
top_dims = np.argsort(var_per_dim)[-5:][::-1]

for d in top_dims:
    signal = pos[:, d]
    spectrum = np.abs(fft(signal - signal.mean()))[:512]
    top_freqs = np.argsort(spectrum)[-3:][::-1]
    periods = [f"{1024/f:.0f}" if f > 0 else "DC" for f in top_freqs]
    print(f"  Dim {d:3d} (var={var_per_dim[d]:.4f}): top periods = {periods}")

# Check a specific low-variance dim too
low_dims = np.argsort(var_per_dim)[:3]
for d in low_dims:
    signal = pos[:, d]
    spectrum = np.abs(fft(signal - signal.mean()))[:512]
    top_freqs = np.argsort(spectrum)[-3:][::-1]
    periods = [f"{1024/f:.0f}" if f > 0 else "DC" for f in top_freqs]
    print(f"  Dim {d:3d} (var={var_per_dim[d]:.6f}): top periods = {periods}")

# ============================================================
# 3. SEMANTIC POSITION REGIONS
# ============================================================
print("\n" + "="*60)
print("3. DO NEARBY POSITIONS CLUSTER?")
print("="*60)

# Cosine heatmap statistics — do positions group into blocks?
# Sample every 50th position
sample_pos = list(range(0, 1024, 50))
sample_embs = pos[sample_pos]
sample_norms = norms_pos[sample_pos]
normed = sample_embs / (sample_norms[:, None] + 1e-10)
cos_mat = normed @ normed.T

print("Cosine between positions (every 50th):")
print("     ", "  ".join([f"{p:4d}" for p in sample_pos[:10]]))
for i in range(min(10, len(sample_pos))):
    vals = " ".join([f"{cos_mat[i,j]:5.2f}" for j in range(min(10, len(sample_pos)))])
    print(f"  {sample_pos[i]:4d} {vals}")

# ============================================================
# 4. ORTHOGONALITY — Token and Position Subspaces
# ============================================================
print("\n" + "="*60)
print("4. TOKEN-POSITION ORTHOGONALITY")
print("="*60)

# SVD of each
from numpy.linalg import svd
U_tok, S_tok, Vt_tok = svd(emb - emb.mean(0), full_matrices=False)
U_pos, S_pos, Vt_pos = svd(pos - pos.mean(0), full_matrices=False)

# How aligned are the principal subspaces?
# Take top-k PCs from each and compute alignment
for k in [5, 10, 50]:
    # Subspace alignment: ||V_tok[:k]^T @ V_pos[:k]||_F / sqrt(k)
    alignment = np.linalg.norm(Vt_tok[:k] @ Vt_pos[:k].T, 'fro') / np.sqrt(k)
    # Random baseline: sqrt(k/768)
    random_baseline = np.sqrt(k / 768)
    print(f"  Top-{k:2d} PC alignment: {alignment:.4f} (random baseline: {random_baseline:.4f})")

# ============================================================
# 5. COMBINED EMBEDDING AT POSITION 0 vs REST
# ============================================================
print("\n" + "="*60)
print("5. HOW POSITION CHANGES TOKEN NEIGHBORHOODS")
print("="*60)

# Since pos[0] ≈ 0, embeddings at pos 0 ARE the token embeddings
# As position increases, how much do neighborhoods change?
norms_tok = np.linalg.norm(emb, axis=1)

def get_top_nn(query_idx, pos_idx, k=20):
    combined = emb + pos[pos_idx]
    vec = combined[query_idx]
    norms_c = np.linalg.norm(combined, axis=1)
    cos = (combined @ vec) / (norms_c * np.linalg.norm(vec) + 1e-10)
    cos[query_idx] = -2
    return set(np.argsort(cos)[-k:])

# Track neighborhood stability across positions for " king"
king_idx = tok.encode(" king")[0]
nn_at_0 = get_top_nn(king_idx, 0)
print(f"' king' neighborhood stability (Jaccard with pos 0, top 20):")
for p in [0, 1, 10, 50, 100, 200, 500, 1000]:
    nn = get_top_nn(king_idx, p)
    jaccard = len(nn & nn_at_0) / len(nn | nn_at_0)
    print(f"  pos {p:4d}: {jaccard:.3f} ({len(nn & nn_at_0)}/20 overlap)")

# ============================================================
# 6. POSITION NORM GROWTH PATTERN
# ============================================================
print("\n" + "="*60)
print("6. POSITION NORM GROWTH PATTERN")
print("="*60)

# Fit: is it linear, sqrt, or log?
positions = np.arange(1, 1024)  # skip pos 0
pos_norms = norms_pos[1:]

# Linear
r_lin, _ = stats.pearsonr(positions, pos_norms)
# Sqrt
r_sqrt, _ = stats.pearsonr(np.sqrt(positions), pos_norms)
# Log
r_log, _ = stats.pearsonr(np.log(positions), pos_norms)

print(f"Norm vs position correlation (excluding pos 0):")
print(f"  Linear: r={r_lin:.4f}")
print(f"  Sqrt:   r={r_sqrt:.4f}")
print(f"  Log:    r={r_log:.4f}")

# Last 100 positions — is there an edge effect?
print(f"\nLast 100 positions — norm stats:")
print(f"  Mean: {norms_pos[-100:].mean():.4f}")
print(f"  Std:  {norms_pos[-100:].std():.4f}")
print(f"  Max:  {norms_pos[-100:].max():.4f} at pos {np.argmax(norms_pos[-100:])+924}")
print(f"  Last position (1023) norm: {norms_pos[-1]:.4f}")

print("\n✅ Phase 3b complete.")
