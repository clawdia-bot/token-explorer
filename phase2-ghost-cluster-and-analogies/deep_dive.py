"""
Deep dive into interesting findings from Phase 1.

1. The "ghost cluster" — control characters that all have norm ≈ 3.09
2. Newline as outlier (norm 2.69, lowest of any single char)
3. Norm-frequency relationship — is it linear or something else?
4. The SPONSORED token — why is it the max norm outlier?
5. Nearest neighbors for interesting tokens
6. "Empty space" — where are there NO tokens?
"""
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from scipy import stats

import os
OUT = os.path.dirname(os.path.abspath(__file__))

print("Loading...")
path = hf_hub_download('gpt2', 'pytorch_model.bin')
sd = torch.load(path, map_location='cpu', weights_only=False)
emb = sd['wte.weight'].numpy()
tok = AutoTokenizer.from_pretrained('gpt2')
tokens = [tok.decode([i]) for i in range(emb.shape[0])]
norms = np.linalg.norm(emb, axis=1)

# ============================================================
# 1. THE GHOST CLUSTER — control chars with norm ≈ 3.09
# ============================================================
print("\n" + "="*60)
print("1. THE GHOST CLUSTER")
print("="*60)

# Tokens 188-221 (control chars + low bytes) all have norm ~3.089
ghost_idx = list(range(188, 222))
ghost_embs = emb[ghost_idx]
ghost_norms = norms[ghost_idx]

print(f"Ghost cluster: tokens 188-221 ({len(ghost_idx)} tokens)")
print(f"  Norm range: {ghost_norms.min():.4f} — {ghost_norms.max():.4f}")
print(f"  Norm std: {ghost_norms.std():.6f}")

# How similar are they to each other?
ghost_normed = ghost_embs / ghost_norms[:, None]
cos_matrix = ghost_normed @ ghost_normed.T
triu = np.triu_indices(len(ghost_idx), k=1)
cos_pairs = cos_matrix[triu]
print(f"  Pairwise cosine within cluster:")
print(f"    Mean: {cos_pairs.mean():.4f}")
print(f"    Min:  {cos_pairs.min():.4f}")
print(f"    Max:  {cos_pairs.max():.4f}")
print(f"    Std:  {cos_pairs.std():.6f}")

# The newline (token 198) — is it different from its siblings?
newline_idx = 198
print(f"\n  Newline (198) norm: {norms[newline_idx]:.4f}")
print(f"  Mean cosine of newline to other ghosts: {cos_matrix[newline_idx - 188, :].mean():.4f}")

# Is this cluster genuinely near-identical, or just similar?
ghost_mean = ghost_embs.mean(axis=0)
ghost_dists = np.linalg.norm(ghost_embs - ghost_mean, axis=1)
print(f"  Dist from cluster centroid: mean={ghost_dists.mean():.4f}, max={ghost_dists.max():.4f}")

# Exclude newline and re-check
ghost_no_nl = [i for i in ghost_idx if i != 198]
ghost_no_nl_embs = emb[ghost_no_nl]
ghost_no_nl_normed = ghost_no_nl_embs / norms[ghost_no_nl][:, None]
cos2 = ghost_no_nl_normed @ ghost_no_nl_normed.T
triu2 = np.triu_indices(len(ghost_no_nl), k=1)
print(f"  WITHOUT newline — pairwise cosine mean: {cos2[triu2].mean():.4f}, min: {cos2[triu2].min():.4f}")

# ============================================================
# 2. NEWLINE — the special character
# ============================================================
print("\n" + "="*60)
print("2. NEWLINE ANALYSIS")
print("="*60)

# What are newline's nearest neighbors?
nl_emb = emb[198]
cos_to_nl = (emb @ nl_emb) / (norms * norms[198] + 1e-10)
nn_nl = np.argsort(cos_to_nl)[-20:][::-1]
print("Newline's nearest neighbors (cosine):")
for i in nn_nl:
    print(f"  [{i:5d}] cos={cos_to_nl[i]:.4f} norm={norms[i]:.4f}  {repr(tokens[i])}")

# Compare newline to space
space_idx = 220
print(f"\nNewline (198) vs Space (220):")
print(f"  Norms: {norms[198]:.4f} vs {norms[220]:.4f}")
cos_nl_sp = np.dot(emb[198], emb[220]) / (norms[198] * norms[220])
print(f"  Cosine: {cos_nl_sp:.4f}")

# ============================================================
# 3. NEAREST NEIGHBORS for interesting tokens
# ============================================================
print("\n" + "="*60)
print("3. NEAREST NEIGHBORS (INTERESTING TOKENS)")
print("="*60)

def find_nn(token_str, k=15):
    """Find nearest neighbors by cosine similarity"""
    # Find the token ID
    matches = [i for i, t in enumerate(tokens) if t == token_str]
    if not matches:
        # Try with space prefix
        matches = [i for i, t in enumerate(tokens) if t.strip() == token_str]
    if not matches:
        print(f"  Token {repr(token_str)} not found")
        return
    idx = matches[0]
    vec = emb[idx]
    cos = (emb @ vec) / (norms * norms[idx] + 1e-10)
    nn = np.argsort(cos)[-k-1:-1][::-1]  # exclude self
    print(f"\nNearest to [{idx}] {repr(tokens[idx])} (norm={norms[idx]:.3f}):")
    for i in nn:
        print(f"  [{i:5d}] cos={cos[i]:.4f} norm={norms[i]:.4f}  {repr(tokens[i])}")

# The outliers
find_nn('SPONSORED')
find_nn(' the')
find_nn(' at')  # smallest norm

# Semantic probes
find_nn(' king')
find_nn(' queen')
find_nn(' dog')
find_nn(' Python')

# ============================================================
# 4. NORM vs FREQUENCY — deeper look
# ============================================================
print("\n" + "="*60)
print("4. NORM-FREQUENCY RELATIONSHIP")
print("="*60)

# Log-index as better frequency proxy (BPE merge frequency is roughly Zipfian)
log_idx = np.log1p(np.arange(len(norms)))
r_log, p_log = stats.pearsonr(log_idx, norms)
print(f"Correlation (log(index) vs norm): r={r_log:.4f}")

# Spearman (rank correlation, no linearity assumption)
rho, p_rho = stats.spearmanr(np.arange(len(norms)), norms)
print(f"Spearman (index vs norm): rho={rho:.4f}")

# The first 256 tokens are byte-level — they break the pattern
r_post256, _ = stats.pearsonr(np.arange(256, len(norms)), norms[256:])
print(f"Correlation (index vs norm, tokens 256+): r={r_post256:.4f}")

# ============================================================
# 5. ANALOGIES — does the space support them?
# ============================================================
print("\n" + "="*60)
print("5. EMBEDDING ANALOGIES")
print("="*60)

def analogy(a, b, c, k=10):
    """a is to b as c is to ?"""
    def get_idx(s):
        matches = [i for i, t in enumerate(tokens) if t == s]
        if not matches:
            matches = [i for i, t in enumerate(tokens) if t.strip() == s.strip()]
        return matches[0] if matches else None
    
    ia, ib, ic = get_idx(a), get_idx(b), get_idx(c)
    if None in (ia, ib, ic):
        print(f"  Missing token: a={ia}, b={ib}, c={ic}")
        return
    
    # Classic: b - a + c
    vec = emb[ib] - emb[ia] + emb[ic]
    vec_norm = np.linalg.norm(vec)
    cos = (emb @ vec) / (norms * vec_norm + 1e-10)
    
    # Exclude inputs
    cos[ia] = -2
    cos[ib] = -2
    cos[ic] = -2
    
    nn = np.argsort(cos)[-k:][::-1]
    print(f"\n{repr(a)} : {repr(b)} :: {repr(c)} : ?")
    for i in nn:
        print(f"  [{i:5d}] cos={cos[i]:.4f}  {repr(tokens[i])}")

analogy(' king', ' queen', ' man')
analogy(' dog', ' dogs', ' cat')
analogy(' France', ' Paris', ' Japan')
analogy(' big', ' bigger', ' small')

# ============================================================
# 6. ISOTROPY TEST — random directions
# ============================================================
print("\n" + "="*60)
print("6. ISOTROPY — RANDOM DIRECTION TEST")
print("="*60)

rng = np.random.RandomState(42)
# Generate random unit vectors and see how many tokens are close
random_dirs = rng.randn(100, 768)
random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)

normed_emb = emb / (norms[:, None] + 1e-10)

max_cos_per_dir = []
for d in random_dirs:
    cos = normed_emb @ d
    max_cos_per_dir.append(cos.max())

max_cos_per_dir = np.array(max_cos_per_dir)
print(f"Max cosine to 100 random unit vectors:")
print(f"  Mean: {max_cos_per_dir.mean():.4f}")
print(f"  Min:  {max_cos_per_dir.min():.4f}")
print(f"  Max:  {max_cos_per_dir.max():.4f}")
print(f"  (For reference: in a uniform 768-sphere, expected max ≈ 0.08)")
print(f"  → Embeddings are {'spread widely' if max_cos_per_dir.mean() > 0.3 else 'clustered in a subspace'}")

# ============================================================
# 7. THE WEIRD TOKENS — product IDs, glitch tokens, etc.
# ============================================================
print("\n" + "="*60)
print("7. WEIRD TOKEN ANALYSIS")
print("="*60)

# Find tokens that are clearly product/API artifacts
weird_patterns = ['externalToEVA', 'quickShip', 'TheNitrome', 'BuyableInstoreAndOnline',
                  'soDeliveryDate', 'inventoryQuantity', 'isSpecialOrderable',
                  'DragonMagazine', 'natureconservancy', 'SPONSORED']

print("Known 'weird' tokens:")
for pat in weird_patterns:
    matches = [i for i, t in enumerate(tokens) if pat in t]
    for i in matches:
        print(f"  [{i:5d}] norm={norms[i]:.4f}  {repr(tokens[i])}")

# How similar are the weird tokens to each other?
weird_idx = []
for pat in weird_patterns:
    for i, t in enumerate(tokens):
        if pat in t:
            weird_idx.append(i)
            break

if len(weird_idx) > 1:
    weird_embs = emb[weird_idx]
    weird_norms = norms[weird_idx]
    weird_normed = weird_embs / weird_norms[:, None]
    cos_w = weird_normed @ weird_normed.T
    triu_w = np.triu_indices(len(weird_idx), k=1)
    print(f"\n  Pairwise cosine among weird tokens: mean={cos_w[triu_w].mean():.4f}")
    print(f"  Compare to global mean pairwise cosine: 0.2690")

# ============================================================
# 8. DIMENSIONAL CONTRIBUTION — which dims matter most?
# ============================================================
print("\n" + "="*60)
print("8. PER-DIMENSION ANALYSIS")
print("="*60)

dim_variance = emb.var(axis=0)
dim_mean = emb.mean(axis=0)
dim_range = emb.max(axis=0) - emb.min(axis=0)

print(f"Per-dimension variance:")
print(f"  Mean: {dim_variance.mean():.4f}")
print(f"  Max:  {dim_variance.max():.4f} (dim {dim_variance.argmax()})")
print(f"  Min:  {dim_variance.min():.4f} (dim {dim_variance.argmin()})")
print(f"  Std:  {dim_variance.std():.4f}")

print(f"\nPer-dimension mean (should be ~0 if centered):")
print(f"  Mean of means: {dim_mean.mean():.4f}")
print(f"  Max mean: {dim_mean.max():.4f} (dim {dim_mean.argmax()})")
print(f"  Min mean: {dim_mean.min():.4f} (dim {dim_mean.argmin()})")

# The dimension with highest mean is the "default direction" — what does it encode?
top_dim = dim_mean.argmax()
vals_top = emb[:, top_dim]
top_tokens = np.argsort(vals_top)[-10:][::-1]
bot_tokens = np.argsort(vals_top)[:10]
print(f"\nDim {top_dim} (highest mean = {dim_mean[top_dim]:.4f}):")
print(f"  Highest values:")
for i in top_tokens:
    print(f"    [{i:5d}] val={vals_top[i]:.4f}  {repr(tokens[i])}")
print(f"  Lowest values:")
for i in bot_tokens:
    print(f"    [{i:5d}] val={vals_top[i]:.4f}  {repr(tokens[i])}")

print("\n✅ Deep dive complete.")
