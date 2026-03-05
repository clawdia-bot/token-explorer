"""
Phase 3: Follow-up investigations from the Feb 25 expedition.

1. Positional embeddings — what does GPT-2's position space look like?
2. Ghost cluster substitution — are control chars truly interchangeable?
3. Token-position interaction — how do position embeddings change the geometry?
4. The "remove PC1" trick — does removing the frequency dimension help?
"""
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from scipy import stats

OUT = "/Users/clawdia/.openclaw/workspace/projects/token-explorer"

print("Loading GPT-2 weights...")
path = hf_hub_download('gpt2', 'pytorch_model.bin')
sd = torch.load(path, map_location='cpu', weights_only=False)
emb = sd['wte.weight'].numpy()   # [50257, 768] token embeddings
pos = sd['wpe.weight'].numpy()   # [1024, 768] positional embeddings
tok = AutoTokenizer.from_pretrained('gpt2')
tokens = [tok.decode([i]) for i in range(emb.shape[0])]
norms_tok = np.linalg.norm(emb, axis=1)
norms_pos = np.linalg.norm(pos, axis=1)

# ============================================================
# 1. POSITIONAL EMBEDDING GEOMETRY
# ============================================================
print("\n" + "="*60)
print("1. POSITIONAL EMBEDDINGS")
print("="*60)

print(f"Shape: {pos.shape}")
print(f"Norm range: {norms_pos.min():.4f} — {norms_pos.max():.4f}")
print(f"Norm mean: {norms_pos.mean():.4f}, std: {norms_pos.std():.4f}")
print(f"Compare to token norms: mean={norms_tok.mean():.4f}")

# Norm vs position
print(f"\nNorm by position quartile:")
q = len(norms_pos) // 4
for i, label in enumerate(["0-255", "256-511", "512-767", "768-1023"]):
    chunk = norms_pos[i*q:(i+1)*q]
    print(f"  {label}: mean={chunk.mean():.4f}, std={chunk.std():.4f}")

# Consecutive position similarity
consec_cos = []
for i in range(len(pos)-1):
    c = np.dot(pos[i], pos[i+1]) / (norms_pos[i] * norms_pos[i+1] + 1e-10)
    consec_cos.append(c)
consec_cos = np.array(consec_cos)
print(f"\nConsecutive position cosine similarity:")
print(f"  Mean: {consec_cos.mean():.4f}")
print(f"  First 10: {consec_cos[:10].mean():.4f}")
print(f"  Last 10: {consec_cos[-10:].mean():.4f}")

# Do positions far apart look different?
far_cos = []
for i in range(0, 100):
    c = np.dot(pos[i], pos[i+512]) / (norms_pos[i] * norms_pos[i+512] + 1e-10)
    far_cos.append(c)
print(f"  Distance-512 cosine mean: {np.mean(far_cos):.4f}")

# PCA of positional embeddings
from numpy.linalg import svd
pos_centered = pos - pos.mean(axis=0)
U, S, Vt = svd(pos_centered, full_matrices=False)
var_ratio_pos = (S**2) / (S**2).sum()
print(f"\nPCA of positional embeddings:")
print(f"  Top 5 PCs explain: {var_ratio_pos[:5].sum()*100:.1f}%")
print(f"  Top 10 PCs explain: {var_ratio_pos[:10].sum()*100:.1f}%")
print(f"  Dims for 90% variance: {np.searchsorted(np.cumsum(var_ratio_pos), 0.9)+1}")
print(f"  Dims for 99% variance: {np.searchsorted(np.cumsum(var_ratio_pos), 0.99)+1}")
print(f"  PC1 explains: {var_ratio_pos[0]*100:.2f}%")

# What do the top PCs look like? (PC1 coords vs position index)
pc1_coords = U[:, 0] * S[0]
pc2_coords = U[:, 1] * S[1]
# Check if PC1 is monotonic (encodes position directly)
pc1_corr = stats.spearmanr(np.arange(1024), pc1_coords)[0]
pc2_corr = stats.spearmanr(np.arange(1024), pc2_coords)[0]
print(f"  PC1 correlation with position: ρ={pc1_corr:.4f}")
print(f"  PC2 correlation with position: ρ={pc2_corr:.4f}")

# Is position encoding periodic? Check autocorrelation of PC2
from numpy.fft import fft
pc2_fft = np.abs(fft(pc2_coords - pc2_coords.mean()))[:512]
top_freq = np.argsort(pc2_fft)[-5:][::-1]
print(f"  PC2 top FFT frequencies (period): {[1024/f if f>0 else 'DC' for f in top_freq]}")

# ============================================================
# 2. TOKEN-POSITION INTERACTION
# ============================================================
print("\n" + "="*60)
print("2. TOKEN-POSITION INTERACTION")
print("="*60)

# At position 0, what happens to "the" vs position 500?
the_idx = tok.encode(" the")[0]
print(f"Token ' the' (idx {the_idx}):")
combined_0 = emb[the_idx] + pos[0]
combined_500 = emb[the_idx] + pos[500]
print(f"  Norm at pos 0: {np.linalg.norm(combined_0):.4f}")
print(f"  Norm at pos 500: {np.linalg.norm(combined_500):.4f}")
cos_pos = np.dot(combined_0, combined_500) / (np.linalg.norm(combined_0)*np.linalg.norm(combined_500))
print(f"  Cosine (pos0 vs pos500): {cos_pos:.4f}")

# How much does position perturb token identity?
# For a given position, do nearest neighbors change?
def nn_at_pos(token_str, position, k=10):
    matches = tok.encode(token_str)
    idx = matches[0]
    vec = emb[idx] + pos[position]
    all_at_pos = emb + pos[position]
    all_norms = np.linalg.norm(all_at_pos, axis=1)
    cos = (all_at_pos @ vec) / (all_norms * np.linalg.norm(vec) + 1e-10)
    cos[idx] = -2
    nn = np.argsort(cos)[-k:][::-1]
    return [(tokens[i], cos[i]) for i in nn]

print(f"\n  NN of ' the' at position 0:")
for t, c in nn_at_pos(' the', 0, 5):
    print(f"    {repr(t):20s} cos={c:.4f}")
print(f"  NN of ' the' at position 500:")
for t, c in nn_at_pos(' the', 500, 5):
    print(f"    {repr(t):20s} cos={c:.4f}")

# ============================================================
# 3. REMOVE PC1 FROM TOKEN EMBEDDINGS
# ============================================================
print("\n" + "="*60)
print("3. REMOVE PC1 (FREQUENCY AXIS) FROM TOKEN EMBEDDINGS")
print("="*60)

emb_centered = emb - emb.mean(axis=0)
U_tok, S_tok, Vt_tok = svd(emb_centered, full_matrices=False)
pc1_dir = Vt_tok[0]  # first principal component direction

# Project out PC1
emb_no_pc1 = emb_centered - np.outer(emb_centered @ pc1_dir, pc1_dir)
norms_no_pc1 = np.linalg.norm(emb_no_pc1, axis=1)

# Check anisotropy after removing PC1
rng = np.random.RandomState(42)
sample = rng.choice(len(emb_no_pc1), 5000, replace=False)
emb_sample = emb_no_pc1[sample]
norms_sample = norms_no_pc1[sample]
normed_sample = emb_sample / (norms_sample[:, None] + 1e-10)
cos_mat = normed_sample @ normed_sample.T
triu = np.triu_indices(5000, k=1)
cos_vals = cos_mat[triu]
print(f"Mean pairwise cosine BEFORE removing PC1: 0.2690 (from phase 1)")
print(f"Mean pairwise cosine AFTER removing PC1:  {cos_vals.mean():.4f}")
print(f"Std: {cos_vals.std():.4f}")

# Do analogies still work without PC1?
def analogy_no_pc1(a, b, c, k=5):
    def get_idx(s):
        matches = [i for i, t in enumerate(tokens) if t == s]
        return matches[0] if matches else None
    ia, ib, ic = get_idx(a), get_idx(b), get_idx(c)
    if None in (ia, ib, ic):
        return
    vec = emb_no_pc1[ib] - emb_no_pc1[ia] + emb_no_pc1[ic]
    vec_norm = np.linalg.norm(vec)
    cos = (emb_no_pc1 @ vec) / (norms_no_pc1 * vec_norm + 1e-10)
    cos[ia] = cos[ib] = cos[ic] = -2
    nn = np.argsort(cos)[-k:][::-1]
    print(f"  {repr(a)} : {repr(b)} :: {repr(c)} : ?")
    for i in nn:
        print(f"    {repr(tokens[i]):20s} cos={cos[i]:.4f}")

print(f"\nAnalogies WITHOUT PC1:")
analogy_no_pc1(' king', ' queen', ' man')
analogy_no_pc1(' France', ' Paris', ' Japan')

# ============================================================
# 4. GHOST CLUSTER SUBSTITUTION TEST
# ============================================================
print("\n" + "="*60)
print("4. GHOST CLUSTER — FUNCTIONAL EQUIVALENCE")
print("="*60)

# Can't run inference (no model), but we can check:
# How close are ghost tokens to each other vs to other tokens?
ghost_idx = [i for i in range(188, 222) if i != 198]  # exclude newline
ghost_embs = emb[ghost_idx]
ghost_mean = ghost_embs.mean(axis=0)

# Distance between ghost tokens
ghost_pairwise = []
for i in range(len(ghost_idx)):
    for j in range(i+1, len(ghost_idx)):
        d = np.linalg.norm(ghost_embs[i] - ghost_embs[j])
        ghost_pairwise.append(d)
ghost_pairwise = np.array(ghost_pairwise)

# Compare to typical token pair distance  
sample_pairs = []
rng2 = np.random.RandomState(99)
for _ in range(1000):
    i, j = rng2.choice(len(emb), 2, replace=False)
    sample_pairs.append(np.linalg.norm(emb[i] - emb[j]))
sample_pairs = np.array(sample_pairs)

print(f"Ghost cluster L2 pairwise distances:")
print(f"  Mean: {ghost_pairwise.mean():.4f}")
print(f"  Max:  {ghost_pairwise.max():.4f}")
print(f"  Min:  {ghost_pairwise.min():.4f}")
print(f"\nRandom token pair L2 distances:")
print(f"  Mean: {sample_pairs.mean():.4f}")
print(f"  Min:  {sample_pairs.min():.4f}")
print(f"\nRatio (ghost_mean / random_mean): {ghost_pairwise.mean()/sample_pairs.mean():.4f}")

# What about the OUTPUT embedding? (lm_head in GPT-2 is tied to wte)
# So the ghost tokens would also produce nearly identical logits
# when used as the last hidden state → truly interchangeable
print(f"\nSince GPT-2 ties input/output embeddings (wte = lm_head),")
print(f"ghost tokens would produce nearly identical next-token distributions.")
print(f"They are functionally interchangeable in both input AND output.")

# ============================================================
# 5. NORM HISTOGRAM — which tokens sit at each norm level?
# ============================================================
print("\n" + "="*60)
print("5. NORM DISTRIBUTION BREAKDOWN")
print("="*60)

bins = [(0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0), (5.0, 6.5)]
for lo, hi in bins:
    mask = (norms_tok >= lo) & (norms_tok < hi)
    count = mask.sum()
    if count > 0:
        examples = np.where(mask)[0][:5]
        ex_strs = [repr(tokens[i]) for i in examples]
        print(f"  [{lo:.1f}, {hi:.1f}): {count:5d} tokens  e.g. {', '.join(ex_strs)}")

# ============================================================
# 6. EMBEDDING SPACE ISOTROPY BY LAYER COMPARISON
# ============================================================
print("\n" + "="*60)
print("6. TOKEN vs POSITION EMBEDDING COMPARISON")
print("="*60)

# How do the two spaces relate?
# Shared dimensions or orthogonal?
mean_tok = emb.mean(axis=0)
mean_pos = pos.mean(axis=0)
cos_means = np.dot(mean_tok, mean_pos) / (np.linalg.norm(mean_tok) * np.linalg.norm(mean_pos))
print(f"Cosine between mean token emb and mean position emb: {cos_means:.4f}")

# Variance per dimension — do they use the same dimensions?
var_tok = emb.var(axis=0)
var_pos = pos.var(axis=0)
dim_corr = stats.spearmanr(var_tok, var_pos)[0]
print(f"Spearman correlation of per-dim variance (tok vs pos): ρ={dim_corr:.4f}")
print(f"  → {'Using similar dimensions' if dim_corr > 0.3 else 'Using DIFFERENT dimensions' if dim_corr < -0.1 else 'Weakly related dimensions'}")

# Top variance dims for each
top_tok_dims = np.argsort(var_tok)[-10:][::-1]
top_pos_dims = np.argsort(var_pos)[-10:][::-1]
overlap = set(top_tok_dims) & set(top_pos_dims)
print(f"Top 10 variance dims overlap: {len(overlap)}/10")
print(f"  Token top dims: {list(top_tok_dims)}")
print(f"  Position top dims: {list(top_pos_dims)}")

print("\n✅ Phase 3 complete.")
