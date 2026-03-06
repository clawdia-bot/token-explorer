"""
Phase 8: Neuron Activation Patterns in Pythia-70m
==================================================
Phase 7 found that L6 MLP is nonlinear, Neuron 348 dominates, and 10% are dead.
Now: what do individual neurons *respond to*? Can we find interpretable patterns?

Questions:
A. Neuron-to-token mapping: For the top neurons, which tokens trigger the
   highest activations? Are there semantic/syntactic patterns?
B. Token-to-neuron mapping: For specific tokens, which neurons fire?
   Do similar tokens activate similar neuron subsets?
C. Neuron activation similarity: Do neurons cluster into functional groups?
D. L0 vs L5 comparison: Are early-layer neurons qualitatively different
   from late-layer neurons?
E. Sparse coding: How many neurons does each token actually need?
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import math
from pathlib import Path
from collections import defaultdict
import os

# ── Load weights and tokenizer ───────────────────────────────
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')

snap_dir = os.path.expanduser(
    '~/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots'
)
snap = os.path.join(snap_dir, os.listdir(snap_dir)[0])
state = torch.load(os.path.join(snap, 'pytorch_model.bin'), map_location='cpu')

HIDDEN = 512
N_HEADS = 8
HEAD_DIM = 64
N_LAYERS = 6
ROTARY_DIM = 16

out_emb = state['embed_out.weight'].float()

# ── Reuse forward pass infrastructure from Phase 7 ───────────

def apply_rotary_emb(x, cos, sin):
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x1 = x_rot[..., :rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2:]
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    x_rot_out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([x_rot_out, x_pass], dim=-1)


def get_rotary(seq_len):
    inv_freq = state['gpt_neox.layers.0.attention.rotary_emb.inv_freq'].float()
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def layer_norm(x, weight, bias):
    return F.layer_norm(x, (HIDDEN,), weight, bias)


def attention(h_ln, layer_idx, cos, sin):
    pfx = f'gpt_neox.layers.{layer_idx}.attention'
    qkv_w = state[f'{pfx}.query_key_value.weight'].float()
    qkv_b = state[f'{pfx}.query_key_value.bias'].float()
    seq_len = h_ln.shape[0]
    qkv = h_ln @ qkv_w.T + qkv_b
    qkv = qkv.view(seq_len, N_HEADS, 3, HEAD_DIM)
    q = qkv[:, :, 0, :].transpose(0, 1)
    k = qkv[:, :, 1, :].transpose(0, 1)
    v = qkv[:, :, 2, :].transpose(0, 1)
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, HIDDEN)
    o_w = state[f'{pfx}.dense.weight'].float()
    o_b = state[f'{pfx}.dense.bias'].float()
    return attn_out @ o_w.T + o_b


def mlp_with_activations(h_ln, layer_idx):
    """MLP forward pass, also returns intermediate (post-GELU) activations."""
    pfx = f'gpt_neox.layers.{layer_idx}.mlp'
    up_w = state[f'{pfx}.dense_h_to_4h.weight'].float()
    up_b = state[f'{pfx}.dense_h_to_4h.bias'].float()
    down_w = state[f'{pfx}.dense_4h_to_h.weight'].float()
    down_b = state[f'{pfx}.dense_4h_to_h.bias'].float()
    h_up = h_ln @ up_w.T + up_b
    h_act = F.gelu(h_up)
    output = h_act @ down_w.T + down_b
    return output, h_act


def mlp(h_ln, layer_idx):
    pfx = f'gpt_neox.layers.{layer_idx}.mlp'
    up_w = state[f'{pfx}.dense_h_to_4h.weight'].float()
    up_b = state[f'{pfx}.dense_h_to_4h.bias'].float()
    down_w = state[f'{pfx}.dense_4h_to_h.weight'].float()
    down_b = state[f'{pfx}.dense_4h_to_h.bias'].float()
    h_up = h_ln @ up_w.T + up_b
    h_act = F.gelu(h_up)
    return h_act @ down_w.T + down_b


def forward_to_layer(tokens, target_layer):
    """Forward pass up to (but not through) target_layer. Returns h and LN(h)."""
    emb_w = state['gpt_neox.embed_in.weight'].float()
    h = emb_w[tokens]
    seq_len = len(tokens)
    cos, sin = get_rotary(seq_len)
    
    for i in range(target_layer):
        ln_w = state[f'gpt_neox.layers.{i}.input_layernorm.weight'].float()
        ln_b = state[f'gpt_neox.layers.{i}.input_layernorm.bias'].float()
        h_ln = layer_norm(h, ln_w, ln_b)
        attn_out = attention(h_ln, i, cos, sin)
        mlp_out = mlp(h_ln, i)
        h = h + attn_out + mlp_out
    
    # Get LN output for target layer
    ln_w = state[f'gpt_neox.layers.{target_layer}.input_layernorm.weight'].float()
    ln_b = state[f'gpt_neox.layers.{target_layer}.input_layernorm.bias'].float()
    h_ln = layer_norm(h, ln_w, ln_b)
    
    return h, h_ln


def get_mlp_activations(tokens, layer_idx):
    """Get post-GELU MLP activations for a specific layer."""
    _, h_ln = forward_to_layer(tokens, layer_idx)
    _, activations = mlp_with_activations(h_ln, layer_idx)
    return activations


def encode(text):
    return torch.tensor(tok.encode(text), dtype=torch.long)


# ── Diverse corpus for activation patterns ────────────────────

corpus = [
    # Factual / encyclopedic
    "The capital of France is Paris and the capital of Germany is Berlin",
    "Water boils at one hundred degrees Celsius at sea level",
    "Albert Einstein published the theory of relativity in nineteen fifteen",
    "The human body contains approximately two hundred and six bones",
    
    # Narrative
    "She walked into the dark room and immediately noticed something was wrong",
    "Once upon a time there was a little girl who lived in the forest",
    "The detective examined the evidence carefully before making his conclusion",
    "He ran as fast as he could but the train had already left the station",
    
    # Technical
    "The function returns a pointer to the first element of the array",
    "Neural networks learn representations through gradient descent optimization",
    "The algorithm has a time complexity of O n log n in the average case",
    "Memory allocation failed because the heap was already full",
    
    # Conversational
    "I think we should go to the park today if the weather is nice",
    "What do you think about the new restaurant that opened downtown",
    "Can you please pass me the salt and pepper from the table",
    "Sorry I am late the traffic was really bad this morning",
    
    # Numbers and structure
    "There are three hundred and sixty five days in a year",
    "The meeting is scheduled for Monday at ten o clock in the morning",
    "Chapter one introduces the main characters and sets the scene",
    "First you need to mix the flour and sugar then add the eggs",
    
    # Abstract / philosophical
    "The meaning of life is a question that has puzzled philosophers for centuries",
    "Time is an illusion created by the movement of matter through space",
    "Consciousness remains one of the greatest mysteries in modern science",
    "Freedom and responsibility are two sides of the same coin",
]


# ══════════════════════════════════════════════════════════════
# PART A: Neuron-to-Token Mapping (L5, the prediction layer)
# ══════════════════════════════════════════════════════════════

print("=" * 70)
print("PART A: NEURON ACTIVATION PATTERNS (L5)")
print("=" * 70)

# Collect activations with token labels
token_labels = []
all_acts_L5 = []

for sent in corpus:
    tokens = encode(sent)
    acts = get_mlp_activations(tokens, 5)  # Layer 5 (L6 in 1-indexed)
    all_acts_L5.append(acts.detach())
    for t in tokens:
        token_labels.append(tok.decode([t.item()]))

acts_L5 = torch.cat(all_acts_L5, dim=0)  # [total_tokens, 2048]
print(f"\nTotal tokens: {acts_L5.shape[0]}")

# Basic stats
mean_act = acts_L5.mean(dim=0)
fire_rate = (acts_L5 > 0.1).float().mean(dim=0)

# Find dead and always-on neurons
dead_mask = fire_rate < 0.01
always_on_mask = fire_rate > 0.90
dead_count = dead_mask.sum().item()
always_on_count = always_on_mask.sum().item()
print(f"Dead neurons (<1% fire rate): {dead_count}/2048 ({dead_count/2048*100:.1f}%)")
print(f"Always-on neurons (>90% fire rate): {always_on_count}/2048")

# Top neurons by mean activation
top_neurons = mean_act.argsort(descending=True)[:30]

print("\n--- Top 30 neurons: What tokens trigger them? ---\n")

for rank, neuron_idx in enumerate(top_neurons[:20]):
    n = neuron_idx.item()
    neuron_acts = acts_L5[:, n]  # activations of this neuron across all tokens
    
    # Top activating tokens
    top_token_indices = neuron_acts.argsort(descending=True)[:15]
    top_tokens_with_acts = [(token_labels[i], neuron_acts[i].item()) for i in top_token_indices]
    
    # Bottom (zero/negative) activating tokens
    bot_token_indices = neuron_acts.argsort()[:10]
    bot_tokens_with_acts = [(token_labels[i], neuron_acts[i].item()) for i in bot_token_indices]
    
    print(f"Neuron {n} (mean={mean_act[n]:.2f}, fire={fire_rate[n]*100:.0f}%):")
    top_str = ", ".join([f"{t!r}({a:.1f})" for t, a in top_tokens_with_acts[:8]])
    print(f"  MAX: {top_str}")
    
    # Try to identify pattern
    top_toks = [t for t, _ in top_tokens_with_acts[:15]]
    print(f"  All top: {top_toks}")
    print()


# ══════════════════════════════════════════════════════════════
# PART B: Token-to-Neuron Mapping
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART B: TOKEN-TO-NEURON MAPPING")
print("=" * 70)
print("For specific token types, which neurons fire? Do similar tokens")
print("activate similar neuron patterns?\n")

# Group tokens by type
type_groups = {
    'articles': [' the', ' a', ' an', ' The'],
    'prepositions': [' in', ' on', ' at', ' of', ' to', ' from', ' for', ' with'],
    'verbs_past': [' walked', ' ran', ' published', ' examined', ' opened', ' noticed'],
    'verbs_present': [' is', ' are', ' has', ' think', ' need', ' remains'],
    'nouns_concrete': [' room', ' park', ' table', ' train', ' eggs', ' salt'],
    'nouns_abstract': [' life', ' time', ' freedom', ' consciousness', ' science'],
    'adjectives': [' dark', ' nice', ' new', ' bad', ' great', ' little', ' first'],
    'punctuation_like': [' and', ' but', ' or', ' then', ' that', ' because'],
}

# For each occurrence of each token type, get its neuron activation pattern
print("--- Neuron activation profiles by token type ---\n")

type_profiles = {}

for type_name, type_tokens in type_groups.items():
    matching_acts = []
    matched_labels = []
    
    for i, label in enumerate(token_labels):
        if label in type_tokens:
            matching_acts.append(acts_L5[i])
            matched_labels.append(label)
    
    if len(matching_acts) < 2:
        continue
    
    matching_acts = torch.stack(matching_acts)  # [n_matches, 2048]
    
    # Mean activation profile for this type
    mean_profile = matching_acts.mean(dim=0)
    type_profiles[type_name] = mean_profile
    
    # Top neurons for this type
    top5 = mean_profile.argsort(descending=True)[:10]
    
    # Sparsity: how many neurons fire on average
    active_count = (matching_acts > 0.1).float().mean(dim=0).sum().item()
    
    print(f"{type_name} ({len(matching_acts)} tokens: {matched_labels[:5]}):")
    print(f"  Active neurons (mean): {active_count:.0f}/2048 ({active_count/2048*100:.1f}%)")
    top_str = ", ".join([f"N{t.item()}({mean_profile[t]:.1f})" for t in top5[:6]])
    print(f"  Top neurons: {top_str}")
    print()


# Cross-type similarity
print("\n--- Cross-type neuron pattern similarity ---")
print("Cosine similarity between mean activation profiles:\n")

type_names = list(type_profiles.keys())
print(f"{'':>20}", end="")
for name in type_names:
    print(f"{name[:8]:>10}", end="")
print()

for name_a in type_names:
    print(f"{name_a:>20}", end="")
    for name_b in type_names:
        sim = F.cosine_similarity(
            type_profiles[name_a].unsqueeze(0),
            type_profiles[name_b].unsqueeze(0)
        ).item()
        print(f"{sim:>10.3f}", end="")
    print()


# ══════════════════════════════════════════════════════════════
# PART C: Sparsity Analysis — How Many Neurons Per Token?
# ══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("PART C: SPARSITY ANALYSIS")
print("=" * 70)
print("How many neurons does each token actually activate?\n")

# Active neuron count per token (threshold: activation > 0.1)
active_per_token = (acts_L5 > 0.1).float().sum(dim=1)  # [total_tokens]

print(f"Active neurons per token:")
print(f"  Mean:   {active_per_token.mean():.1f}")
print(f"  Median: {active_per_token.median():.1f}")
print(f"  Min:    {active_per_token.min():.0f}")
print(f"  Max:    {active_per_token.max():.0f}")
print(f"  Std:    {active_per_token.std():.1f}")

# Distribution
print(f"\nDistribution of active neuron counts:")
for threshold in [100, 200, 500, 800, 1000, 1200, 1500]:
    count = (active_per_token < threshold).sum().item()
    print(f"  <{threshold}: {count}/{len(active_per_token)} tokens ({count/len(active_per_token)*100:.1f}%)")

# Tokens with fewest and most active neurons
print(f"\nTokens with FEWEST active neurons:")
sparse_indices = active_per_token.argsort()[:10]
for idx in sparse_indices:
    i = idx.item()
    print(f"  {token_labels[i]!r}: {active_per_token[i]:.0f} neurons")

print(f"\nTokens with MOST active neurons:")
dense_indices = active_per_token.argsort(descending=True)[:10]
for idx in dense_indices:
    i = idx.item()
    print(f"  {token_labels[i]!r}: {active_per_token[i]:.0f} neurons")

# Energy concentration: what fraction of total activation is in top-K neurons?
print(f"\nEnergy concentration (mean across tokens):")
sorted_acts, _ = acts_L5.sort(dim=1, descending=True)
total_energy = sorted_acts.sum(dim=1, keepdim=True).clamp(min=1e-10)
for k in [10, 20, 50, 100, 200]:
    frac = (sorted_acts[:, :k].sum(dim=1) / total_energy.squeeze()).mean().item()
    print(f"  Top {k:>4} neurons: {frac*100:.1f}% of total activation")


# ══════════════════════════════════════════════════════════════
# PART D: L0 vs L5 Neuron Comparison
# ══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("PART D: L0 vs L5 NEURON COMPARISON")
print("=" * 70)
print("L0 and L5 are the two load-bearing layers. Are their neurons different?\n")

# Collect L0 activations
all_acts_L0 = []
for sent in corpus:
    tokens = encode(sent)
    acts = get_mlp_activations(tokens, 0)  # Layer 0
    all_acts_L0.append(acts.detach())

acts_L0 = torch.cat(all_acts_L0, dim=0)

mean_act_L0 = acts_L0.mean(dim=0)
fire_rate_L0 = (acts_L0 > 0.1).float().mean(dim=0)
active_per_token_L0 = (acts_L0 > 0.1).float().sum(dim=1)

dead_L0 = (fire_rate_L0 < 0.01).sum().item()
always_on_L0 = (fire_rate_L0 > 0.90).sum().item()

print(f"{'Metric':<35} {'L0':>10} {'L5':>10}")
print("-" * 55)
print(f"{'Dead neurons (<1% fire)':<35} {dead_L0:>10} {dead_count:>10}")
print(f"{'Always-on neurons (>90% fire)':<35} {always_on_L0:>10} {always_on_count:>10}")
print(f"{'Mean activation (all neurons)':<35} {mean_act_L0.mean():.4f}{'':<3} {mean_act.mean():.4f}")
print(f"{'Max mean activation':<35} {mean_act_L0.max():.2f}{'':<5} {mean_act.max():.2f}")
print(f"{'Active neurons/token (mean)':<35} {active_per_token_L0.mean():.1f}{'':<4} {active_per_token.mean():.1f}")
print(f"{'Active neurons/token (std)':<35} {active_per_token_L0.std():.1f}{'':<4} {active_per_token.std():.1f}")

# Activation magnitude distribution
print(f"\nActivation magnitude distribution:")
for layer_name, acts in [("L0", acts_L0), ("L5", acts_L5)]:
    nonzero = acts[acts > 0.1]
    print(f"  {layer_name}: mean={nonzero.mean():.2f}, median={nonzero.median():.2f}, "
          f"p95={nonzero.quantile(0.95):.2f}, max={nonzero.max():.2f}")

# Top neurons comparison
print(f"\nTop 10 neurons by mean activation:")
top_L0 = mean_act_L0.argsort(descending=True)[:10]
top_L5 = mean_act.argsort(descending=True)[:10]

print(f"  L0: {[f'N{n.item()}({mean_act_L0[n]:.1f})' for n in top_L0]}")
print(f"  L5: {[f'N{n.item()}({mean_act[n]:.1f})' for n in top_L5]}")

# L0 top neuron patterns
print(f"\n--- L0 Top Neurons: What do they respond to? ---\n")
for rank, neuron_idx in enumerate(top_L0[:10]):
    n = neuron_idx.item()
    neuron_acts = acts_L0[:, n]
    top_token_indices = neuron_acts.argsort(descending=True)[:10]
    top_toks = [token_labels[i] for i in top_token_indices]
    print(f"  L0 N{n} (mean={mean_act_L0[n]:.2f}, fire={fire_rate_L0[n]*100:.0f}%): {top_toks[:8]}")


# Energy concentration comparison
print(f"\nEnergy concentration comparison:")
sorted_L0, _ = acts_L0.sort(dim=1, descending=True)
total_L0 = sorted_L0.sum(dim=1, keepdim=True).clamp(min=1e-10)
sorted_L5, _ = acts_L5.sort(dim=1, descending=True)
total_L5 = sorted_L5.sum(dim=1, keepdim=True).clamp(min=1e-10)

for k in [10, 50, 100, 200]:
    frac_L0 = (sorted_L0[:, :k].sum(dim=1) / total_L0.squeeze()).mean().item()
    frac_L5 = (sorted_L5[:, :k].sum(dim=1) / total_L5.squeeze()).mean().item()
    print(f"  Top {k:>4}: L0={frac_L0*100:.1f}%  L5={frac_L5*100:.1f}%")


# ══════════════════════════════════════════════════════════════
# PART E: Neuron Clustering — Do Neurons Form Functional Groups?
# ══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("PART E: NEURON FUNCTIONAL CLUSTERING (L5)")
print("=" * 70)
print("Do neurons with similar activation patterns cluster together?\n")

# Use only non-dead neurons for clustering
alive_mask = ~dead_mask
alive_indices = torch.where(alive_mask)[0]
alive_acts = acts_L5[:, alive_mask]  # [tokens, alive_neurons]

print(f"Alive neurons: {alive_indices.shape[0]}")

# Compute neuron-neuron correlation matrix
# Each neuron is a vector of activations across tokens
# Transpose: [neurons, tokens]
neuron_vecs = alive_acts.T  # [alive_neurons, tokens]

# Correlation via cosine similarity (sample if too many neurons)
n_alive = neuron_vecs.shape[0]
print(f"Computing {n_alive}×{n_alive} cosine similarity matrix...")

# Normalize
neuron_vecs_n = neuron_vecs / neuron_vecs.norm(dim=1, keepdim=True).clamp(min=1e-10)
corr_matrix = neuron_vecs_n @ neuron_vecs_n.T  # [alive, alive]

# Distribution of correlations
upper_tri = corr_matrix[torch.triu(torch.ones(n_alive, n_alive), diagonal=1).bool()]
print(f"\nNeuron-neuron correlation distribution:")
print(f"  Mean:   {upper_tri.mean():.4f}")
print(f"  Std:    {upper_tri.std():.4f}")
print(f"  Min:    {upper_tri.min():.4f}")
print(f"  Max:    {upper_tri.max():.4f}")
print(f"  >0.8:   {(upper_tri > 0.8).sum().item()} pairs")
print(f"  >0.9:   {(upper_tri > 0.9).sum().item()} pairs")
print(f"  <-0.3:  {(upper_tri < -0.3).sum().item()} pairs (anti-correlated)")

# Most correlated neuron pairs
top_corr_flat = upper_tri.argsort(descending=True)[:20]
# Convert flat indices back to matrix indices
triu_indices = torch.triu(torch.ones(n_alive, n_alive), diagonal=1).nonzero()

print(f"\nMost correlated neuron pairs:")
for flat_idx in top_corr_flat[:10]:
    i, j = triu_indices[flat_idx]
    ni = alive_indices[i].item()
    nj = alive_indices[j].item()
    corr = upper_tri[flat_idx].item()
    print(f"  N{ni} ↔ N{nj}: {corr:.4f}")

# Most anti-correlated pairs
bot_corr_flat = upper_tri.argsort()[:10]
print(f"\nMost anti-correlated neuron pairs:")
for flat_idx in bot_corr_flat[:5]:
    i, j = triu_indices[flat_idx]
    ni = alive_indices[i].item()
    nj = alive_indices[j].item()
    corr = upper_tri[flat_idx].item()
    
    # What tokens activate each?
    acts_i = acts_L5[:, ni]
    acts_j = acts_L5[:, nj]
    top_i = [token_labels[k] for k in acts_i.argsort(descending=True)[:5]]
    top_j = [token_labels[k] for k in acts_j.argsort(descending=True)[:5]]
    
    print(f"  N{ni} ↔ N{nj}: {corr:.4f}")
    print(f"    N{ni} fires on: {top_i}")
    print(f"    N{nj} fires on: {top_j}")


# ══════════════════════════════════════════════════════════════
# PART F: Neuron Selectivity — Generalist vs Specialist
# ══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("PART F: NEURON SELECTIVITY — GENERALISTS vs SPECIALISTS")
print("=" * 70)
print("Some neurons fire for everything (generalists), others are selective.\n")

# Selectivity = entropy of activation distribution across tokens
# Low entropy = specialist (fires hard on few tokens)
# High entropy = generalist (fires similarly on many tokens)

# Use softmax of activations as a probability distribution
alive_acts_pos = acts_L5[:, alive_mask].clamp(min=0)  # [tokens, neurons]

# For each neuron, compute entropy of its activation distribution
neuron_entropies = []
max_entropy = math.log(acts_L5.shape[0])  # uniform distribution entropy

for neuron_local_idx in range(alive_acts_pos.shape[1]):
    neuron_col = alive_acts_pos[:, neuron_local_idx]
    total = neuron_col.sum()
    if total < 0.01:
        neuron_entropies.append(0.0)
        continue
    probs = neuron_col / total
    # Filter zeros for log
    probs_nz = probs[probs > 0]
    entropy = -(probs_nz * probs_nz.log()).sum().item()
    neuron_entropies.append(entropy / max_entropy)  # normalize to [0, 1]

neuron_entropies = torch.tensor(neuron_entropies)

print(f"Normalized entropy distribution (0=specialist, 1=generalist):")
print(f"  Mean:   {neuron_entropies.mean():.4f}")
print(f"  Std:    {neuron_entropies.std():.4f}")
print(f"  <0.5:   {(neuron_entropies < 0.5).sum().item()} neurons (specialist-leaning)")
print(f"  >0.9:   {(neuron_entropies > 0.9).sum().item()} neurons (strong generalists)")
print(f"  <0.3:   {(neuron_entropies < 0.3).sum().item()} neurons (strong specialists)")

# Most specialist neurons
specialist_order = neuron_entropies.argsort()
print(f"\nMost SPECIALIST neurons (lowest entropy):")
for rank in range(10):
    local_idx = specialist_order[rank].item()
    n = alive_indices[local_idx].item()
    ent = neuron_entropies[local_idx].item()
    neuron_acts_here = acts_L5[:, n]
    top_tok_idx = neuron_acts_here.argsort(descending=True)[:8]
    top_toks = [(token_labels[i], neuron_acts_here[i].item()) for i in top_tok_idx]
    toks_str = ", ".join([f"{t!r}({a:.1f})" for t, a in top_toks])
    print(f"  N{n} (entropy={ent:.4f}, fire={fire_rate[n]*100:.0f}%): {toks_str}")

# Most generalist neurons
generalist_order = neuron_entropies.argsort(descending=True)
print(f"\nMost GENERALIST neurons (highest entropy):")
for rank in range(10):
    local_idx = generalist_order[rank].item()
    n = alive_indices[local_idx].item()
    ent = neuron_entropies[local_idx].item()
    print(f"  N{n} (entropy={ent:.4f}, mean_act={mean_act[n]:.2f}, fire={fire_rate[n]*100:.0f}%)")


# ══════════════════════════════════════════════════════════════
# Summary stats for JSON export
# ══════════════════════════════════════════════════════════════

results = {
    "total_tokens": int(acts_L5.shape[0]),
    "corpus_size": len(corpus),
    "L5": {
        "dead_neurons": dead_count,
        "always_on_neurons": always_on_count,
        "mean_active_per_token": float(active_per_token.mean()),
        "median_active_per_token": float(active_per_token.median()),
        "top5_neurons": [int(n.item()) for n in top_neurons[:5]],
        "mean_neuron_entropy": float(neuron_entropies.mean()),
        "neuron_correlation_mean": float(upper_tri.mean()),
        "high_corr_pairs_gt_0_8": int((upper_tri > 0.8).sum().item()),
    },
    "L0": {
        "dead_neurons": dead_L0,
        "always_on_neurons": always_on_L0,
        "mean_active_per_token": float(active_per_token_L0.mean()),
    }
}

with open(Path(__file__).parent / "phase8_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\nResults saved to phase8_results.json")
print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
