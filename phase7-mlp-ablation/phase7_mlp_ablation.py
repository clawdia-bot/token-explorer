"""
Phase 7: MLP Decomposition & Layer Ablation in Pythia-70m
=========================================================
Two questions:

A. MLP Decomposition: Is L6's MLP a simple linear projection into output space,
   or does the nonlinearity (GELU) matter? If we replace GELU with identity,
   how much alignment survives?
   
   Also: what does the MLP's "up" projection select for? Which neurons fire
   most, and what tokens activate them?

B. Layer Ablation: What happens to final output alignment when we remove
   individual layers? Which layers are load-bearing vs. decorative?
   
   Also: cumulative ablation — remove layers in order of least important
   to find the minimal circuit.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import math
from pathlib import Path
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

out_emb = state['embed_out.weight'].float()  # [vocab, 512]

# ── RoPE ─────────────────────────────────────────────────────

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


# ── Forward pass building blocks ─────────────────────────────

def layer_norm(x, weight, bias):
    return F.layer_norm(x, (HIDDEN,), weight, bias)


def attention(h_ln, layer_idx, cos, sin):
    """Full multi-head attention for one layer."""
    pfx = f'gpt_neox.layers.{layer_idx}.attention'
    qkv_w = state[f'{pfx}.query_key_value.weight'].float()  # [3*512, 512]
    qkv_b = state[f'{pfx}.query_key_value.bias'].float()
    
    seq_len = h_ln.shape[0]
    qkv = h_ln @ qkv_w.T + qkv_b  # [seq, 3*512]
    
    # Split into per-head q, k, v — transpose to [heads, seq, head_dim] first
    qkv = qkv.view(seq_len, N_HEADS, 3, HEAD_DIM)
    q = qkv[:, :, 0, :].transpose(0, 1)  # [heads, seq, head_dim]
    k = qkv[:, :, 1, :].transpose(0, 1)
    v = qkv[:, :, 2, :].transpose(0, 1)
    
    # Apply RoPE to q, k (expects [heads, seq, head_dim])
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    
    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)  # [heads, seq, head_dim]
    
    attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, HIDDEN)
    
    # Output projection
    o_w = state[f'{pfx}.dense.weight'].float()
    o_b = state[f'{pfx}.dense.bias'].float()
    return attn_out @ o_w.T + o_b


def mlp(h_ln, layer_idx, linear_mode=False):
    """MLP for one layer. If linear_mode=True, replace GELU with identity."""
    pfx = f'gpt_neox.layers.{layer_idx}.mlp'
    up_w = state[f'{pfx}.dense_h_to_4h.weight'].float()  # [2048, 512]
    up_b = state[f'{pfx}.dense_h_to_4h.bias'].float()
    down_w = state[f'{pfx}.dense_4h_to_h.weight'].float()  # [512, 2048]
    down_b = state[f'{pfx}.dense_4h_to_h.bias'].float()
    
    h_up = h_ln @ up_w.T + up_b  # [seq, 2048]
    
    if linear_mode:
        h_act = h_up  # skip GELU
    else:
        h_act = F.gelu(h_up)
    
    return h_act @ down_w.T + down_b


def full_forward(tokens, skip_layers=None, linear_mlp_layers=None):
    """
    Full forward pass through Pythia-70m.
    skip_layers: set of layer indices to skip entirely
    linear_mlp_layers: set of layer indices where MLP uses identity instead of GELU
    Returns final hidden states [seq, 512].
    """
    skip_layers = skip_layers or set()
    linear_mlp_layers = linear_mlp_layers or set()
    
    # Embedding
    emb_w = state['gpt_neox.embed_in.weight'].float()
    h = emb_w[tokens]  # [seq, 512]
    
    seq_len = len(tokens)
    cos, sin = get_rotary(seq_len)
    
    for i in range(N_LAYERS):
        if i in skip_layers:
            continue
        
        # Pythia uses parallel residual: h = h + attn(LN(h)) + mlp(LN(h))
        ln_w = state[f'gpt_neox.layers.{i}.input_layernorm.weight'].float()
        ln_b = state[f'gpt_neox.layers.{i}.input_layernorm.bias'].float()
        
        # Pythia-70m uses same LN for both attn and MLP (parallel)
        h_ln = layer_norm(h, ln_w, ln_b)
        
        attn_out = attention(h_ln, i, cos, sin)
        mlp_out = mlp(h_ln, i, linear_mode=(i in linear_mlp_layers))
        
        h = h + attn_out + mlp_out
    
    # Final layer norm
    ln_f_w = state['gpt_neox.final_layer_norm.weight'].float()
    ln_f_b = state['gpt_neox.final_layer_norm.bias'].float()
    h = layer_norm(h, ln_f_w, ln_f_b)
    
    return h


def output_alignment(h):
    """Mean cosine similarity between hidden states and their top-1 predicted output embedding."""
    # For each position, find the most likely next token and measure alignment
    logits = h @ out_emb.T  # [seq, vocab]
    top_tokens = logits.argmax(dim=-1)  # [seq]
    top_embs = out_emb[top_tokens]  # [seq, 512]
    
    cos_sim = F.cosine_similarity(h, top_embs, dim=-1)
    return cos_sim.mean().item()


def perplexity(h, target_tokens):
    """Perplexity of the model on target tokens (shifted by 1)."""
    logits = h @ out_emb.T  # [seq, vocab]
    # Shift: predict position i+1 from position i
    logits = logits[:-1]  # [seq-1, vocab]
    targets = target_tokens[1:]  # [seq-1]
    
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs[torch.arange(len(targets)), targets]
    
    return torch.exp(-target_log_probs.mean()).item()


# ── Test sentences ────────────────────────────────────────────

sentences = [
    "The cat sat on the mat and looked out the window at the birds",
    "In the beginning was the word and the word was with God",
    "She walked into the room and immediately noticed something was wrong",
    "The president of the United States gave a speech about climate change",
    "Once upon a time there was a little girl who lived in the forest",
]


def encode(text):
    return torch.tensor(tok.encode(text), dtype=torch.long)


# ══════════════════════════════════════════════════════════════
# PART A: MLP Decomposition
# ══════════════════════════════════════════════════════════════

print("=" * 70)
print("PART A: MLP DECOMPOSITION")
print("=" * 70)

# A1: Linear vs nonlinear MLP — does GELU matter?
print("\n--- A1: GELU vs Identity in L6 MLP ---")
print("If L6 MLP is ~linear, replacing GELU with identity should preserve alignment.\n")

for sent in sentences[:3]:
    tokens = encode(sent)
    
    # Normal forward
    h_normal = full_forward(tokens)
    align_normal = output_alignment(h_normal)
    ppl_normal = perplexity(h_normal, tokens)
    
    # L6 MLP with identity instead of GELU
    h_linear = full_forward(tokens, linear_mlp_layers={5})
    align_linear = output_alignment(h_linear)
    ppl_linear = perplexity(h_linear, tokens)
    
    print(f'"{sent[:50]}..."')
    print(f'  Normal:     alignment={align_normal:.4f}  ppl={ppl_normal:.1f}')
    print(f'  Linear L6:  alignment={align_linear:.4f}  ppl={ppl_linear:.1f}')
    print(f'  Δ alignment: {align_linear - align_normal:+.4f}  Δ ppl: {ppl_linear - ppl_normal:+.1f}')
    print()


# A2: What about making ALL MLPs linear?
print("\n--- A2: All MLPs linear (GELU → identity everywhere) ---\n")

for sent in sentences[:3]:
    tokens = encode(sent)
    
    h_normal = full_forward(tokens)
    align_normal = output_alignment(h_normal)
    ppl_normal = perplexity(h_normal, tokens)
    
    h_all_linear = full_forward(tokens, linear_mlp_layers=set(range(N_LAYERS)))
    align_all_linear = output_alignment(h_all_linear)
    ppl_all_linear = perplexity(h_all_linear, tokens)
    
    print(f'"{sent[:50]}..."')
    print(f'  Normal:      alignment={align_normal:.4f}  ppl={ppl_normal:.1f}')
    print(f'  All linear:  alignment={align_all_linear:.4f}  ppl={ppl_all_linear:.1f}')
    print(f'  Δ alignment: {align_all_linear - align_normal:+.4f}  Δ ppl: {ppl_all_linear - ppl_normal:+.1f}')
    print()


# A3: Neuron-level analysis — which L6 MLP neurons fire most?
print("\n--- A3: L6 MLP Neuron Analysis ---")
print("Which of the 2048 intermediate neurons fire hardest?\n")

pfx = 'gpt_neox.layers.5.mlp'
up_w = state[f'{pfx}.dense_h_to_4h.weight'].float()  # [2048, 512]
up_b = state[f'{pfx}.dense_h_to_4h.bias'].float()
down_w = state[f'{pfx}.dense_4h_to_h.weight'].float()  # [512, 2048]
down_b = state[f'{pfx}.dense_4h_to_h.bias'].float()

# Collect activations across all sentences
all_activations = []
for sent in sentences:
    tokens = encode(sent)
    
    # Forward to get L6 input
    emb_w = state['gpt_neox.embed_in.weight'].float()
    h = emb_w[tokens]
    seq_len = len(tokens)
    cos, sin = get_rotary(seq_len)
    
    for i in range(N_LAYERS):
        ln_w = state[f'gpt_neox.layers.{i}.input_layernorm.weight'].float()
        ln_b = state[f'gpt_neox.layers.{i}.input_layernorm.bias'].float()
        h_ln = layer_norm(h, ln_w, ln_b)
        
        if i == 5:  # L6 — grab pre-MLP activations
            h_up = h_ln @ up_w.T + up_b  # [seq, 2048]
            h_gelu = F.gelu(h_up)
            all_activations.append(h_gelu.detach())
        
        attn_out = attention(h_ln, i, cos, sin)
        mlp_out = mlp(h_ln, i)
        h = h + attn_out + mlp_out

activations = torch.cat(all_activations, dim=0)  # [total_tokens, 2048]
print(f"Total tokens analyzed: {activations.shape[0]}")

# Mean activation per neuron
mean_act = activations.mean(dim=0)  # [2048]
max_act = activations.max(dim=0).values
std_act = activations.std(dim=0)

# Fraction of tokens where neuron fires (activation > 0.1)
fire_rate = (activations > 0.1).float().mean(dim=0)

# Top neurons by mean activation
top_by_mean = mean_act.argsort(descending=True)[:20]
print("\nTop 20 neurons by mean activation:")
print(f"{'Neuron':>8} {'Mean':>8} {'Max':>8} {'Std':>8} {'Fire%':>8}")
for idx in top_by_mean:
    i = idx.item()
    print(f"{i:>8} {mean_act[i]:>8.3f} {max_act[i]:>8.3f} {std_act[i]:>8.3f} {fire_rate[i]*100:>7.1f}%")

# Dead neurons (never fire > 0.1)
dead = (fire_rate < 0.01).sum().item()
print(f"\nDead neurons (fire <1% of tokens): {dead}/{2048} ({dead/2048*100:.1f}%)")

# Always-on neurons (fire > 90% of tokens)
always_on = (fire_rate > 0.9).sum().item()
print(f"Always-on neurons (fire >90%): {always_on}/{2048}")


# A4: Neuron output directions — do top neurons point toward output space?
print("\n--- A4: Neuron Output Directions ---")
print("Each neuron's contribution = activation * column of down_proj.")
print("Do the most active neurons' columns align with common output embeddings?\n")

# Top 10 neurons by mean activation
top10 = top_by_mean[:10]
for idx in top10:
    i = idx.item()
    neuron_dir = down_w[:, i]  # [512] — this neuron's output direction
    neuron_dir_n = neuron_dir / neuron_dir.norm()
    
    # Cosine with all output embeddings
    out_emb_n = out_emb / out_emb.norm(dim=1, keepdim=True)
    cos_sims = out_emb_n @ neuron_dir_n  # [vocab]
    
    top5_tokens = cos_sims.argsort(descending=True)[:5]
    bot5_tokens = cos_sims.argsort()[:5]
    
    top_str = ', '.join([f'{tok.decode([t.item()])!r}({cos_sims[t]:.3f})' for t in top5_tokens])
    bot_str = ', '.join([f'{tok.decode([t.item()])!r}({cos_sims[t]:.3f})' for t in bot5_tokens])
    
    print(f"Neuron {i} (mean={mean_act[i]:.3f}, fire={fire_rate[i]*100:.0f}%):")
    print(f"  Promotes: {top_str}")
    print(f"  Suppresses: {bot_str}")
    print()


# A5: Effective rank of MLP weight matrices
print("\n--- A5: Effective Rank of L6 MLP ---")
print("If MLP is low-rank, it's essentially a linear projection.\n")

# W_down @ diag(gelu(x)) @ W_up — but let's check the static matrices
# The composed linear part (ignoring GELU) would be W_down @ W_up
composed = down_w @ up_w  # [512, 512]

# SVD
U, S, Vh = torch.linalg.svd(composed)
total_energy = (S ** 2).sum()
cumvar = torch.cumsum(S ** 2, dim=0) / total_energy

# Effective rank (number of singular values for 90%, 95%, 99% variance)
for threshold in [0.5, 0.9, 0.95, 0.99]:
    rank = (cumvar < threshold).sum().item() + 1
    print(f"  Rank for {threshold*100:.0f}% variance: {rank}/512")

# Participation ratio of singular values
pr = (S.sum() ** 2) / (S ** 2).sum()
print(f"  Participation ratio: {pr.item():.1f}/512")

# How much does the linear approximation (W_down @ W_up) match actual MLP behavior?
print("\n  Comparing W_down @ W_up (linear) vs actual MLP on real inputs:")
for sent in sentences[:2]:
    tokens = encode(sent)
    emb_w = state['gpt_neox.embed_in.weight'].float()
    h = emb_w[tokens]
    seq_len = len(tokens)
    cos_r, sin_r = get_rotary(seq_len)
    
    for i in range(5):  # Forward through layers 0-4
        ln_w = state[f'gpt_neox.layers.{i}.input_layernorm.weight'].float()
        ln_b = state[f'gpt_neox.layers.{i}.input_layernorm.bias'].float()
        h_ln = layer_norm(h, ln_w, ln_b)
        attn_out = attention(h_ln, i, cos_r, sin_r)
        mlp_out = mlp(h_ln, i)
        h = h + attn_out + mlp_out
    
    # L6 input
    ln_w = state['gpt_neox.layers.5.input_layernorm.weight'].float()
    ln_b = state['gpt_neox.layers.5.input_layernorm.bias'].float()
    h_ln = layer_norm(h, ln_w, ln_b)
    
    # Actual MLP output
    mlp_actual = mlp(h_ln, 5, linear_mode=False)
    
    # Linear approximation: W_down @ W_up @ h_ln + bias terms
    mlp_linear_approx = h_ln @ composed.T + (up_b @ down_w.T + down_b)
    
    # Cosine similarity between actual and linear
    cos_sim = F.cosine_similarity(mlp_actual, mlp_linear_approx, dim=-1).mean()
    
    # Relative norm difference
    norm_ratio = mlp_linear_approx.norm() / mlp_actual.norm()
    
    print(f'  "{sent[:40]}...": cos_sim={cos_sim:.4f}, norm_ratio={norm_ratio:.3f}')


# ══════════════════════════════════════════════════════════════
# PART B: Layer Ablation
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART B: LAYER ABLATION")
print("=" * 70)

# B1: Remove each layer individually
print("\n--- B1: Single Layer Ablation ---")
print("Skip one layer at a time, measure alignment and perplexity.\n")

# Use all sentences, average results
all_tokens = [encode(s) for s in sentences]

results = {}
for skip in [None] + list(range(N_LAYERS)):
    skip_set = {skip} if skip is not None else set()
    label = f"Skip L{skip}" if skip is not None else "Full model"
    
    aligns = []
    ppls = []
    for tokens in all_tokens:
        h = full_forward(tokens, skip_layers=skip_set)
        aligns.append(output_alignment(h))
        ppls.append(perplexity(h, tokens))
    
    mean_align = np.mean(aligns)
    mean_ppl = np.mean(ppls)
    results[label] = (mean_align, mean_ppl)
    
    print(f"  {label:<15}  alignment={mean_align:.4f}  ppl={mean_ppl:.1f}")

# B2: Cumulative ablation — remove layers in order of least impact
print("\n--- B2: Cumulative Ablation (greedy) ---")
print("Iteratively remove the least important remaining layer.\n")

remaining = set(range(N_LAYERS))
removed_order = []

for step in range(N_LAYERS):
    best_layer = None
    best_align = -1
    
    for candidate in remaining:
        skip_set = set(range(N_LAYERS)) - remaining | {candidate}
        
        aligns = []
        for tokens in all_tokens:
            h = full_forward(tokens, skip_layers=skip_set)
            aligns.append(output_alignment(h))
        
        mean_align = np.mean(aligns)
        if mean_align > best_align:
            best_align = mean_align
            best_layer = candidate
    
    remaining.remove(best_layer)
    removed_order.append(best_layer)
    skip_set = set(range(N_LAYERS)) - remaining
    
    # Also get ppl
    ppls = []
    for tokens in all_tokens:
        h = full_forward(tokens, skip_layers=skip_set)
        ppls.append(perplexity(h, tokens))
    
    active = sorted(remaining)
    print(f"  Step {step+1}: Remove L{best_layer} → active={active}  alignment={best_align:.4f}  ppl={np.mean(ppls):.1f}")

print(f"\n  Removal order (least → most important): {removed_order}")
print(f"  Last standing: L{removed_order[-1]} (most critical layer)")


# B3: Minimal circuit — just embedding + one layer + output
print("\n--- B3: Minimal Circuits (single layer) ---")
print("Only keep embedding → one layer → final LN → output.\n")

for keep in range(N_LAYERS):
    skip_set = set(range(N_LAYERS)) - {keep}
    
    aligns = []
    ppls = []
    for tokens in all_tokens:
        h = full_forward(tokens, skip_layers=skip_set)
        aligns.append(output_alignment(h))
        ppls.append(perplexity(h, tokens))
    
    print(f"  Only L{keep}:  alignment={np.mean(aligns):.4f}  ppl={np.mean(ppls):.1f}")


print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
