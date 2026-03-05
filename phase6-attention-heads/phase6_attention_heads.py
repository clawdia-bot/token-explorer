"""
Phase 6: Attention Head Analysis in Pythia-70m Layer 6
======================================================
The big question from Phase 5: Pythia's output alignment jumps from 0.16 (L5)
to 0.95 (L6) in a single layer. What drives this?

Pythia uses parallel residual: h = h + attn(LN(h)) + mlp(LN(h))
So the L6 residual has three components:
  1. The residual stream from L5 (contributes ~0.16 alignment)
  2. L6 attention output
  3. L6 MLP output

Plan:
  A. Decompose L6 into attn vs MLP contributions to output alignment
  B. Per-head ablation: zero each of the 8 heads, measure alignment drop
  C. Per-head output alignment: which heads' outputs point toward output embeddings?
  D. Attention pattern analysis: what are the critical heads attending to?
  E. Residual stream decomposition: cumulative contribution through all 6 layers
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

out_emb = state['embed_out.weight'].float()


# ── RoPE ─────────────────────────────────────────────────────

def apply_rotary_emb(x, cos, sin):
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x1 = x_rot[..., :rotary_dim//2]
    x2 = x_rot[..., rotary_dim//2:]
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    x_rot_out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([x_rot_out, x_pass], dim=-1)


def get_rope(seq_len):
    inv_freq = state['gpt_neox.layers.0.attention.rotary_emb.inv_freq'].float()
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


# ── Layer forward with detailed outputs ──────────────────────

def layer_forward(h, layer_idx, cos, sin, return_head_outputs=False):
    """Forward one Pythia layer. Returns (h_new, attn_out, mlp_out, [head_outputs], [attn_weights])."""
    seq_len = h.shape[0]
    prefix = f'gpt_neox.layers.{layer_idx}'

    # LN for attention
    ln_w = state[f'{prefix}.input_layernorm.weight'].float()
    ln_b = state[f'{prefix}.input_layernorm.bias'].float()
    h_norm = F.layer_norm(h, (HIDDEN,), ln_w, ln_b)

    # QKV
    qkv_w = state[f'{prefix}.attention.query_key_value.weight'].float()
    qkv_b = state[f'{prefix}.attention.query_key_value.bias'].float()
    qkv = h_norm @ qkv_w.t() + qkv_b
    qkv = qkv.view(seq_len, N_HEADS, 3, HEAD_DIM)
    q = qkv[:, :, 0, :].transpose(0, 1)
    k = qkv[:, :, 1, :].transpose(0, 1)
    v = qkv[:, :, 2, :].transpose(0, 1)

    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)

    scores = (q @ k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)  # [heads, seq, seq]

    head_v_out = attn_weights @ v  # [heads, seq, head_dim]

    # Output projection — decompose per head
    dense_w = state[f'{prefix}.attention.dense.weight'].float()  # [hidden, hidden]
    dense_b = state[f'{prefix}.attention.dense.bias'].float()

    # dense_w maps concatenated head outputs [h0|h1|...|h7] -> hidden
    # Each head i contributes: head_v_out[i] @ dense_w[:, i*64:(i+1)*64].t()
    head_outputs = []
    for i in range(N_HEADS):
        ho = head_v_out[i]  # [seq, head_dim]
        w_slice = dense_w[:, i*HEAD_DIM:(i+1)*HEAD_DIM]  # [hidden, head_dim]
        head_outputs.append(ho @ w_slice.t())  # [seq, hidden]

    attn_out_full = head_v_out.transpose(0, 1).contiguous().view(seq_len, HIDDEN)
    attn_out = attn_out_full @ dense_w.t() + dense_b

    # MLP (parallel — from same LN input, but using post_attention_layernorm)
    pln_w = state[f'{prefix}.post_attention_layernorm.weight'].float()
    pln_b = state[f'{prefix}.post_attention_layernorm.bias'].float()
    h_norm_mlp = F.layer_norm(h, (HIDDEN,), pln_w, pln_b)

    mlp_up_w = state[f'{prefix}.mlp.dense_h_to_4h.weight'].float()
    mlp_up_b = state[f'{prefix}.mlp.dense_h_to_4h.bias'].float()
    mlp_down_w = state[f'{prefix}.mlp.dense_4h_to_h.weight'].float()
    mlp_down_b = state[f'{prefix}.mlp.dense_4h_to_h.bias'].float()

    mlp_out = h_norm_mlp @ mlp_up_w.t() + mlp_up_b
    mlp_out = F.gelu(mlp_out)
    mlp_out = mlp_out @ mlp_down_w.t() + mlp_down_b

    h_new = h + attn_out + mlp_out

    return h_new, attn_out, mlp_out, head_outputs, attn_weights


def full_forward_to_layer(input_ids, stop_after_layer=None):
    """Forward pass returning hidden states and optional detailed L6 info."""
    seq_len = input_ids.shape[0]
    cos, sin = get_rope(seq_len)

    wte = state['gpt_neox.embed_in.weight'].float()
    h = wte[input_ids]

    layer_details = {}
    for layer_idx in range(N_LAYERS):
        h_new, attn_out, mlp_out, head_outputs, attn_weights = layer_forward(
            h, layer_idx, cos, sin, return_head_outputs=True
        )
        layer_details[layer_idx] = {
            'h_pre': h.clone(),
            'attn_out': attn_out,
            'mlp_out': mlp_out,
            'head_outputs': [ho.clone() for ho in head_outputs],
            'attn_weights': attn_weights.clone(),
            'h_post': h_new.clone(),
        }
        h = h_new
        if stop_after_layer is not None and layer_idx == stop_after_layer:
            break

    # Final LN
    ln_f_w = state['gpt_neox.final_layer_norm.weight'].float()
    ln_f_b = state['gpt_neox.final_layer_norm.bias'].float()
    h_final = F.layer_norm(h, (HIDDEN,), ln_f_w, ln_f_b)

    return h_final, layer_details


def output_alignment(hidden, input_ids):
    """Mean cosine between hidden[i] and out_emb[next_token]."""
    cosines = []
    for j in range(len(input_ids) - 1):
        next_id = input_ids[j + 1].item()
        cos = F.cosine_similarity(hidden[j].unsqueeze(0), out_emb[next_id].unsqueeze(0)).item()
        cosines.append(cos)
    return np.mean(cosines)


# ── Test sentences ───────────────────────────────────────────

sentences = [
    "The king and queen ruled the kingdom with wisdom and grace.",
    "Machine learning models can generate surprisingly coherent text.",
    "In 1969, astronauts landed on the moon for the first time.",
    "The cat sat on the mat and watched the birds outside.",
    "Philosophy asks questions that science cannot yet answer.",
    "Tokyo is the capital of Japan, and Paris is the capital of France.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "Quantum mechanics describes the behavior of particles at very small scales.",
]


# ══════════════════════════════════════════════════════════════
# PART A: Decompose L6 into attn vs MLP contribution
# ══════════════════════════════════════════════════════════════

print("=" * 80)
print("PART A: L6 DECOMPOSITION — Attention vs MLP vs Residual")
print("=" * 80)

ln_f_w = state['gpt_neox.final_layer_norm.weight'].float()
ln_f_b = state['gpt_neox.final_layer_norm.bias'].float()

all_residual_align = []
all_attn_align = []
all_mlp_align = []
all_full_align = []

for sent in sentences:
    ids = torch.tensor(tok.encode(sent), dtype=torch.long)
    with torch.no_grad():
        _, details = full_forward_to_layer(ids)

    d6 = details[5]  # Layer index 5 = L6
    h_pre = d6['h_pre']       # residual stream before L6
    attn_out = d6['attn_out']  # L6 attention contribution
    mlp_out = d6['mlp_out']    # L6 MLP contribution
    h_post = d6['h_post']      # full L6 output

    # Apply final LN to each component added to residual
    h_residual_only = F.layer_norm(h_pre, (HIDDEN,), ln_f_w, ln_f_b)
    h_residual_attn = F.layer_norm(h_pre + attn_out, (HIDDEN,), ln_f_w, ln_f_b)
    h_residual_mlp = F.layer_norm(h_pre + mlp_out, (HIDDEN,), ln_f_w, ln_f_b)
    h_full = F.layer_norm(h_post, (HIDDEN,), ln_f_w, ln_f_b)

    all_residual_align.append(output_alignment(h_residual_only, ids))
    all_attn_align.append(output_alignment(h_residual_attn, ids))
    all_mlp_align.append(output_alignment(h_residual_mlp, ids))
    all_full_align.append(output_alignment(h_full, ids))

print(f"  Residual only (skip L6):     {np.mean(all_residual_align):.4f}")
print(f"  Residual + attn only:        {np.mean(all_attn_align):.4f}")
print(f"  Residual + MLP only:         {np.mean(all_mlp_align):.4f}")
print(f"  Full L6 (residual+attn+mlp): {np.mean(all_full_align):.4f}")
print(f"\n  Attn boost:  {np.mean(all_attn_align) - np.mean(all_residual_align):+.4f}")
print(f"  MLP boost:   {np.mean(all_mlp_align) - np.mean(all_residual_align):+.4f}")
print(f"  Combined:    {np.mean(all_full_align) - np.mean(all_residual_align):+.4f}")


# ══════════════════════════════════════════════════════════════
# PART B: Per-head ablation (zero each head, measure alignment drop)
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PART B: PER-HEAD ABLATION — Zero each L6 head")
print("=" * 80)

head_ablation_drops = {i: [] for i in range(N_HEADS)}

for sent in sentences:
    ids = torch.tensor(tok.encode(sent), dtype=torch.long)
    with torch.no_grad():
        _, details = full_forward_to_layer(ids)

    d6 = details[5]
    h_pre = d6['h_pre']
    attn_out = d6['attn_out']
    mlp_out = d6['mlp_out']
    head_outputs = d6['head_outputs']

    # Full alignment
    h_full = F.layer_norm(h_pre + attn_out + mlp_out, (HIDDEN,), ln_f_w, ln_f_b)
    full_align = output_alignment(h_full, ids)

    # Ablate each head: subtract its contribution from attn_out
    # Note: attn_out = sum(head_outputs) + bias (bias is shared, leave it)
    for head_i in range(N_HEADS):
        attn_ablated = attn_out - head_outputs[head_i]
        h_ablated = F.layer_norm(h_pre + attn_ablated + mlp_out, (HIDDEN,), ln_f_w, ln_f_b)
        ablated_align = output_alignment(h_ablated, ids)
        head_ablation_drops[head_i].append(full_align - ablated_align)

print(f"  {'Head':>6} {'MeanDrop':>10} {'StdDrop':>10} {'Impact':>10}")
print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
for i in range(N_HEADS):
    drops = head_ablation_drops[i]
    mean_d = np.mean(drops)
    std_d = np.std(drops)
    impact = "HIGH" if abs(mean_d) > 0.05 else "med" if abs(mean_d) > 0.02 else "low"
    print(f"  H{i:>4} {mean_d:>+10.4f} {std_d:>10.4f} {impact:>10}")


# ══════════════════════════════════════════════════════════════
# PART C: Per-head output alignment with output embeddings
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PART C: PER-HEAD OUTPUT DIRECTION — Alignment with output embeddings")
print("=" * 80)

head_direct_align = {i: [] for i in range(N_HEADS)}

for sent in sentences:
    ids = torch.tensor(tok.encode(sent), dtype=torch.long)
    with torch.no_grad():
        _, details = full_forward_to_layer(ids)

    d6 = details[5]
    head_outputs = d6['head_outputs']

    for head_i in range(N_HEADS):
        ho = head_outputs[head_i]  # [seq, hidden]
        cosines = []
        for j in range(len(ids) - 1):
            next_id = ids[j + 1].item()
            cos = F.cosine_similarity(ho[j].unsqueeze(0), out_emb[next_id].unsqueeze(0)).item()
            cosines.append(cos)
        head_direct_align[head_i].append(np.mean(cosines))

print(f"  {'Head':>6} {'MeanAlign':>10} {'Role':>20}")
print(f"  {'─'*6} {'─'*10} {'─'*20}")
for i in range(N_HEADS):
    align = np.mean(head_direct_align[i])
    role = "→ output space" if align > 0.05 else "orthogonal" if abs(align) < 0.02 else "← away"
    print(f"  H{i:>4} {align:>+10.4f} {role:>20}")


# ══════════════════════════════════════════════════════════════
# PART D: Attention patterns of critical heads
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PART D: ATTENTION PATTERNS OF L6 HEADS")
print("=" * 80)

# Use one sentence for interpretable visualization
demo_sent = "The cat sat on the mat and watched the birds outside."
demo_ids = torch.tensor(tok.encode(demo_sent), dtype=torch.long)
demo_tokens = [tok.decode([t]) for t in demo_ids]

with torch.no_grad():
    _, details = full_forward_to_layer(demo_ids)

d6 = details[5]
attn_w = d6['attn_weights']  # [heads, seq, seq]

print(f"  Tokens: {demo_tokens}")
print(f"  Seq length: {len(demo_tokens)}\n")

for head_i in range(N_HEADS):
    w = attn_w[head_i]  # [seq, seq]
    print(f"  Head {head_i}:")

    # What does each position primarily attend to?
    # Compute entropy and dominant attention target
    entropy = -(w * (w + 1e-10).log()).sum(dim=-1)  # [seq]
    mean_entropy = entropy.mean().item()
    max_entropy = math.log(len(demo_tokens))

    # Attention to position 0 (BOS-like), self, and previous token
    attn_to_first = w[:, 0].mean().item()
    diag_attn = torch.diagonal(w).mean().item()
    # Previous token attention (off-diagonal by 1)
    prev_attn = 0
    for j in range(1, len(demo_tokens)):
        prev_attn += w[j, j-1].item()
    prev_attn /= (len(demo_tokens) - 1)

    # Induction-like: does it attend to token after previous occurrence of current token?
    # (simplified check)

    print(f"    Entropy: {mean_entropy:.2f} / {max_entropy:.2f} (normalized: {mean_entropy/max_entropy:.2f})")
    print(f"    Attn to pos 0: {attn_to_first:.3f}")
    print(f"    Self-attention: {diag_attn:.3f}")
    print(f"    Previous-token: {prev_attn:.3f}")

    # Show top-3 attention targets for last few positions
    for pos in range(max(0, len(demo_tokens)-3), len(demo_tokens)):
        top3 = w[pos].topk(3)
        targets = [(demo_tokens[idx], val.item()) for val, idx in zip(top3.values, top3.indices)]
        print(f"    '{demo_tokens[pos]}' attends to: {', '.join(f'{t}({v:.2f})' for t,v in targets)}")
    print()


# ══════════════════════════════════════════════════════════════
# PART E: Residual stream decomposition — all 6 layers
# ══════════════════════════════════════════════════════════════

print("=" * 80)
print("PART E: RESIDUAL STREAM DECOMPOSITION — All layers")
print("=" * 80)
print("  Output alignment after cumulatively adding each layer's contribution\n")

for sent in sentences[:3]:  # First 3 for detail
    ids = torch.tensor(tok.encode(sent), dtype=torch.long)
    with torch.no_grad():
        _, details = full_forward_to_layer(ids)

    print(f"  \"{sent[:60]}...\"")

    # Start from embedding
    wte = state['gpt_neox.embed_in.weight'].float()
    h_cumulative = wte[ids]

    for layer_idx in range(N_LAYERS):
        d = details[layer_idx]
        attn_out = d['attn_out']
        mlp_out = d['mlp_out']

        # Alignment before this layer
        h_pre_ln = F.layer_norm(h_cumulative, (HIDDEN,), ln_f_w, ln_f_b)
        pre_align = output_alignment(h_pre_ln, ids)

        # Add attention only
        h_plus_attn = h_cumulative + attn_out
        h_plus_attn_ln = F.layer_norm(h_plus_attn, (HIDDEN,), ln_f_w, ln_f_b)
        attn_align = output_alignment(h_plus_attn_ln, ids)

        # Add both
        h_cumulative = h_cumulative + attn_out + mlp_out
        h_post_ln = F.layer_norm(h_cumulative, (HIDDEN,), ln_f_w, ln_f_b)
        post_align = output_alignment(h_post_ln, ids)

        print(f"    L{layer_idx+1}: pre={pre_align:+.3f} → +attn={attn_align:+.3f} → +mlp={post_align:+.3f}"
              f"  (attn Δ={attn_align-pre_align:+.3f}, mlp Δ={post_align-attn_align:+.3f})")

    print()


# ══════════════════════════════════════════════════════════════
# PART F: Head specialization — norm and variance analysis
# ══════════════════════════════════════════════════════════════

print("=" * 80)
print("PART F: HEAD OUTPUT NORMS ACROSS ALL LAYERS")
print("=" * 80)
print("  Which heads contribute the most energy at each layer?\n")

for sent in sentences[:2]:
    ids = torch.tensor(tok.encode(sent), dtype=torch.long)
    with torch.no_grad():
        _, details = full_forward_to_layer(ids)

    print(f"  \"{sent[:60]}\"")
    print(f"  {'Layer':>7} " + " ".join(f"{'H'+str(i):>7}" for i in range(N_HEADS)))

    for layer_idx in range(N_LAYERS):
        d = details[layer_idx]
        head_norms = [ho.norm(dim=-1).mean().item() for ho in d['head_outputs']]
        print(f"  L{layer_idx+1:>5} " + " ".join(f"{n:>7.2f}" for n in head_norms))
    print()


# ── Save results ─────────────────────────────────────────────
results = {
    'decomposition': {
        'residual_only': float(np.mean(all_residual_align)),
        'residual_attn': float(np.mean(all_attn_align)),
        'residual_mlp': float(np.mean(all_mlp_align)),
        'full': float(np.mean(all_full_align)),
    },
    'head_ablation': {
        f'H{i}': float(np.mean(head_ablation_drops[i])) for i in range(N_HEADS)
    },
    'head_direct_alignment': {
        f'H{i}': float(np.mean(head_direct_align[i])) for i in range(N_HEADS)
    },
}

out_path = Path(__file__).parent / 'phase6_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done!")
