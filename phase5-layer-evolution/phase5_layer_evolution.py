"""
Phase 5: Layer Evolution in Pythia-70m (and GPT-2)
===================================================
How does Pythia's perfectly isotropic input embedding become anisotropic 
through its 6 transformer layers?

Manual forward pass using raw state dicts (torch 2.2.2, no HF model classes).

Key questions:
1. At which layer does anisotropy emerge?
2. Does the final hidden state converge toward output embedding structure?
3. How do norms, participation ratio, and effective dimensionality change per layer?
4. Do analogies start working in later layers?
5. Does semantic clustering emerge gradually or suddenly?
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import math
from pathlib import Path

# ── Load raw weights ─────────────────────────────────────────

def load_state(model_name):
    if model_name == 'gpt2':
        snap = os.path.expanduser(
            '~/.cache/huggingface/hub/models--gpt2/snapshots/'
            '607a30d783dfa663caf39e06633721c8d4cfcd7e'
        )
    elif model_name == 'pythia':
        snap_dir = os.path.expanduser(
            '~/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots'
        )
        snap = os.path.join(snap_dir, os.listdir(snap_dir)[0])
    return torch.load(os.path.join(snap, 'pytorch_model.bin'), map_location='cpu')


# ── Tokenizer (lightweight, no model classes) ────────────────
from transformers import AutoTokenizer
tok_gpt2 = AutoTokenizer.from_pretrained('gpt2')
tok_pythia = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')


# ── GPT-2 Manual Forward Pass ───────────────────────────────
# Architecture: Pre-LN GPT-2 (LN before attn, LN before MLP, sequential residual)
# 12 layers, 12 heads, hidden=768, head_dim=64

def gpt2_forward(state, input_ids):
    """Manual GPT-2 forward pass, collecting hidden states after each layer."""
    seq_len = input_ids.shape[0]
    hidden = 768
    n_heads = 12
    head_dim = 64
    
    # Embedding
    wte = state['wte.weight'].float()
    wpe = state['wpe.weight'].float()
    h = wte[input_ids] + wpe[:seq_len]
    
    hidden_states = [h.clone()]  # Layer 0 = embedding output
    
    for layer_idx in range(12):
        prefix = f'h.{layer_idx}'
        
        # Layer Norm 1
        ln1_w = state[f'{prefix}.ln_1.weight'].float()
        ln1_b = state[f'{prefix}.ln_1.bias'].float()
        h_norm = F.layer_norm(h, (hidden,), ln1_w, ln1_b)
        
        # Self-Attention (fused QKV)
        attn_w = state[f'{prefix}.attn.c_attn.weight'].float()  # [hidden, 3*hidden]
        attn_b = state[f'{prefix}.attn.c_attn.bias'].float()
        qkv = h_norm @ attn_w + attn_b  # [seq, 3*hidden]
        q, k, v = qkv.split(hidden, dim=-1)
        
        # Reshape to heads
        q = q.view(seq_len, n_heads, head_dim).transpose(0, 1)  # [heads, seq, head_dim]
        k = k.view(seq_len, n_heads, head_dim).transpose(0, 1)
        v = v.view(seq_len, n_heads, head_dim).transpose(0, 1)
        
        # Attention scores with causal mask
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        # Attention output
        attn_out = attn_weights @ v  # [heads, seq, head_dim]
        attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, hidden)
        
        # Output projection
        proj_w = state[f'{prefix}.attn.c_proj.weight'].float()
        proj_b = state[f'{prefix}.attn.c_proj.bias'].float()
        attn_out = attn_out @ proj_w + proj_b
        
        # Residual
        h = h + attn_out
        
        # Layer Norm 2
        ln2_w = state[f'{prefix}.ln_2.weight'].float()
        ln2_b = state[f'{prefix}.ln_2.bias'].float()
        h_norm2 = F.layer_norm(h, (hidden,), ln2_w, ln2_b)
        
        # MLP
        mlp_fc_w = state[f'{prefix}.mlp.c_fc.weight'].float()
        mlp_fc_b = state[f'{prefix}.mlp.c_fc.bias'].float()
        mlp_proj_w = state[f'{prefix}.mlp.c_proj.weight'].float()
        mlp_proj_b = state[f'{prefix}.mlp.c_proj.bias'].float()
        
        mlp_out = h_norm2 @ mlp_fc_w + mlp_fc_b
        mlp_out = F.gelu(mlp_out, approximate='tanh')  # GPT-2 uses approx gelu
        mlp_out = mlp_out @ mlp_proj_w + mlp_proj_b
        
        # Residual
        h = h + mlp_out
        
        hidden_states.append(h.clone())
    
    # Final layer norm
    ln_f_w = state['ln_f.weight'].float()
    ln_f_b = state['ln_f.bias'].float()
    h_final = F.layer_norm(h, (hidden,), ln_f_w, ln_f_b)
    hidden_states.append(h_final)  # Post-LN as extra "layer"
    
    return hidden_states


# ── Pythia (GPT-NeoX) Manual Forward Pass ────────────────────
# Architecture: Parallel residual (attn + MLP computed in parallel from same LN output)
# 6 layers, 8 heads, hidden=512, head_dim=64, rotary_dim=16
# RoPE applied to first 16 dims of each head

def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to the first rotary_dim dimensions."""
    # x: [heads, seq, head_dim], cos/sin: [seq, rotary_dim/2]
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    
    x1 = x_rot[..., :rotary_dim//2]
    x2 = x_rot[..., rotary_dim//2:]
    
    # Rotate
    cos = cos.unsqueeze(0)  # [1, seq, rotary_dim/2]
    sin = sin.unsqueeze(0)
    x_rot_out = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], dim=-1)
    
    return torch.cat([x_rot_out, x_pass], dim=-1)


def pythia_forward(state, input_ids):
    """Manual Pythia-70m forward pass with parallel residual and RoPE."""
    seq_len = input_ids.shape[0]
    hidden = 512
    n_heads = 8
    head_dim = 64
    rotary_dim = 16  # 0.25 * 64
    
    # Embedding (no positional — RoPE handles it)
    wte = state['gpt_neox.embed_in.weight'].float()
    h = wte[input_ids]
    
    hidden_states = [h.clone()]
    
    # Precompute RoPE frequencies
    inv_freq = state['gpt_neox.layers.0.attention.rotary_emb.inv_freq'].float()
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [seq, rotary_dim/2]
    cos_cached = freqs.cos()
    sin_cached = freqs.sin()
    
    for layer_idx in range(6):
        prefix = f'gpt_neox.layers.{layer_idx}'
        
        # Input layer norm (shared for both attn and MLP in parallel residual)
        ln_w = state[f'{prefix}.input_layernorm.weight'].float()
        ln_b = state[f'{prefix}.input_layernorm.bias'].float()
        h_norm = F.layer_norm(h, (hidden,), ln_w, ln_b)
        
        # === Attention branch ===
        qkv_w = state[f'{prefix}.attention.query_key_value.weight'].float()
        qkv_b = state[f'{prefix}.attention.query_key_value.bias'].float()
        qkv = h_norm @ qkv_w.t() + qkv_b  # [seq, 3*hidden]
        
        # Split into Q, K, V — GPT-NeoX interleaves heads
        qkv = qkv.view(seq_len, n_heads, 3, head_dim)
        q = qkv[:, :, 0, :].transpose(0, 1)  # [heads, seq, head_dim]
        k = qkv[:, :, 1, :].transpose(0, 1)
        v = qkv[:, :, 2, :].transpose(0, 1)
        
        # Apply RoPE
        q = apply_rotary_emb(q, cos_cached, sin_cached)
        k = apply_rotary_emb(k, cos_cached, sin_cached)
        
        # Attention
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_out = attn_weights @ v
        attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, hidden)
        
        # Output projection
        dense_w = state[f'{prefix}.attention.dense.weight'].float()
        dense_b = state[f'{prefix}.attention.dense.bias'].float()
        attn_out = attn_out @ dense_w.t() + dense_b
        
        # === MLP branch (parallel — uses SAME h_norm, not post-attention) ===
        # Post-attention layer norm is used for MLP input in parallel mode
        pln_w = state[f'{prefix}.post_attention_layernorm.weight'].float()
        pln_b = state[f'{prefix}.post_attention_layernorm.bias'].float()
        h_norm_mlp = F.layer_norm(h, (hidden,), pln_w, pln_b)
        
        mlp_up_w = state[f'{prefix}.mlp.dense_h_to_4h.weight'].float()
        mlp_up_b = state[f'{prefix}.mlp.dense_h_to_4h.bias'].float()
        mlp_down_w = state[f'{prefix}.mlp.dense_4h_to_h.weight'].float()
        mlp_down_b = state[f'{prefix}.mlp.dense_4h_to_h.bias'].float()
        
        mlp_out = h_norm_mlp @ mlp_up_w.t() + mlp_up_b
        mlp_out = F.gelu(mlp_out)  # Pythia uses exact GELU
        mlp_out = mlp_out @ mlp_down_w.t() + mlp_down_b
        
        # Parallel residual: h = h + attn_out + mlp_out
        h = h + attn_out + mlp_out
        
        hidden_states.append(h.clone())
    
    # Final layer norm
    ln_f_w = state['gpt_neox.final_layer_norm.weight'].float()
    ln_f_b = state['gpt_neox.final_layer_norm.bias'].float()
    h_final = F.layer_norm(h, (hidden,), ln_f_w, ln_f_b)
    hidden_states.append(h_final)
    
    return hidden_states


# ── Metrics ──────────────────────────────────────────────────

def compute_metrics(H, n_pairs=5000):
    """Anisotropy, norms, participation ratio, PC1% for a [n_tokens, hidden] tensor."""
    H = H.float()
    n, d = H.shape
    
    # Anisotropy
    np.random.seed(42)
    idx = np.random.choice(n, size=(min(n_pairs, n*(n-1)//2), 2), replace=True)
    mask = idx[:, 0] != idx[:, 1]
    idx = idx[mask]
    cos = F.cosine_similarity(H[idx[:, 0]], H[idx[:, 1]], dim=1)
    
    # Norms
    norms = torch.norm(H, dim=1)
    
    # SVD
    centered = H - H.mean(dim=0)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    var_exp = (S ** 2) / (S ** 2).sum()
    pr = 1.0 / (var_exp ** 2).sum().item()
    cumvar = var_exp.cumsum(0)
    dims_90 = (cumvar < 0.9).sum().item() + 1
    
    return {
        'anisotropy': cos.mean().item(),
        'norm_mean': norms.mean().item(),
        'norm_std': norms.std().item(),
        'pc1_pct': var_exp[0].item() * 100,
        'top5_pct': var_exp[:5].sum().item() * 100,
        'participation_ratio': pr,
        'dims_for_90pct': dims_90,
    }


# ── Main ─────────────────────────────────────────────────────

print("Loading weights...")
gpt2_state = load_state('gpt2')
pythia_state = load_state('pythia')
print("Done.\n")

# Input sentences
sentences = [
    "The king and queen ruled the kingdom with wisdom and grace.",
    "Machine learning models can generate surprisingly coherent text.",
    "In 1969, astronauts landed on the moon for the first time.",
    "The cat sat on the mat and watched the birds outside.",
    "Philosophy asks questions that science cannot yet answer.",
    "Tokyo is the capital of Japan, and Paris is the capital of France.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "Quantum mechanics describes the behavior of particles at very small scales.",
    "She walked through the garden, admiring the flowers and the butterflies.",
    "Mathematics is the language in which the universe is written.",
    "The stock market crashed in 2008, causing a global recession.",
    "Dogs are loyal companions, while cats prefer their independence.",
    "The ancient Romans built roads that still exist today across Europe.",
    "Artificial intelligence is transforming every industry on the planet.",
    "He picked up the book and began reading the first chapter slowly.",
    "The temperature dropped below zero, and snow covered the streets.",
]


def collect_all_hidden_states(forward_fn, state, tokenizer, sentences):
    """Run all sentences, collect hidden states per layer (concatenated across sentences)."""
    all_layers = None
    
    for sent in sentences:
        ids = torch.tensor(tokenizer.encode(sent), dtype=torch.long)
        with torch.no_grad():
            layer_states = forward_fn(state, ids)
        
        if all_layers is None:
            all_layers = [[] for _ in range(len(layer_states))]
        for i, h in enumerate(layer_states):
            all_layers[i].append(h)
    
    return [torch.cat(layer_list, dim=0) for layer_list in all_layers]


print("Running GPT-2 forward pass through all layers...")
gpt2_layers = collect_all_hidden_states(gpt2_forward, gpt2_state, tok_gpt2, sentences)
print(f"  {len(gpt2_layers)} checkpoints (emb + 12 layers + final_LN), {gpt2_layers[0].shape[0]} tokens")

print("Running Pythia forward pass through all layers...")
pythia_layers = collect_all_hidden_states(pythia_forward, pythia_state, tok_pythia, sentences)
print(f"  {len(pythia_layers)} checkpoints (emb + 6 layers + final_LN), {pythia_layers[0].shape[0]} tokens")


# ── Layer-by-layer metrics ───────────────────────────────────
print("\n" + "=" * 80)
print("LAYER-BY-LAYER METRICS")
print("=" * 80)

results = {}

for model_name, layers, n_transformer_layers in [
    ('GPT-2 (12L)', gpt2_layers, 12),
    ('Pythia-70m (6L)', pythia_layers, 6)
]:
    key = 'gpt2' if 'GPT-2' in model_name else 'pythia'
    results[key] = []
    
    print(f"\n{'─' * 80}")
    print(f"  {model_name}")
    print(f"{'─' * 80}")
    print(f"  {'Layer':>8} {'Aniso':>8} {'NormMu':>8} {'NormSig':>8} {'PC1%':>7} {'Top5%':>7} {'PR':>6} {'D90':>5}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*5}")
    
    for i, layer_h in enumerate(layers):
        m = compute_metrics(layer_h)
        results[key].append(m)
        
        if i == 0:
            label = "emb"
        elif i <= n_transformer_layers:
            label = f"L{i}"
        else:
            label = "final_LN"
        
        print(f"  {label:>8} {m['anisotropy']:>8.4f} {m['norm_mean']:>8.2f} "
              f"{m['norm_std']:>8.2f} {m['pc1_pct']:>6.1f}% {m['top5_pct']:>6.1f}% "
              f"{m['participation_ratio']:>6.0f} {m['dims_for_90pct']:>5}")


# ── Semantic clustering through layers ───────────────────────
print("\n" + "=" * 80)
print("SEMANTIC CLUSTERING THROUGH LAYERS")
print("=" * 80)

cluster_sent = "The dog and cat chased the bird while the mathematician solved the equation and the physicist studied particles"

for model_name, forward_fn, state, tokenizer in [
    ('GPT-2', gpt2_forward, gpt2_state, tok_gpt2),
    ('Pythia', pythia_forward, pythia_state, tok_pythia)
]:
    print(f"\n  {model_name}:")
    ids = torch.tensor(tokenizer.encode(cluster_sent), dtype=torch.long)
    tokens = [tokenizer.decode([t]) for t in ids]
    
    with torch.no_grad():
        layer_states = forward_fn(state, ids)
    
    # Find token positions for semantic groups
    def find_tok(word):
        target_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(target_ids) != 1:
            return None
        target_id = target_ids[0]
        for i, t in enumerate(ids.tolist()):
            if t == target_id:
                return i
        return None
    
    animals = [p for p in [find_tok(' dog'), find_tok(' cat'), find_tok(' bird')] if p is not None]
    science = [p for p in [find_tok(' mathematician'), find_tok(' equation'), find_tok(' physicist'), find_tok(' particles')] if p is not None]
    
    if len(animals) < 2 or len(science) < 2:
        print(f"    Tokens: {tokens}")
        print(f"    Animals found: {len(animals)}, Science found: {len(science)} — skipping")
        continue
    
    print(f"    Animals: {[tokens[i] for i in animals]}")
    print(f"    Science: {[tokens[i] for i in science]}")
    
    n_layers = len(layer_states)
    for i, h in enumerate(layer_states):
        H = h.float()
        
        def mean_pw_cos(positions):
            if len(positions) < 2:
                return float('nan')
            vecs = H[positions]
            cos = F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=2)
            mask = torch.triu(torch.ones(len(positions), len(positions), dtype=bool), diagonal=1)
            return cos[mask].mean().item()
        
        def mean_cross_cos(pa, pb):
            return F.cosine_similarity(
                H[pa].unsqueeze(1), H[pb].unsqueeze(0), dim=2
            ).mean().item()
        
        within_a = mean_pw_cos(animals)
        within_s = mean_pw_cos(science)
        between = mean_cross_cos(animals, science)
        sep = (within_a + within_s) / 2 - between
        
        if i == 0: label = "emb"
        elif i < n_layers - 1: label = f"L{i}"
        else: label = "final_LN"
        
        print(f"    {label:>8}: animals={within_a:+.3f} science={within_s:+.3f} "
              f"between={between:+.3f} sep={sep:+.3f}")


# ── Convergence to output embeddings (Pythia) ───────────────
print("\n" + "=" * 80)
print("PYTHIA: HIDDEN STATE → OUTPUT EMBEDDING CONVERGENCE")
print("=" * 80)

out_emb = pythia_state['embed_out.weight'].float()

conv_sent = "The cat sat on the mat and watched the birds outside"
ids_p = torch.tensor(tok_pythia.encode(conv_sent), dtype=torch.long)

with torch.no_grad():
    p_layers = pythia_forward(pythia_state, ids_p)

print(f"  Sentence: {conv_sent}")
print(f"  Mean cosine between hidden[i] and output_emb[next_token]:")
print(f"  {'Layer':>8} {'MeanCos':>10}")

for i, h in enumerate(p_layers):
    h = h.float()
    cosines = []
    for j in range(len(ids_p) - 1):
        next_id = ids_p[j + 1].item()
        cos = F.cosine_similarity(h[j].unsqueeze(0), out_emb[next_id].unsqueeze(0)).item()
        cosines.append(cos)
    
    mean_cos = np.mean(cosines)
    if i == 0: label = "emb"
    elif i <= 6: label = f"L{i}"
    else: label = "final_LN"
    print(f"  {label:>8} {mean_cos:>10.4f}")


# ── Analogies through layers ────────────────────────────────
print("\n" + "=" * 80)
print("ANALOGIES THROUGH LAYERS (in-context)")
print("=" * 80)

analogy_sent = "The king and queen met the man and woman in France near Paris and in Japan near Tokyo"

analogy_pairs = [
    (" king", " queen", " man", " woman"),
    (" France", " Paris", " Japan", " Tokyo"),
]

for model_name, forward_fn, state, tokenizer in [
    ('GPT-2', gpt2_forward, gpt2_state, tok_gpt2),
    ('Pythia', pythia_forward, pythia_state, tok_pythia)
]:
    print(f"\n  {model_name}:")
    ids = torch.tensor(tokenizer.encode(analogy_sent), dtype=torch.long)
    tokens = [tokenizer.decode([t]) for t in ids]
    
    with torch.no_grad():
        layer_states = forward_fn(state, ids)
    
    def find_tok_pos(word):
        target_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(target_ids) != 1:
            return None
        target_id = target_ids[0]
        for i, t in enumerate(ids.tolist()):
            if t == target_id:
                return i
        return None
    
    for a, b, c, expected in analogy_pairs:
        pa, pb, pc, pe = find_tok_pos(a), find_tok_pos(b), find_tok_pos(c), find_tok_pos(expected)
        if any(p is None for p in [pa, pb, pc, pe]):
            print(f"    {a}:{b}::{c}:{expected} — token not found")
            continue
        
        print(f"    {a}:{b}::{c}:? (expected:{expected})")
        for i, h in enumerate(layer_states):
            H = h.float()
            vec = H[pb] - H[pa] + H[pc]
            
            cos_exp = F.cosine_similarity(vec.unsqueeze(0), H[pe].unsqueeze(0)).item()
            
            # Best match excluding a, b, c
            cos_all = F.cosine_similarity(vec.unsqueeze(0), H, dim=1)
            cos_all[pa] = -2; cos_all[pb] = -2; cos_all[pc] = -2
            best = cos_all.argmax().item()
            best_cos = cos_all[best].item()
            
            hit = "✓" if best == pe else "✗"
            if i == 0: label = "emb"
            elif i < len(layer_states) - 1: label = f"L{i}"
            else: label = "final_LN"
            
            print(f"      {label:>8}: cos_expected={cos_exp:.3f} "
                  f"nearest={repr(tokens[best])}({best_cos:.3f}) {hit}")


# ── Save JSON ────────────────────────────────────────────────
out_path = Path(__file__).parent / 'phase5_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved metrics to {out_path}")

print("\nDone!")
