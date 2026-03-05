"""
Phase 4: Cross-Model Embedding Comparison
==========================================
GPT-2 (learned pos, tied weights) vs GPT-Neo-125M (learned pos, tied weights) 
vs Pythia-70m (RoPE, untied weights)

Key questions:
1. Is token embedding anisotropy universal or GPT-2 specific?
2. Does RoPE change token embedding geometry?
3. Are ghost clusters universal?
4. How do norm-frequency relationships compare?
5. Do analogies work in all embedding spaces?
6. GPT-Neo has 2048 positions — does position 0 anomaly persist? What about 1024-2048?
"""

import torch
import numpy as np
import json
import os
from pathlib import Path

# ── Load Models ──────────────────────────────────────────────

def load_gpt2():
    snap = os.path.expanduser(
        '~/.cache/huggingface/hub/models--gpt2/snapshots/'
        '607a30d783dfa663caf39e06633721c8d4cfcd7e'
    )
    state = torch.load(os.path.join(snap, 'pytorch_model.bin'), map_location='cpu')
    return {
        'name': 'GPT-2',
        'wte': state['wte.weight'],
        'wpe': state['wpe.weight'],
        'hidden': 768,
        'vocab': 50257,
        'max_pos': 1024,
        'pos_type': 'learned',
        'tied': True,
    }

def load_gpt_neo():
    snap_dir = os.path.expanduser(
        '~/.cache/huggingface/hub/models--EleutherAI--gpt-neo-125m/snapshots'
    )
    snap = os.path.join(snap_dir, os.listdir(snap_dir)[0])
    state = torch.load(os.path.join(snap, 'pytorch_model.bin'), map_location='cpu')
    return {
        'name': 'GPT-Neo-125M',
        'wte': state['transformer.wte.weight'],
        'wpe': state['transformer.wpe.weight'],
        'hidden': 768,
        'vocab': 50257,
        'max_pos': 2048,
        'pos_type': 'learned',
        'tied': True,
    }

def load_pythia():
    snap_dir = os.path.expanduser(
        '~/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots'
    )
    snap = os.path.join(snap_dir, os.listdir(snap_dir)[0])
    state = torch.load(os.path.join(snap, 'pytorch_model.bin'), map_location='cpu')
    return {
        'name': 'Pythia-70m',
        'wte': state['gpt_neox.embed_in.weight'],
        'wte_out': state['embed_out.weight'],
        'wpe': None,
        'hidden': 512,
        'vocab': 50304,
        'max_pos': 2048,
        'pos_type': 'RoPE',
        'tied': False,
    }

print("Loading models...")
gpt2 = load_gpt2()
neo = load_gpt_neo()
pythia = load_pythia()
models = [gpt2, neo, pythia]
print("All loaded.\n")

# ── Shared tokenizer (GPT-2 family) ─────────────────────────
# GPT-2 and GPT-Neo share the same tokenizer (50257 tokens)
# Pythia uses the same tokenizer but padded to 50304

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')

def token_id(text):
    ids = tok.encode(text)
    assert len(ids) == 1, f"'{text}' encodes to {len(ids)} tokens"
    return ids[0]

# ── 1. Anisotropy ────────────────────────────────────────────
print("=" * 60)
print("1. ANISOTROPY: Mean Pairwise Cosine Similarity")
print("=" * 60)

for m in models:
    wte = m['wte'].float()
    # Sample 5000 random pairs
    n = wte.shape[0]
    np.random.seed(42)
    idx = np.random.choice(n, size=(5000, 2), replace=True)
    vecs_a = wte[idx[:, 0]]
    vecs_b = wte[idx[:, 1]]
    cos = torch.nn.functional.cosine_similarity(vecs_a, vecs_b, dim=1)
    m['anisotropy'] = cos.mean().item()
    m['anisotropy_std'] = cos.std().item()
    print(f"  {m['name']:20s}: mean cos = {m['anisotropy']:.4f} ± {m['anisotropy_std']:.4f}")

print()

# ── 2. Norm Statistics ───────────────────────────────────────
print("=" * 60)
print("2. TOKEN EMBEDDING NORMS")
print("=" * 60)

for m in models:
    wte = m['wte'].float()
    norms = torch.norm(wte, dim=1)
    m['norm_mean'] = norms.mean().item()
    m['norm_std'] = norms.std().item()
    m['norm_min'] = norms.min().item()
    m['norm_max'] = norms.max().item()
    
    # Top-5 highest norm tokens (if GPT-2 tokenizer applies)
    top5 = norms.topk(5)
    print(f"\n  {m['name']}:")
    print(f"    Mean: {m['norm_mean']:.3f} ± {m['norm_std']:.3f}")
    print(f"    Range: [{m['norm_min']:.3f}, {m['norm_max']:.3f}]")
    print(f"    Top-5 norms:")
    for idx, norm in zip(top5.indices, top5.values):
        try:
            token_str = tok.decode([idx.item()])
            print(f"      [{idx.item():5d}] {repr(token_str):20s} norm={norm.item():.3f}")
        except:
            print(f"      [{idx.item():5d}] <decode error>       norm={norm.item():.3f}")

print()

# ── 3. Ghost Cluster Analysis ────────────────────────────────
print("=" * 60)
print("3. GHOST CLUSTER: Control Characters")
print("=" * 60)

# Control chars: tokens 0-31 in GPT-2's byte-level BPE
# These are byte tokens for ASCII control chars
control_ids = list(range(0, 20))  # First 20 byte-level tokens

for m in models:
    wte = m['wte'].float()
    max_id = min(max(control_ids), wte.shape[0] - 1)
    valid_ids = [i for i in control_ids if i < wte.shape[0]]
    ghost = wte[valid_ids]
    
    # Pairwise distances within ghost cluster
    dists = torch.cdist(ghost.unsqueeze(0), ghost.unsqueeze(0)).squeeze()
    # Upper triangle only
    mask = torch.triu(torch.ones_like(dists, dtype=bool), diagonal=1)
    ghost_dist = dists[mask].mean().item()
    
    # Compare to random pairs
    np.random.seed(42)
    rand_idx = np.random.choice(wte.shape[0], size=(100, 2))
    rand_dists = torch.norm(wte[rand_idx[:, 0]] - wte[rand_idx[:, 1]], dim=1)
    rand_dist = rand_dists.mean().item()
    
    ratio = ghost_dist / rand_dist
    m['ghost_dist'] = ghost_dist
    m['ghost_ratio'] = ratio
    print(f"  {m['name']:20s}: ghost L2={ghost_dist:.3f}, random L2={rand_dist:.3f}, ratio={ratio:.3f}")

print()

# ── 4. PC1 Dominance ─────────────────────────────────────────
print("=" * 60)
print("4. PC1 DOMINANCE & ISOTROPY AFTER REMOVAL")
print("=" * 60)

for m in models:
    wte = m['wte'].float()
    # Center
    mean = wte.mean(dim=0)
    centered = wte - mean
    
    # SVD for top PCs
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    
    # Variance explained by PC1
    total_var = (S ** 2).sum().item()
    pc1_var = (S[0] ** 2).item()
    pc1_pct = pc1_var / total_var * 100
    
    # Top-5 PCs variance
    top5_var = (S[:5] ** 2).sum().item() / total_var * 100
    
    # Anisotropy after removing PC1
    pc1_dir = Vh[0]
    proj = (wte @ pc1_dir).unsqueeze(1) * pc1_dir.unsqueeze(0)
    wte_nopc1 = wte - proj
    
    np.random.seed(42)
    n = wte_nopc1.shape[0]
    idx = np.random.choice(n, size=(5000, 2))
    cos_after = torch.nn.functional.cosine_similarity(
        wte_nopc1[idx[:, 0]], wte_nopc1[idx[:, 1]], dim=1
    ).mean().item()
    
    m['pc1_pct'] = pc1_pct
    m['top5_pct'] = top5_var
    m['anisotropy_nopc1'] = cos_after
    
    # Participation ratio
    normalized_s = (S ** 2) / (S ** 2).sum()
    pr = 1.0 / (normalized_s ** 2).sum().item()
    m['participation_ratio'] = pr
    
    print(f"\n  {m['name']}:")
    print(f"    PC1 explains:     {pc1_pct:.1f}%")
    print(f"    Top-5 PCs:        {top5_var:.1f}%")
    print(f"    Participation:    {pr:.0f}/{wte.shape[1]}")
    print(f"    Anisotropy:       {m['anisotropy']:.4f} → {cos_after:.4f} (after PC1 removal)")

print()

# ── 5. Analogies ─────────────────────────────────────────────
print("=" * 60)
print("5. ANALOGY TEST (a:b::c:?)")
print("=" * 60)

analogies = [
    ("king", "queen", "man", "woman"),
    ("France", "Paris", "Japan", "Tokyo"),
    ("big", "bigger", "small", "smaller"),
]

def test_analogy(wte, a, b, c, expected, model_name):
    """a:b::c:? → expected"""
    try:
        id_a, id_b, id_c, id_exp = token_id(a), token_id(b), token_id(c), token_id(expected)
    except:
        return None  # Token not single-token in this tokenizer
    
    if max(id_a, id_b, id_c, id_exp) >= wte.shape[0]:
        return None
    
    vec = wte[id_b].float() - wte[id_a].float() + wte[id_c].float()
    
    # Find nearest, excluding a, b, c
    exclude = {id_a, id_b, id_c}
    cos = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), wte.float(), dim=1)
    for ex in exclude:
        cos[ex] = -2
    
    top5 = cos.topk(5)
    results = []
    for idx, score in zip(top5.indices, top5.values):
        try:
            t = tok.decode([idx.item()])
        except:
            t = f"<{idx.item()}>"
        results.append((t, score.item()))
    
    hit = any(idx.item() == id_exp for idx in top5.indices)
    return results, hit

# Use space-prefixed tokens for GPT-2 BPE
analogy_tokens = [
    (" king", " queen", " man", " woman"),
    (" France", " Paris", " Japan", " Tokyo"),
    (" big", " bigger", " small", " smaller"),
]

for m in models:
    print(f"\n  {m['name']}:")
    hits = 0
    total = 0
    for a, b, c, expected in analogy_tokens:
        result = test_analogy(m['wte'], a, b, c, expected, m['name'])
        if result is None:
            print(f"    {a}:{b}::{c}:? → SKIPPED (token issue)")
            continue
        results, hit = result
        total += 1
        if hit:
            hits += 1
        top1 = results[0]
        marker = "✓" if hit else "✗"
        print(f"    {a}:{b}::{c}:? → {repr(top1[0])} ({top1[1]:.3f}) [{marker}] expected:{repr(expected)}")
    if total > 0:
        print(f"    Score: {hits}/{total}")

print()

# ── 6. Positional Embedding Comparison (GPT-2 vs GPT-Neo) ───
print("=" * 60)
print("6. POSITIONAL EMBEDDINGS: GPT-2 vs GPT-Neo-125M")
print("=" * 60)

for m in [gpt2, neo]:
    wpe = m['wpe'].float()
    norms = torch.norm(wpe, dim=1)
    
    print(f"\n  {m['name']} (max_pos={m['max_pos']}):")
    print(f"    Norm range: [{norms.min().item():.3f}, {norms.max().item():.3f}]")
    print(f"    Norm mean:  {norms.mean().item():.3f} ± {norms.std().item():.3f}")
    
    # Key positions
    key_pos = [0, 1, 10, 100, 512]
    if m['max_pos'] > 1024:
        key_pos.extend([1023, 1024, 1500, m['max_pos'] - 1])
    else:
        key_pos.append(m['max_pos'] - 1)
    
    print(f"    Key positions:")
    for p in key_pos:
        if p < wpe.shape[0]:
            print(f"      pos {p:5d}: norm = {norms[p].item():.3f}")
    
    # Adjacent similarity decay
    print(f"    Cosine decay:")
    for gap in [1, 10, 100, 250, 500]:
        if gap < wpe.shape[0]:
            cos = torch.nn.functional.cosine_similarity(
                wpe[:-gap], wpe[gap:], dim=1
            ).mean().item()
            print(f"      gap {gap:4d}: mean cos = {cos:.4f}")
    
    # Periodicity: FFT on highest-variance dimension
    var_per_dim = wpe.var(dim=0)
    top_dim = var_per_dim.argmax().item()
    signal = wpe[:, top_dim].numpy()
    fft = np.fft.rfft(signal)
    magnitudes = np.abs(fft)
    freqs = np.fft.rfftfreq(len(signal), d=1)
    top_freq_idx = np.argsort(magnitudes[1:])[-3:] + 1  # skip DC
    print(f"    Top FFT periods (dim {top_dim}):")
    for fi in reversed(top_freq_idx):
        if freqs[fi] > 0:
            period = 1.0 / freqs[fi]
            print(f"      period {period:.0f}, magnitude {magnitudes[fi]:.2f}")

print()

# ── 7. Token-Position Orthogonality (GPT-2 vs GPT-Neo) ──────
print("=" * 60)
print("7. TOKEN-POSITION SUBSPACE ORTHOGONALITY")
print("=" * 60)

for m in [gpt2, neo]:
    wte = m['wte'].float()
    wpe = m['wpe'].float()
    
    # Per-dim variance correlation
    wte_var = wte.var(dim=0)
    wpe_var = wpe.var(dim=0)
    
    # Pearson correlation
    wte_c = wte_var - wte_var.mean()
    wpe_c = wpe_var - wpe_var.mean()
    corr = (wte_c * wpe_c).sum() / (torch.norm(wte_c) * torch.norm(wpe_c))
    
    # Top-10 dim overlap
    wte_top10 = set(wte_var.topk(10).indices.tolist())
    wpe_top10 = set(wpe_var.topk(10).indices.tolist())
    overlap = len(wte_top10 & wpe_top10)
    
    print(f"\n  {m['name']}:")
    print(f"    Variance correlation:  ρ = {corr.item():.3f}")
    print(f"    Top-10 dim overlap:    {overlap}/10")

print()

# ── 8. Pythia: Input vs Output Embedding Comparison ──────────
print("=" * 60)
print("8. PYTHIA: INPUT vs OUTPUT EMBEDDINGS (untied)")
print("=" * 60)

wte_in = pythia['wte'].float()
wte_out = pythia['wte_out'].float()

# Overall similarity
cos_io = torch.nn.functional.cosine_similarity(wte_in, wte_out, dim=1)
print(f"  Per-token input/output cosine:")
print(f"    Mean:  {cos_io.mean().item():.4f}")
print(f"    Std:   {cos_io.std().item():.4f}")
print(f"    Min:   {cos_io.min().item():.4f}")
print(f"    Max:   {cos_io.max().item():.4f}")

# Do high-norm input tokens also have high-norm output tokens?
in_norms = torch.norm(wte_in, dim=1)
out_norms = torch.norm(wte_out, dim=1)
norm_corr_c = torch.corrcoef(torch.stack([in_norms, out_norms]))[0, 1]
print(f"\n  Norm correlation (in vs out): {norm_corr_c.item():.3f}")

# Anisotropy of output embeddings
np.random.seed(42)
n = wte_out.shape[0]
idx = np.random.choice(n, size=(5000, 2))
cos_out = torch.nn.functional.cosine_similarity(
    wte_out[idx[:, 0]], wte_out[idx[:, 1]], dim=1
).mean().item()
print(f"  Output embedding anisotropy: {cos_out:.4f}")
print(f"  Input embedding anisotropy:  {pythia['anisotropy']:.4f}")

print()

# ── 9. Summary Table ─────────────────────────────────────────
print("=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Metric':<30} {'GPT-2':>12} {'GPT-Neo':>12} {'Pythia':>12}")
print("-" * 68)
print(f"{'Hidden dim':<30} {gpt2['hidden']:>12} {neo['hidden']:>12} {pythia['hidden']:>12}")
print(f"{'Vocab size':<30} {gpt2['vocab']:>12} {neo['vocab']:>12} {pythia['vocab']:>12}")
print(f"{'Position type':<30} {'learned':>12} {'learned':>12} {'RoPE':>12}")
print(f"{'Tied weights':<30} {'yes':>12} {'yes':>12} {'no':>12}")
print(f"{'Anisotropy (mean cos)':<30} {gpt2['anisotropy']:>12.4f} {neo['anisotropy']:>12.4f} {pythia['anisotropy']:>12.4f}")
print(f"{'Anisotropy (no PC1)':<30} {gpt2['anisotropy_nopc1']:>12.4f} {neo['anisotropy_nopc1']:>12.4f} {pythia['anisotropy_nopc1']:>12.4f}")
print(f"{'PC1 variance %':<30} {gpt2['pc1_pct']:>11.1f}% {neo['pc1_pct']:>11.1f}% {pythia['pc1_pct']:>11.1f}%")
print(f"{'Participation ratio':<30} {gpt2['participation_ratio']:>12.0f} {neo['participation_ratio']:>12.0f} {pythia['participation_ratio']:>12.0f}")
print(f"{'Norm mean':<30} {gpt2['norm_mean']:>12.3f} {neo['norm_mean']:>12.3f} {pythia['norm_mean']:>12.3f}")
print(f"{'Ghost cluster ratio':<30} {gpt2['ghost_ratio']:>12.3f} {neo['ghost_ratio']:>12.3f} {pythia['ghost_ratio']:>12.3f}")

print("\n\nDone! Save results and write FINDINGS_PHASE4.md")
