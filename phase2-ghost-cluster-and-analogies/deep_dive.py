"""
Phase 2: Ghost Cluster, Analogies, and Nearest Neighbors
GPT-2 (50,257 tokens x 768 dimensions)

Deeper probing into specific phenomena found in Phase 1:
  1. Ghost cluster — control characters that collapsed to one vector
  2. Newline — the exception that broke free
  3. Nearest neighbors — the model's semantic taxonomy
  4. Embedding analogies — linear relationships without fine-tuning
  5. Weird tokens — training data archaeology
"""

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import json
import os
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(OUT, '..', 'phase1-norms-and-structure'))
from tokenutils import token_display

print("Loading GPT-2 embeddings...")
path = hf_hub_download('gpt2', 'pytorch_model.bin')
sd = torch.load(path, map_location='cpu', weights_only=False)
emb = sd['wte.weight'].numpy()
tok = AutoTokenizer.from_pretrained('gpt2')
tokens = [tok.decode([i]) for i in range(emb.shape[0])]
labels = [token_display(tok, i) for i in range(emb.shape[0])]
norms = np.linalg.norm(emb, axis=1)
normed_emb = emb / (norms[:, None] + 1e-10)

print(f"Embedding matrix: {emb.shape[0]} tokens x {emb.shape[1]} dimensions")


def nearest_neighbors(idx, k=15):
    """Return top-k nearest neighbors by cosine similarity (excluding self)."""
    cos = normed_emb @ normed_emb[idx]
    cos[idx] = -2  # exclude self
    nn = np.argsort(cos)[-k:][::-1]
    return [(int(i), float(cos[i]), float(norms[i])) for i in nn]


def analogy(a, b, c, k=10):
    """a is to b as c is to ? Returns top-k results."""
    def get_idx(s):
        matches = [i for i, t in enumerate(tokens) if t == s]
        return matches[0] if matches else None

    ia, ib, ic = get_idx(a), get_idx(b), get_idx(c)
    if None in (ia, ib, ic):
        return []

    vec = emb[ib] - emb[ia] + emb[ic]
    vec_norm = np.linalg.norm(vec)
    cos = (emb @ vec) / (norms * vec_norm + 1e-10)
    for idx in (ia, ib, ic):
        cos[idx] = -2
    nn = np.argsort(cos)[-k:][::-1]
    return [(int(i), float(cos[i])) for i in nn]


# ============================================================
# 1. THE GHOST CLUSTER
# ============================================================
print("\n" + "="*60)
print("1. THE GHOST CLUSTER — control characters collapsed to one vector")
print("="*60)

ghost_idx = list(range(188, 222))
ghost_embs = emb[ghost_idx]
ghost_norms = norms[ghost_idx]

ghost_normed = ghost_embs / ghost_norms[:, None]
cos_matrix = ghost_normed @ ghost_normed.T
triu = np.triu_indices(len(ghost_idx), k=1)
cos_all = cos_matrix[triu]

# Without newline
ghost_no_nl = [i for i in ghost_idx if i != 198]
ghost_no_nl_normed = emb[ghost_no_nl] / norms[ghost_no_nl][:, None]
cos_no_nl = ghost_no_nl_normed @ ghost_no_nl_normed.T
triu_no_nl = np.triu_indices(len(ghost_no_nl), k=1)
cos_excl_nl = cos_no_nl[triu_no_nl]

nl_cos_to_ghosts = cos_matrix[198 - 188, :].mean()

print(f"  {len(ghost_idx)} tokens (188-221), pairwise cosine {cos_excl_nl.mean():.3f} (excluding newline)")
print(f"  Newline (198) broke free: cosine {nl_cos_to_ghosts:.3f} to siblings, norm {norms[198]:.3f}")
print(f"  -> No training signal = no differentiation. Newline is the exception because it's meaningful.")

# ============================================================
# 2. NEWLINE — the special character
# ============================================================
print("\n" + "="*60)
print("2. NEWLINE — a control character that earned its identity")
print("="*60)

nl_neighbors = nearest_neighbors(198, k=10)
top3 = nl_neighbors[:3]
print(f"  Nearest neighbors: {repr(labels[top3[0][0]])}, {repr(labels[top3[1][0]])}, {repr(labels[top3[2][0]])}")
print(f"  -> Newline's neighbors are sentence/paragraph starters: 'The', endoftext, '('")

nl_vs_space = float(normed_emb[198] @ normed_emb[220])
print(f"  Newline vs space: cosine {nl_vs_space:.3f} (different roles despite both being whitespace)")

# ============================================================
# 3. NEAREST NEIGHBORS — semantic taxonomy
# ============================================================
print("\n" + "="*60)
print("3. NEAREST NEIGHBORS — the model's semantic taxonomy")
print("="*60)

probe_tokens = {
    ' the': 'determiners/pronouns',
    ' at': 'prepositions',
    ' king': 'royalty',
    ' queen': 'royalty (gendered)',
    ' dog': 'animals',
    ' Python': 'programming languages',
}

nn_results = {}
for token_str, expected in probe_tokens.items():
    matches = [i for i, t in enumerate(tokens) if t == token_str]
    if not matches:
        continue
    idx = matches[0]
    neighbors = nearest_neighbors(idx, k=15)
    nn_results[token_str] = neighbors
    top3 = neighbors[:3]
    print(f"  {repr(token_str):12s} -> {repr(labels[top3[0][0]])}, {repr(labels[top3[1][0]])}, {repr(labels[top3[2][0]])} ({expected})")

print(f"  -> Clean semantic clustering before any attention or context.")

# ============================================================
# 4. EMBEDDING ANALOGIES — linear relationships without fine-tuning
# ============================================================
print("\n" + "="*60)
print("4. EMBEDDING ANALOGIES — linear relationships in raw embeddings")
print("="*60)

analogy_tests = [
    (' king', ' queen', ' man', 'gender'),
    (' dog', ' dogs', ' cat', 'pluralization'),
    (' France', ' Paris', ' Japan', 'capital city'),
    (' big', ' bigger', ' small', 'comparative'),
]

analogy_results = {}
for a, b, c, label in analogy_tests:
    results = analogy(a, b, c, k=5)
    analogy_results[f"{a}:{b}::{c}:?"] = results
    if results:
        top = results[0]
        print(f"  {a}:{b}::{c}:? -> {repr(labels[top[0]])} (cos={top[1]:.3f}) [{label}]")

print(f"  -> Analogies work in static embeddings — the embedding layer already learned linear structure.")

# ============================================================
# 5. WEIRD TOKENS — training data archaeology
# ============================================================
print("\n" + "="*60)
print("5. WEIRD TOKENS — e-commerce ghosts and glitch tokens")
print("="*60)

weird_patterns = ['externalToEVA', 'quickShip', 'TheNitrome', 'BuyableInstoreAndOnline',
                  'soDeliveryDate', 'inventoryQuantity', 'isSpecialOrderable',
                  'DragonMagazine', 'natureconservancy', 'SPONSORED']

weird_idx = []
weird_info = []
for pat in weird_patterns:
    matches = [i for i, t in enumerate(tokens) if pat in t]
    for i in matches:
        weird_idx.append(i)
        weird_info.append({'idx': int(i), 'token': repr(labels[i]), 'norm': float(norms[i])})

# Pairwise similarity among weird tokens
unique_weird = list(set(weird_idx))
if len(unique_weird) > 1:
    weird_normed = normed_emb[unique_weird]
    cos_w = weird_normed @ weird_normed.T
    triu_w = np.triu_indices(len(unique_weird), k=1)
    weird_mean_cos = float(cos_w[triu_w].mean())

    # Global mean for comparison
    rng = np.random.RandomState(42)
    sample = rng.choice(emb.shape[0], 5000, replace=False)
    sample_normed = normed_emb[sample]
    global_cos = sample_normed @ sample_normed.T
    global_triu = np.triu_indices(5000, k=1)
    global_mean_cos = float(global_cos[global_triu].mean())

    print(f"  {len(weird_info)} weird tokens found (e-commerce artifacts, Reddit usernames, glitch tokens)")
    print(f"  Pairwise cosine: {weird_mean_cos:.3f} (vs global mean {global_mean_cos:.3f})")
    print(f"  -> The model groups 'things I barely saw in training' into a distinct region.")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    'ghost_cluster': {
        'token_range': '188-221',
        'count': len(ghost_idx),
        'pairwise_cosine_excl_newline': {
            'mean': float(cos_excl_nl.mean()),
            'min': float(cos_excl_nl.min()),
            'max': float(cos_excl_nl.max()),
        },
        'newline_cosine_to_ghosts': float(nl_cos_to_ghosts),
        'newline_norm': float(norms[198]),
    },
    'newline_neighbors': [
        {'idx': i, 'cosine': c, 'norm': n, 'token': repr(labels[i])}
        for i, c, n in nl_neighbors
    ],
    'nearest_neighbors': {
        token_str: [
            {'idx': i, 'cosine': c, 'norm': n, 'token': repr(labels[i])}
            for i, c, n in neighbors
        ]
        for token_str, neighbors in nn_results.items()
    },
    'analogies': {
        key: [{'idx': i, 'cosine': c, 'token': repr(labels[i])} for i, c in results]
        for key, results in analogy_results.items()
    },
    'weird_tokens': {
        'tokens': weird_info,
        'pairwise_cosine': weird_mean_cos,
        'global_mean_cosine': global_mean_cos,
    },
}

with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save data for visualizations — expanded ghost matrix with reference tokens
ref_tokens = {' the': 262, ' dog': 3290, '\n\n': 628, ' Python': 11361}
expanded_idx = ghost_idx + list(ref_tokens.values())
expanded_embs = emb[expanded_idx]
expanded_norms = norms[expanded_idx]
expanded_normed = expanded_embs / (expanded_norms[:, None] + 1e-10)
expanded_cos = expanded_normed @ expanded_normed.T
np.save(os.path.join(OUT, 'ghost_cosine_matrix.npy'), expanded_cos)
expanded_labels = [labels[i] for i in ghost_idx] + list(ref_tokens.keys())
with open(os.path.join(OUT, 'ghost_labels.json'), 'w') as f:
    json.dump(expanded_labels, f)

# Save analogy token embeddings for PCA projection
analogy_data = {}
for a, b, c, label in analogy_tests:
    def get_idx(s):
        return next((i for i, t in enumerate(tokens) if t == s), None)
    ia, ib, ic = get_idx(a), get_idx(b), get_idx(c)
    top_result = analogy_results[f"{a}:{b}::{c}:?"]
    id_idx = top_result[0][0] if top_result else None
    if None not in (ia, ib, ic, id_idx):
        analogy_data[label] = {
            'indices': [int(ia), int(ib), int(ic), int(id_idx)],
            'labels': [labels[ia], labels[ib], labels[ic], labels[id_idx]],
            'vectors': [emb[ia].tolist(), emb[ib].tolist(), emb[ic].tolist(), emb[id_idx].tolist()],
        }
with open(os.path.join(OUT, 'analogy_vectors.json'), 'w') as f:
    json.dump(analogy_data, f)

# Save nearest neighbor embeddings for radial graphs
nn_viz_data = {}
for token_str, neighbors in nn_results.items():
    matches = [i for i, t in enumerate(tokens) if t == token_str]
    if matches:
        center_idx = matches[0]
        nn_viz_data[token_str] = {
            'center': {'idx': int(center_idx), 'label': labels[center_idx]},
            'neighbors': [{'idx': i, 'cosine': c, 'label': labels[i]} for i, c, _ in neighbors[:5]],
        }
with open(os.path.join(OUT, 'nn_viz_data.json'), 'w') as f:
    json.dump(nn_viz_data, f)

print(f"\nDetailed data saved to results.json and visualization data files.")
print("Run charts.py for visualizations.")
