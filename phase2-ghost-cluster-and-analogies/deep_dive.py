"""
Phase 2: Ghost Cluster, Analogies, and Nearest Neighbors
Supports any model in the registry (default: GPT-2).

Deeper probing into specific phenomena found in Phase 1:
  1. Ghost cluster — tokens that collapsed to nearly identical vectors
  2. Newline — the exception that broke free (if applicable)
  3. Nearest neighbors — the model's semantic taxonomy
  4. Embedding analogies — linear relationships without fine-tuning
  5. Outlier tokens — training data archaeology

Usage: poetry run python phase2-ghost-cluster-and-analogies/deep_dive.py [--model MODEL]
"""

import argparse
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.models import load_model, add_model_arg, resolve_token

parser = argparse.ArgumentParser(description="Phase 2: Ghost clusters, analogies, and neighbors")
add_model_arg(parser)
args = parser.parse_args()

m = load_model(args.model)
emb, tok, tokens, labels, norms, normed_emb = m.emb, m.tokenizer, m.tokens, m.labels, m.norms, m.normed_emb

print(f"Embedding matrix: {m.vocab_size} tokens x {m.hidden_dim} dimensions")

# Output directory
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', m.slug)
os.makedirs(OUT, exist_ok=True)


def nearest_neighbors(idx, k=15):
    """Return top-k nearest neighbors by cosine similarity (excluding self)."""
    cos = normed_emb @ normed_emb[idx]
    cos[idx] = -2  # exclude self
    nn = np.argsort(cos)[-k:][::-1]
    return [(int(i), float(cos[i]), float(norms[i])) for i in nn]


def analogy(a_str, b_str, c_str, k=10):
    """a is to b as c is to ? Returns top-k results."""
    ia = resolve_token(m, a_str)
    ib = resolve_token(m, b_str)
    ic = resolve_token(m, c_str)
    if None in (ia, ib, ic):
        missing = [s for s, i in [(a_str, ia), (b_str, ib), (c_str, ic)] if i is None]
        print(f"    (skipping: tokens not found: {missing})")
        return [], (ia, ib, ic)

    vec = emb[ib] - emb[ia] + emb[ic]
    vec_norm = np.linalg.norm(vec)
    cos = (emb @ vec) / (norms * vec_norm + 1e-10)
    for idx in (ia, ib, ic):
        cos[idx] = -2
    nn = np.argsort(cos)[-k:][::-1]
    return [(int(i), float(cos[i])) for i in nn], (ia, ib, ic)


def find_ghost_cluster(threshold=0.95, percentile=5):
    """Dynamically find ghost cluster: low-norm tokens with high mutual cosine.

    Strategy:
    1. Take the bottom percentile of tokens by norm
    2. Compute pairwise cosine among them
    3. Find connected components where cosine > threshold
    4. Return the largest tight cluster
    """
    cutoff = np.percentile(norms, percentile)
    candidates = np.where(norms <= cutoff)[0]

    if len(candidates) < 2:
        return [], np.array([])

    cand_normed = normed_emb[candidates]
    cos_matrix = cand_normed @ cand_normed.T

    # Build adjacency and find connected components via BFS
    adj = cos_matrix > threshold
    np.fill_diagonal(adj, False)
    visited = set()
    components = []

    for i in range(len(candidates)):
        if i in visited:
            continue
        # BFS
        component = []
        queue = [i]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            neighbors = np.where(adj[node])[0]
            for n in neighbors:
                if n not in visited:
                    queue.append(n)
        components.append(component)

    if not components:
        return [], np.array([])

    # Return largest component
    largest = max(components, key=len)
    ghost_indices = candidates[largest].tolist()
    ghost_indices.sort()

    # Compute the cosine matrix for just the ghost cluster
    ghost_normed = normed_emb[ghost_indices]
    ghost_cos = ghost_normed @ ghost_normed.T

    return ghost_indices, ghost_cos


# ============================================================
# 1. GHOST CLUSTER (dynamic detection)
# ============================================================
print("\n" + "="*60)
print("1. GHOST CLUSTER — tokens that collapsed to similar vectors")
print("="*60)

ghost_idx, ghost_cos_matrix = find_ghost_cluster()

if len(ghost_idx) >= 2:
    ghost_norms = norms[ghost_idx]
    triu = np.triu_indices(len(ghost_idx), k=1)
    cos_all = ghost_cos_matrix[triu]

    # Find newline token dynamically
    nl_tokens = tok.encode("\n")
    nl_idx = nl_tokens[0] if len(nl_tokens) == 1 else None
    nl_in_ghost = nl_idx is not None and nl_idx in ghost_idx

    if nl_in_ghost:
        ghost_no_nl = [i for i in ghost_idx if i != nl_idx]
        ghost_no_nl_normed = normed_emb[ghost_no_nl]
        cos_no_nl = ghost_no_nl_normed @ ghost_no_nl_normed.T
        triu_no_nl = np.triu_indices(len(ghost_no_nl), k=1)
        cos_excl_nl = cos_no_nl[triu_no_nl]

        nl_pos = ghost_idx.index(nl_idx)
        nl_cos_to_ghosts = float(ghost_cos_matrix[nl_pos, :].mean())

        print(f"  {len(ghost_idx)} tokens in ghost cluster, pairwise cosine {cos_excl_nl.mean():.3f} (excluding newline)")
        print(f"  Newline (idx {nl_idx}) broke free: cosine {nl_cos_to_ghosts:.3f} to siblings, norm {norms[nl_idx]:.3f}")
    else:
        print(f"  {len(ghost_idx)} tokens in ghost cluster, pairwise cosine {cos_all.mean():.3f}")
        if nl_idx is not None:
            print(f"  Newline (idx {nl_idx}) is NOT in the ghost cluster for this model")

    print(f"  Token index range: {ghost_idx[0]}-{ghost_idx[-1]}")
    print(f"  Mean norm: {ghost_norms.mean():.3f} (vs global mean {norms.mean():.3f})")

    # Show a few example ghost tokens
    examples = ghost_idx[:5]
    print(f"  Examples: {', '.join(repr(labels[i]) for i in examples)}")
else:
    print(f"  No ghost cluster found (fewer than 2 tightly-clustered low-norm tokens)")
    nl_idx = None
    nl_in_ghost = False

# ============================================================
# 2. NEWLINE ANALYSIS
# ============================================================
print("\n" + "="*60)
print("2. NEWLINE — a control character that earned its identity")
print("="*60)

nl_tokens = tok.encode("\n")
nl_idx = nl_tokens[0] if len(nl_tokens) == 1 else None

nl_neighbors = []
if nl_idx is not None:
    nl_neighbors = nearest_neighbors(nl_idx, k=10)
    top3 = nl_neighbors[:3]
    print(f"  Newline token idx: {nl_idx}, norm: {norms[nl_idx]:.3f}")
    print(f"  Nearest neighbors: {repr(labels[top3[0][0]])}, {repr(labels[top3[1][0]])}, {repr(labels[top3[2][0]])}")
else:
    print(f"  Newline is not a single token in this tokenizer (encoded as {nl_tokens})")

# ============================================================
# 3. NEAREST NEIGHBORS — semantic taxonomy
# ============================================================
print("\n" + "="*60)
print("3. NEAREST NEIGHBORS — the model's semantic taxonomy")
print("="*60)

probe_words = [
    ('the', 'determiners/pronouns'),
    ('at', 'prepositions'),
    ('king', 'royalty'),
    ('queen', 'royalty (gendered)'),
    ('dog', 'animals'),
    ('Python', 'programming languages'),
]

nn_results = {}
for word, expected in probe_words:
    idx = resolve_token(m, word)
    if idx is None:
        print(f"  {word:12s} -> (not found as single token)")
        continue
    neighbors = nearest_neighbors(idx, k=15)
    nn_results[tokens[idx]] = neighbors
    top3 = neighbors[:3]
    print(f"  {repr(tokens[idx]):12s} -> {repr(labels[top3[0][0]])}, {repr(labels[top3[1][0]])}, {repr(labels[top3[2][0]])} ({expected})")

print(f"  -> Clean semantic clustering before any attention or context.")

# ============================================================
# 4. EMBEDDING ANALOGIES — linear relationships
# ============================================================
print("\n" + "="*60)
print("4. EMBEDDING ANALOGIES — linear relationships in raw embeddings")
print("="*60)

analogy_tests = [
    ('king', 'queen', 'man', 'gender'),
    ('dog', 'dogs', 'cat', 'pluralization'),
    ('France', 'Paris', 'Japan', 'capital city'),
    ('big', 'bigger', 'small', 'comparative'),
]

analogy_results = {}
for a, b, c, label in analogy_tests:
    result_list, (ia, ib, ic) = analogy(a, b, c, k=5)
    key = f"{a}:{b}::{c}:?"
    analogy_results[key] = result_list
    if result_list:
        top = result_list[0]
        print(f"  {key} -> {repr(labels[top[0]])} (cos={top[1]:.3f}) [{label}]")

# ============================================================
# 5. OUTLIER TOKENS — model-agnostic weird token detection
# ============================================================
print("\n" + "="*60)
print("5. OUTLIER TOKENS — training data archaeology")
print("="*60)

# Find outliers: top 0.1% by norm
outlier_threshold = np.percentile(norms, 99.9)
outlier_mask = norms >= outlier_threshold
outlier_indices = np.where(outlier_mask)[0]

# Also find isolated tokens (low max-neighbor cosine)
rng = np.random.RandomState(42)
n_check = min(2000, m.vocab_size)
check_idx = rng.choice(m.vocab_size, n_check, replace=False)
max_neighbor_cos = np.zeros(n_check)
for j, idx in enumerate(check_idx):
    cos = normed_emb @ normed_emb[idx]
    cos[idx] = -2
    max_neighbor_cos[j] = cos.max()

isolated_threshold = np.percentile(max_neighbor_cos, 1)
isolated_sample_idx = check_idx[max_neighbor_cos <= isolated_threshold]

outlier_info = []
for i in outlier_indices:
    outlier_info.append({'idx': int(i), 'token': repr(labels[i]), 'norm': float(norms[i]), 'type': 'high_norm'})
for i in isolated_sample_idx:
    if i not in outlier_indices:
        outlier_info.append({'idx': int(i), 'token': repr(labels[i]), 'norm': float(norms[i]), 'type': 'isolated'})

# Pairwise similarity among all outliers
all_outlier_idx = list(set(list(outlier_indices) + list(isolated_sample_idx)))
weird_mean_cos = 0.0
global_mean_cos = 0.0
if len(all_outlier_idx) > 1:
    o_normed = normed_emb[all_outlier_idx]
    cos_o = o_normed @ o_normed.T
    triu_o = np.triu_indices(len(all_outlier_idx), k=1)
    weird_mean_cos = float(cos_o[triu_o].mean())

    sample = rng.choice(m.vocab_size, 5000, replace=False)
    sample_normed = normed_emb[sample]
    global_cos = sample_normed @ sample_normed.T
    global_triu = np.triu_indices(5000, k=1)
    global_mean_cos = float(global_cos[global_triu].mean())

    print(f"  {len(outlier_indices)} high-norm outliers (norm >= {outlier_threshold:.2f})")
    print(f"  {len(isolated_sample_idx)} isolated tokens (low nearest-neighbor cosine, from {n_check} sampled)")
    print(f"  Outlier pairwise cosine: {weird_mean_cos:.3f} (vs global mean {global_mean_cos:.3f})")
    for item in outlier_info[:10]:
        print(f"    {item['token']:30s} norm={item['norm']:.3f} ({item['type']})")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    'model': {
        'name': m.name,
        'slug': m.slug,
        'hf_id': m.hf_id,
        'vocab_size': m.vocab_size,
        'hidden_dim': m.hidden_dim,
    },
    'ghost_cluster': {
        'count': len(ghost_idx),
        'indices': ghost_idx,
        'pairwise_cosine': {
            'mean': float(ghost_cos_matrix[np.triu_indices(len(ghost_idx), k=1)].mean()) if len(ghost_idx) >= 2 else None,
            'min': float(ghost_cos_matrix[np.triu_indices(len(ghost_idx), k=1)].min()) if len(ghost_idx) >= 2 else None,
            'max': float(ghost_cos_matrix[np.triu_indices(len(ghost_idx), k=1)].max()) if len(ghost_idx) >= 2 else None,
        },
        'mean_norm': float(norms[ghost_idx].mean()) if ghost_idx else None,
        'newline_idx': nl_idx,
        'newline_in_ghost': nl_in_ghost,
    },
    'newline_neighbors': [
        {'idx': i, 'cosine': c, 'norm': n, 'token': repr(labels[i])}
        for i, c, n in nl_neighbors
    ] if nl_neighbors else [],
    'nearest_neighbors': {
        token_str: [
            {'idx': i, 'cosine': c, 'norm': n, 'token': repr(labels[i])}
            for i, c, n in neighbors
        ]
        for token_str, neighbors in nn_results.items()
    },
    'analogies': {
        key: [{'idx': i, 'cosine': c, 'token': repr(labels[i])} for i, c in result_list]
        for key, result_list in analogy_results.items()
    },
    'outlier_tokens': {
        'tokens': outlier_info,
        'pairwise_cosine': weird_mean_cos,
        'global_mean_cosine': global_mean_cos,
    },
}

with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save ghost cluster cosine matrix with reference tokens for heatmap
ref_words = ['the', 'dog', 'Python']
ref_tokens_map = {}
for word in ref_words:
    idx = resolve_token(m, word)
    if idx is not None:
        ref_tokens_map[tokens[idx]] = idx

# Also try to add a double-newline if it exists
nn_token = tok.encode("\n\n")
if len(nn_token) == 1:
    ref_tokens_map['\n\n'] = nn_token[0]

if ghost_idx:
    expanded_idx = ghost_idx + list(ref_tokens_map.values())
    expanded_embs = emb[expanded_idx]
    expanded_norms_arr = norms[expanded_idx]
    expanded_normed = expanded_embs / (expanded_norms_arr[:, None] + 1e-10)
    expanded_cos = expanded_normed @ expanded_normed.T
    np.save(os.path.join(OUT, 'ghost_cosine_matrix.npy'), expanded_cos)
    expanded_labels = [labels[i] for i in ghost_idx] + list(ref_tokens_map.keys())
    with open(os.path.join(OUT, 'ghost_labels.json'), 'w') as f:
        json.dump({'labels': expanded_labels, 'n_ghost': len(ghost_idx)}, f)

# Save analogy vectors for PCA projection
analogy_data = {}
for a, b, c, label in analogy_tests:
    ia = resolve_token(m, a)
    ib = resolve_token(m, b)
    ic = resolve_token(m, c)
    key = f"{a}:{b}::{c}:?"
    top_result = analogy_results.get(key, [])
    id_idx = top_result[0][0] if top_result else None
    if None not in (ia, ib, ic, id_idx):
        analogy_data[label] = {
            'indices': [int(ia), int(ib), int(ic), int(id_idx)],
            'labels': [labels[ia], labels[ib], labels[ic], labels[id_idx]],
            'vectors': [emb[ia].tolist(), emb[ib].tolist(), emb[ic].tolist(), emb[id_idx].tolist()],
        }
with open(os.path.join(OUT, 'analogy_vectors.json'), 'w') as f:
    json.dump(analogy_data, f)

# Save nearest neighbor data for visualization
nn_viz_data = {}
for token_str, neighbors in nn_results.items():
    idx = resolve_token(m, token_str.strip())
    if idx is not None:
        nn_viz_data[token_str] = {
            'center': {'idx': int(idx), 'label': labels[idx]},
            'neighbors': [{'idx': i, 'cosine': c, 'label': labels[i]} for i, c, _ in neighbors[:5]],
        }
with open(os.path.join(OUT, 'nn_viz_data.json'), 'w') as f:
    json.dump(nn_viz_data, f)

print(f"\nResults saved to {OUT}/")
print("Run charts.py for visualizations.")
