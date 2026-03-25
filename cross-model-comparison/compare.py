"""
Cross-Model Comparison Engine

Reads Phase 1 + Phase 2 results from multiple models and computes:
  1. Isotropy radar profile
  2. Ghost-cluster summary
  3. Analogy scorecard from curated exact probes
  4. Neighborhood agreement inside a curated concept inventory

Prerequisites: Run explore.py and deep_dive.py for each model first.

Usage: poetry run python cross-model-comparison/compare.py [--models MODEL1 MODEL2 ...] [--all]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.models import MODEL_REGISTRY, load_model
from common.probes import (
    COMPARISON_ANALOGIES,
    COMPARISON_NEIGHBOR_PROBES,
    CONCEPTS,
    token_for_concept,
    validate_probe_pack,
)

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="Cross-model embedding comparison")
parser.add_argument('--models', nargs='+', help="Model slugs to compare")
parser.add_argument('--all', action='store_true', help="Compare all models with Phase 1 results")
args = parser.parse_args()


def find_available_models():
    """Find models that have Phase 1 results."""
    available = []
    for slug in MODEL_REGISTRY:
        p1 = ROOT / 'phase1-norms-and-structure' / 'results' / slug / 'results.json'
        if p1.exists():
            available.append(slug)
    return available


def load_phase1_results(slug):
    path = ROOT / 'phase1-norms-and-structure' / 'results' / slug / 'results.json'
    with open(path) as f:
        return json.load(f)


def load_phase2_results(slug):
    path = ROOT / 'phase2-ghost-cluster-and-analogies' / 'results' / slug / 'results.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


if args.all:
    model_slugs = find_available_models()
elif args.models:
    model_slugs = args.models
else:
    model_slugs = find_available_models()

if len(model_slugs) < 2:
    print(f"Need at least 2 models with Phase 1 results. Found: {model_slugs}")
    print("Run explore.py --model <slug> for more models first.")
    sys.exit(1)

print(f"Comparing {len(model_slugs)} models: {', '.join(model_slugs)}")

p1_results = {slug: load_phase1_results(slug) for slug in model_slugs}
p2_results = {slug: load_phase2_results(slug) for slug in model_slugs}

# ============================================================
# 1. ISOTROPY RADAR PROFILE
# ============================================================
print("\n" + "=" * 60)
print("1. ISOTROPY RADAR PROFILE")
print("=" * 60)

isotropy = {}
for slug in model_slugs:
    result = p1_results[slug]
    model_info = result.get('model', {})
    hidden_dim = model_info.get('hidden_dim', result['shape'][1])

    isotropy[slug] = {
        'name': model_info.get('name', slug),
        'hidden_dim': hidden_dim,
        'vocab_size': model_info.get('vocab_size', result['shape'][0]),
        'anisotropy': result['anisotropy']['mean_cosine'],
        'participation_ratio': result['pca']['participation_ratio'],
        'participation_ratio_normalized': result['pca']['participation_ratio'] / hidden_dim,
        'entropy_effective_dims': result['pca']['entropy_effective_dims'],
        'entropy_effective_dims_normalized': result['pca']['entropy_effective_dims'] / hidden_dim,
        'mean_norm': result['norms']['mean'],
        'norm_std': result['norms']['std'],
        'norm_range': result['norms']['max'] - result['norms']['min'],
    }
    item = isotropy[slug]
    print(
        f"  {item['name']:20s}  aniso={item['anisotropy']:.3f}  "
        f"PR={item['participation_ratio']:.0f}/{hidden_dim}  "
        f"eff_dim={item['entropy_effective_dims']:.0f}/{hidden_dim}  "
        f"mean_norm={item['mean_norm']:.3f}"
    )

# ============================================================
# 2. GHOST CLUSTER SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("2. GHOST CLUSTER SUMMARY")
print("=" * 60)

ghost_data = {}
for slug in model_slugs:
    result = p2_results.get(slug)
    if result and 'ghost_cluster' in result:
        ghost = result['ghost_cluster']
        ghost_data[slug] = {
            'name': result.get('model', {}).get('name', slug),
            'count': ghost['count'],
            'mean_cosine': ghost['pairwise_cosine']['mean'],
            'min_cosine': ghost['pairwise_cosine']['min'],
            'max_cosine': ghost['pairwise_cosine']['max'],
            'diameter': ghost.get('diameter'),
            'mean_norm': ghost.get('mean_norm'),
            'newline_in_ghost': ghost.get('newline_in_ghost', False),
        }
        item = ghost_data[slug]
        cos_str = f"{item['mean_cosine']:.3f}" if item['mean_cosine'] is not None else "N/A"
        min_str = f"{item['min_cosine']:.3f}" if item['min_cosine'] is not None else "N/A"
        print(
            f"  {item['name']:20s}  {item['count']} tokens, mean cosine={cos_str}, "
            f"min cosine={min_str}, newline_in_ghost={item['newline_in_ghost']}"
        )
    else:
        print(f"  {slug:20s}  (no Phase 2 results)")

# ============================================================
# 3-4 require loading exact probe embeddings
# ============================================================
print("\n" + "=" * 60)
print("Loading models for curated exact-probe comparisons...")
print("=" * 60)

model_data = {slug: load_model(slug) for slug in model_slugs}
probe_indices = validate_probe_pack(model_data)
inventory_ids = list(CONCEPTS.keys())
print(f"\nCurated concept inventory: {len(inventory_ids)} exact concepts")

# ============================================================
# 3. ANALOGY SCORECARD
# ============================================================
print("\n" + "=" * 60)
print("3. ANALOGY SCORECARD")
print("=" * 60)

scorecard = {}
for slug in model_slugs:
    model = model_data[slug]
    resolved = probe_indices[slug]
    model_scores = {}

    for label, a_concept, b_concept, c_concept, expected_concept in COMPARISON_ANALOGIES:
        ia = resolved[a_concept]
        ib = resolved[b_concept]
        ic = resolved[c_concept]
        expected_idx = resolved[expected_concept]

        vec = model.emb[ib] - model.emb[ia] + model.emb[ic]
        vec_norm = np.linalg.norm(vec)
        cos = (model.emb @ vec) / (model.norms * vec_norm + 1e-10)
        for idx in (ia, ib, ic):
            cos[idx] = -2

        top_idx = np.argsort(cos)[-5:][::-1]
        top1_idx = int(top_idx[0])
        top1_token = model.tokens[top1_idx]
        top1_cos = float(cos[top1_idx])

        expected_rank = None
        rank_list = list(top_idx)
        if expected_idx in rank_list:
            expected_rank = rank_list.index(expected_idx) + 1
        else:
            all_sorted = np.argsort(cos)[::-1]
            pos = np.where(all_sorted == expected_idx)[0]
            expected_rank = int(pos[0]) + 1 if len(pos) > 0 else None

        success = top1_idx == expected_idx
        model_scores[label] = {
            'top1': top1_token,
            'cosine': round(top1_cos, 4),
            'expected': token_for_concept(slug, expected_concept),
            'expected_rank': expected_rank,
            'success': success,
            'skipped': False,
        }

    scorecard[slug] = model_scores
    successes = sum(1 for value in model_scores.values() if value['success'])
    total = len(COMPARISON_ANALOGIES)
    print(f"  {model.name:20s}  {successes}/{total} correct")

print("\n  Detailed results:")
header = f"  {'Analogy':20s}"
for slug in model_slugs:
    header += f"  {model_data[slug].name:>15s}"
print(header)
print("  " + "-" * len(header))

for label, *_rest in COMPARISON_ANALOGIES:
    row = f"  {label:20s}"
    for slug in model_slugs:
        result = scorecard[slug][label]
        if result['success']:
            row += f"  {result['top1']:>15s}"
        else:
            row += f"  {('*' + (result['top1'] or '?')):>15s}"
    print(row)
print("  (* = wrong answer)")

# ============================================================
# 4. NEIGHBORHOOD AGREEMENT
# ============================================================
print("\n" + "=" * 60)
print("4. NEIGHBORHOOD AGREEMENT")
print("=" * 60)

k_neighbors = 5
concept_neighbors = {}

for slug in model_slugs:
    model = model_data[slug]
    resolved = probe_indices[slug]
    inventory_idx = np.array([resolved[concept_id] for concept_id in inventory_ids], dtype=int)
    inventory_vectors = model.normed_emb[inventory_idx]
    inventory_cos = inventory_vectors @ inventory_vectors.T
    concept_neighbors[slug] = {}

    for probe_concept in COMPARISON_NEIGHBOR_PROBES:
        probe_pos = inventory_ids.index(probe_concept)
        cos = inventory_cos[probe_pos].copy()
        cos[probe_pos] = -2
        neighbor_pos = np.argsort(cos)[-k_neighbors:][::-1]
        concept_neighbors[slug][probe_concept] = [inventory_ids[pos] for pos in neighbor_pos]

print(f"  Using {len(COMPARISON_NEIGHBOR_PROBES)} probes inside a {len(inventory_ids)}-concept inventory")

jaccard_matrix = {}
for i, slug_a in enumerate(model_slugs):
    for slug_b in model_slugs[i + 1:]:
        jaccards = []
        per_probe = {}
        for probe_concept in COMPARISON_NEIGHBOR_PROBES:
            neigh_a = set(concept_neighbors[slug_a][probe_concept])
            neigh_b = set(concept_neighbors[slug_b][probe_concept])
            score = len(neigh_a & neigh_b) / len(neigh_a | neigh_b) if neigh_a | neigh_b else 0.0
            jaccards.append(score)
            per_probe[probe_concept] = round(score, 4)

        mean_score = float(np.mean(jaccards))
        pair_key = f"{slug_a}_vs_{slug_b}"
        jaccard_matrix[pair_key] = {
            'model_a': slug_a,
            'model_b': slug_b,
            'mean_jaccard': mean_score,
            'k_neighbors': k_neighbors,
            'per_probe': per_probe,
        }
        print(f"  {model_data[slug_a].name:15s} vs {model_data[slug_b].name:15s}  mean Jaccard = {mean_score:.3f}")

# ============================================================
# SAVE ALL RESULTS
# ============================================================

comparison = {
    'models': model_slugs,
    'concept_inventory_size': len(inventory_ids),
    'neighbor_probe_count': len(COMPARISON_NEIGHBOR_PROBES),
    'neighbor_k': k_neighbors,
    'isotropy': isotropy,
    'ghost_universality': ghost_data,
    'analogy_scorecard': scorecard,
    'neighborhood_jaccard': jaccard_matrix,
}

with open(OUT / 'comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"\nResults saved to {OUT}/comparison.json")
print("Run dashboard.py to generate the visualization.")
