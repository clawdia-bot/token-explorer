"""
Cross-Model Comparison Engine

Reads Phase 1 + Phase 2 results from multiple models and computes:
  1. Isotropy radar profile (anisotropy, participation ratio, effective dims)
  2. Ghost cluster universality (do control tokens cluster in all models?)
  3. Analogy scorecard (standardized battery across all models)
  4. Neighborhood Jaccard overlap (do models agree on semantic neighbors?)
  5. Outlier migration (are weird tokens weird everywhere?)

Prerequisites: Run explore.py and deep_dive.py for each model first.

Usage: poetry run python cross-model-comparison/compare.py [--models MODEL1 MODEL2 ...] [--all]
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy import stats as sp_stats
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.models import load_model, resolve_token, MODEL_REGISTRY

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


def build_shared_vocabulary(models):
    """Find tokens that decode to the same string across all models.

    Returns: dict mapping decoded_string -> {slug: token_index}
    """
    per_model = {}
    for slug, m in models.items():
        token_map = {}
        for i, t in enumerate(m.tokens):
            stripped = t.strip()
            if stripped and stripped not in token_map:
                token_map[stripped] = i
        per_model[slug] = token_map

    # Intersection of all decoded strings
    slugs = list(models.keys())
    shared_strings = set(per_model[slugs[0]].keys())
    for slug in slugs[1:]:
        shared_strings &= set(per_model[slug].keys())

    shared = {}
    for s in shared_strings:
        shared[s] = {slug: per_model[slug][s] for slug in slugs}

    return shared


# ============================================================
# Determine which models to compare
# ============================================================
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

# Load Phase 1 results (no model loading needed for isotropy/ghost sections)
p1_results = {slug: load_phase1_results(slug) for slug in model_slugs}
p2_results = {slug: load_phase2_results(slug) for slug in model_slugs}

# ============================================================
# 1. ISOTROPY RADAR PROFILE
# ============================================================
print("\n" + "="*60)
print("1. ISOTROPY RADAR PROFILE")
print("="*60)

isotropy = {}
for slug in model_slugs:
    r = p1_results[slug]
    model_info = r.get('model', {})
    hidden_dim = model_info.get('hidden_dim', r['shape'][1])

    isotropy[slug] = {
        'name': model_info.get('name', slug),
        'hidden_dim': hidden_dim,
        'vocab_size': model_info.get('vocab_size', r['shape'][0]),
        'anisotropy': r['anisotropy']['mean_cosine'],
        'participation_ratio': r['pca']['participation_ratio'],
        'participation_ratio_normalized': r['pca']['participation_ratio'] / hidden_dim,
        'entropy_effective_dims': r['pca']['entropy_effective_dims'],
        'entropy_effective_dims_normalized': r['pca']['entropy_effective_dims'] / hidden_dim,
        'mean_norm': r['norms']['mean'],
        'norm_std': r['norms']['std'],
        'norm_range': r['norms']['max'] - r['norms']['min'],
    }
    i = isotropy[slug]
    print(f"  {i['name']:20s}  aniso={i['anisotropy']:.3f}  PR={i['participation_ratio']:.0f}/{hidden_dim}  "
          f"eff_dim={i['entropy_effective_dims']:.0f}/{hidden_dim}  mean_norm={i['mean_norm']:.3f}")

# ============================================================
# 2. GHOST CLUSTER UNIVERSALITY
# ============================================================
print("\n" + "="*60)
print("2. GHOST CLUSTER UNIVERSALITY")
print("="*60)

ghost_data = {}
for slug in model_slugs:
    r2 = p2_results.get(slug)
    if r2 and 'ghost_cluster' in r2:
        gc = r2['ghost_cluster']
        ghost_data[slug] = {
            'name': r2.get('model', {}).get('name', slug),
            'count': gc['count'],
            'mean_cosine': gc['pairwise_cosine']['mean'] if gc['pairwise_cosine']['mean'] is not None else None,
            'mean_norm': gc.get('mean_norm'),
            'newline_in_ghost': gc.get('newline_in_ghost', False),
        }
        g = ghost_data[slug]
        cos_str = f"{g['mean_cosine']:.3f}" if g['mean_cosine'] is not None else "N/A"
        print(f"  {g['name']:20s}  {g['count']} tokens, mean cosine={cos_str}, "
              f"newline_in_ghost={g['newline_in_ghost']}")
    else:
        print(f"  {slug:20s}  (no Phase 2 results)")

# ============================================================
# 3-5 require loading actual embeddings
# ============================================================
print("\n" + "="*60)
print("Loading models for embedding-level comparisons...")
print("="*60)

# Load one at a time, extract what we need, keep lightweight
model_data = {}
for slug in model_slugs:
    m = load_model(slug)
    model_data[slug] = m

shared_vocab = build_shared_vocabulary(model_data)
print(f"\nShared vocabulary: {len(shared_vocab)} tokens across {len(model_slugs)} models")

# ============================================================
# 3. ANALOGY SCORECARD
# ============================================================
print("\n" + "="*60)
print("3. ANALOGY SCORECARD")
print("="*60)

analogy_battery = [
    ('king', 'queen', 'man', 'woman', 'gender'),
    ('dog', 'dogs', 'cat', 'cats', 'plural'),
    ('France', 'Paris', 'Japan', 'Tokyo', 'capital'),
    ('big', 'bigger', 'small', 'smaller', 'comparative'),
    ('good', 'best', 'bad', 'worst', 'superlative'),
    ('walk', 'walked', 'run', 'ran', 'past_tense'),
    ('Spain', 'Spanish', 'Germany', 'German', 'demonym'),
    ('hot', 'cold', 'up', 'down', 'antonym'),
    ('man', 'woman', 'boy', 'girl', 'gender2'),
    ('eat', 'ate', 'drink', 'drank', 'past_tense2'),
    ('Italy', 'Rome', 'Germany', 'Berlin', 'capital2'),
    ('fast', 'faster', 'slow', 'slower', 'comparative2'),
]

scorecard = {}
for slug in model_slugs:
    m = model_data[slug]
    model_scores = {}
    for a, b, c, expected, label in analogy_battery:
        ia = resolve_token(m, a)
        ib = resolve_token(m, b)
        ic = resolve_token(m, c)

        if None in (ia, ib, ic):
            model_scores[label] = {'top1': None, 'cosine': None, 'expected_rank': None, 'success': False, 'skipped': True}
            continue

        vec = m.emb[ib] - m.emb[ia] + m.emb[ic]
        vec_norm = np.linalg.norm(vec)
        cos = (m.emb @ vec) / (m.norms * vec_norm + 1e-10)
        for idx in (ia, ib, ic):
            cos[idx] = -2

        top_idx = np.argsort(cos)[-5:][::-1]
        top1_token = m.tokens[top_idx[0]].strip()
        top1_cos = float(cos[top_idx[0]])

        # Check if expected answer is in top-5
        ie = resolve_token(m, expected)
        expected_rank = None
        if ie is not None:
            rank_list = list(top_idx)
            if ie in rank_list:
                expected_rank = rank_list.index(ie) + 1
            else:
                # Search further
                all_sorted = np.argsort(cos)[::-1]
                pos = np.where(all_sorted == ie)[0]
                expected_rank = int(pos[0]) + 1 if len(pos) > 0 else None

        success = top1_token.lower() == expected.lower()
        model_scores[label] = {
            'top1': top1_token,
            'cosine': round(top1_cos, 4),
            'expected': expected,
            'expected_rank': expected_rank,
            'success': success,
            'skipped': False,
        }

    scorecard[slug] = model_scores
    successes = sum(1 for v in model_scores.values() if v['success'])
    skipped = sum(1 for v in model_scores.values() if v.get('skipped'))
    total = len(analogy_battery) - skipped
    print(f"  {m.name:20s}  {successes}/{total} correct" + (f" ({skipped} skipped)" if skipped else ""))

# Print detailed scorecard
print("\n  Detailed results:")
header = f"  {'Analogy':20s}"
for slug in model_slugs:
    name = model_data[slug].name
    header += f"  {name:>15s}"
print(header)
print("  " + "-" * len(header))

for a, b, c, expected, label in analogy_battery:
    row = f"  {label:20s}"
    for slug in model_slugs:
        s = scorecard[slug][label]
        if s.get('skipped'):
            row += f"  {'(skip)':>15s}"
        elif s['success']:
            row += f"  {s['top1']:>15s}"
        else:
            row += f"  {('*' + (s['top1'] or '?')):>15s}"
    print(row)
print("  (* = wrong answer)")

# ============================================================
# 4. NEIGHBORHOOD JACCARD OVERLAP
# ============================================================
print("\n" + "="*60)
print("4. NEIGHBORHOOD JACCARD OVERLAP")
print("="*60)

probe_words = ['the', 'king', 'dog', 'good', 'France', 'water', 'big', 'run',
               'she', 'one', 'day', 'time', 'new', 'old', 'man', 'world',
               'just', 'make', 'think', 'back']

# Filter to probes that exist in all models
valid_probes = []
for word in probe_words:
    if all(resolve_token(model_data[s], word) is not None for s in model_slugs):
        valid_probes.append(word)

print(f"  Using {len(valid_probes)} probe tokens present in all models")

k_neighbors = 10
jaccard_matrix = {}

for i, s1 in enumerate(model_slugs):
    for s2 in model_slugs[i+1:]:
        m1, m2 = model_data[s1], model_data[s2]
        jaccards = []

        for word in valid_probes:
            idx1 = resolve_token(m1, word)
            idx2 = resolve_token(m2, word)

            # Get top-k neighbors as decoded strings
            cos1 = m1.normed_emb @ m1.normed_emb[idx1]
            cos1[idx1] = -2
            nn1 = set(m1.tokens[i].strip().lower() for i in np.argsort(cos1)[-k_neighbors:][::-1])

            cos2 = m2.normed_emb @ m2.normed_emb[idx2]
            cos2[idx2] = -2
            nn2 = set(m2.tokens[i].strip().lower() for i in np.argsort(cos2)[-k_neighbors:][::-1])

            jacc = len(nn1 & nn2) / len(nn1 | nn2) if nn1 | nn2 else 0
            jaccards.append(jacc)

        pair_key = f"{s1}_vs_{s2}"
        mean_j = float(np.mean(jaccards))
        jaccard_matrix[pair_key] = {
            'model_a': s1,
            'model_b': s2,
            'mean_jaccard': mean_j,
            'per_token': {w: round(j, 4) for w, j in zip(valid_probes, jaccards)},
        }
        print(f"  {m1.name:15s} vs {m2.name:15s}  mean Jaccard = {mean_j:.3f}")

# ============================================================
# 5. OUTLIER MIGRATION
# ============================================================
print("\n" + "="*60)
print("5. OUTLIER MIGRATION (norm rank correlation)")
print("="*60)

# For shared vocab tokens, rank by norm in each model
shared_tokens = list(shared_vocab.keys())
print(f"  Using {len(shared_tokens)} shared-vocabulary tokens")

norm_ranks = {}
for slug in model_slugs:
    m = model_data[slug]
    indices = [shared_vocab[t][slug] for t in shared_tokens]
    token_norms = m.norms[indices]
    # Convert to ranks
    norm_ranks[slug] = sp_stats.rankdata(token_norms)

outlier_correlations = {}
for i, s1 in enumerate(model_slugs):
    for s2 in model_slugs[i+1:]:
        rho, pval = sp_stats.spearmanr(norm_ranks[s1], norm_ranks[s2])
        pair_key = f"{s1}_vs_{s2}"
        outlier_correlations[pair_key] = {
            'model_a': s1,
            'model_b': s2,
            'spearman_rho': round(float(rho), 4),
            'p_value': float(pval),
        }
        n1 = model_data[s1].name
        n2 = model_data[s2].name
        print(f"  {n1:15s} vs {n2:15s}  Spearman rho = {rho:.3f} (p={pval:.2e})")

# ============================================================
# SAVE ALL RESULTS
# ============================================================

comparison = {
    'models': model_slugs,
    'shared_vocab_size': len(shared_tokens),
    'isotropy': isotropy,
    'ghost_universality': ghost_data,
    'analogy_scorecard': scorecard,
    'neighborhood_jaccard': jaccard_matrix,
    'outlier_migration': outlier_correlations,
}

with open(OUT / 'comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"\nResults saved to {OUT}/comparison.json")
print("Run dashboard.py to generate the visualization.")
