"""
Phase 2 Visualization: Ghost Cluster Heatmap

For interactive exploration, use:
  - analogy_explorer.py (analogies, localhost:8765)
  - neighbor_explorer.py (nearest neighbors, localhost:8766)

Reads precomputed data from deep_dive.py — no model loading required.

Usage: poetry run python phase2-ghost-cluster-and-analogies/charts.py [--model MODEL]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.html import inject_dark_mode
from common.viz import HEATMAP_SEPARATOR, SOFT_HEATMAP_SCALE

parser = argparse.ArgumentParser(description="Phase 2: Ghost cluster heatmap")
parser.add_argument('--model', default='gpt2', help="Model slug (must have run deep_dive.py first)")
args = parser.parse_args()

BASE = Path(__file__).parent / 'results' / args.model
if not BASE.exists():
    raise FileNotFoundError(f"No results for '{args.model}' — run deep_dive.py --model {args.model} first")

# ── Load data ─────────────────────────────────────────────────
cos_matrix_path = BASE / 'ghost_cosine_matrix.npy'
labels_path = BASE / 'ghost_labels.json'

if not cos_matrix_path.exists():
    print(f"No ghost cluster data for '{args.model}' — model may not have a ghost cluster.")
    exit(0)

cos_matrix = np.load(cos_matrix_path)
with open(labels_path) as f:
    ghost_data = json.load(f)

# New format has {labels, n_ghost}; handle both formats
if isinstance(ghost_data, dict):
    ghost_labels = ghost_data['labels']
    n_ghost = ghost_data['n_ghost']
else:
    # Legacy format: plain list
    ghost_labels = ghost_data
    n_ghost = len(ghost_labels) - 4  # old format had 4 reference tokens appended

# Load model name from results.json if available
model_name = args.model
results_path = BASE / 'results.json'
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    model_name = results.get('model', {}).get('name', args.model)

# ══════════════════════════════════════════════════════════════
# Ghost Cluster Heatmap (with reference tokens)
# ══════════════════════════════════════════════════════════════

display_labels = []
for i, label in enumerate(ghost_labels):
    if i < n_ghost:
        short = repr(label).strip("'")
        if len(short) > 10:
            short = short[:8] + '..'
        display_labels.append(f'{short} ({i})')
    else:
        # Reference tokens — highlight with star prefix
        display_labels.append(f'* {label}')

fig = go.Figure(data=go.Heatmap(
    z=cos_matrix,
    x=display_labels,
    y=display_labels,
    colorscale=SOFT_HEATMAP_SCALE,
    zmin=0.0, zmax=1.0,
    colorbar=dict(title='Cosine', thickness=18, outlinewidth=0),
    hovertemplate='%{x}<br>%{y}<br>cosine=%{z:.3f}<extra></extra>',
))

# Add a horizontal/vertical line to separate ghost cluster from reference tokens
fig.add_hline(y=n_ghost - 0.5, line=dict(color=HEATMAP_SEPARATOR, width=2))
fig.add_vline(x=n_ghost - 0.5, line=dict(color=HEATMAP_SEPARATOR, width=2))

fig.update_layout(
    title=f'{model_name} — Ghost Cluster vs Reference Tokens (Cosine Similarity)',
    width=1000, height=900,
    template='plotly_dark',
    paper_bgcolor='#151515',
    plot_bgcolor='#151515',
    xaxis=dict(tickangle=45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=8), autorange='reversed'),
)

out_path = BASE / 'ghost_heatmap.html'
html = fig.to_html(include_plotlyjs=True, full_html=True)
with open(out_path, 'w') as f:
    f.write(inject_dark_mode(html))
print(f"Saved to {out_path}")
