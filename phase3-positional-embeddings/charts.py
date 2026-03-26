"""
Phase 3 visualization: positional geometry and token-position interaction.

Reads precomputed data from explore.py — no model loading required.

Usage: poetry run python phase3-positional-embeddings/charts.py [--model MODEL]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.html import inject_dark_mode
from common.viz import SOFT_HEATMAP_SCALE


parser = argparse.ArgumentParser(description="Phase 3: Positional embedding charts")
parser.add_argument('--model', default='gpt2', help="Model slug (must have run explore.py first)")
args = parser.parse_args()

BASE = Path(__file__).parent / 'results' / args.model
if not BASE.exists():
    raise FileNotFoundError(f"No results for '{args.model}' — run explore.py --model {args.model} first")

with open(BASE / 'results.json') as f:
    results = json.load(f)

out_path = BASE / 'phase3_charts.html'
model_name = results.get('model', {}).get('name', args.model)

if not results.get('compatible'):
    skip_reason = results.get('skip_reason', 'This model is not compatible with Phase 3.')
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{model_name} — Phase 3</title>
</head>
<body>
  <main style="max-width:800px;margin:48px auto;font-family:SF Mono,Consolas,monospace;">
    <h1>{model_name} — Phase 3</h1>
    <p>{skip_reason}</p>
    <p>Phase 3 currently analyzes learned absolute position matrices only.</p>
  </main>
</body>
</html>
"""
    with open(out_path, 'w') as f:
        f.write(inject_dark_mode(html))
    print(f"Saved to {out_path}")
    raise SystemExit(0)

pos_norms = np.load(BASE / 'position_norms.npy')
pos_coords = np.load(BASE / 'position_pca_coords.npy')
cos_matrix = np.load(BASE / 'position_cosine_matrix.npy')

heatmap_positions = results['position_similarity']['sampled_heatmap_positions']
interaction = results['token_position_interaction']
interaction_positions = [int(p) for p in interaction['positions']]
mean_jaccard = [interaction['mean_jaccard_by_position'][str(p)] for p in interaction_positions]
mean_drift = [interaction['mean_token_drift_by_position'][str(p)] for p in interaction_positions]
analogy_top1 = [interaction['analogy_top1_by_position'][str(p)] for p in interaction_positions]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "Position Norm Profile",
        "PC1 vs PC2 Path",
        "Sampled Position Cosine Heatmap",
        "Probe Stability Across Positions",
    ],
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "heatmap"}, {"type": "scatter"}],
    ],
)

positions = np.arange(len(pos_norms))
fig.add_trace(
    go.Scatter(
        x=positions,
        y=pos_norms,
        mode='lines',
        line=dict(color='#b8a06a', width=2),
        name='norm',
        hovertemplate='pos %{x}: %{y:.3f}<extra></extra>',
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=pos_coords[:, 0],
        y=pos_coords[:, 1],
        mode='lines+markers',
        marker=dict(size=4, color=positions, colorscale='Viridis', showscale=True, colorbar=dict(title='Position')),
        line=dict(color='rgba(231,223,210,0.25)', width=1),
        name='pc-path',
        hovertemplate='pos %{marker.color}: (%{x:.2f}, %{y:.2f})<extra></extra>',
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Heatmap(
        z=cos_matrix,
        x=heatmap_positions,
        y=heatmap_positions,
        colorscale=SOFT_HEATMAP_SCALE,
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title='Cosine'),
        hovertemplate='pos %{x} vs %{y}: %{z:.3f}<extra></extra>',
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=interaction_positions,
        y=mean_jaccard,
        mode='lines+markers',
        line=dict(color='#78a7a3', width=2),
        name='mean nn jaccard',
        hovertemplate='pos %{x}: %{y:.3f}<extra></extra>',
    ),
    row=2,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=interaction_positions,
        y=mean_drift,
        mode='lines+markers',
        line=dict(color='#e7dfd2', width=2),
        name='mean token drift cosine',
        hovertemplate='pos %{x}: %{y:.3f}<extra></extra>',
    ),
    row=2,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=interaction_positions,
        y=analogy_top1,
        mode='lines+markers',
        line=dict(color='#3e5c76', width=2, dash='dash'),
        name='analogy top1 rate',
        hovertemplate='pos %{x}: %{y:.2f}<extra></extra>',
    ),
    row=2,
    col=2,
)

fig.update_xaxes(title_text='Position', row=1, col=1)
fig.update_yaxes(title_text='L2 norm', row=1, col=1)
fig.update_xaxes(title_text='PC1', row=1, col=2)
fig.update_yaxes(title_text='PC2', row=1, col=2)
fig.update_xaxes(title_text='Position', row=2, col=1)
fig.update_yaxes(title_text='Position', row=2, col=1, autorange='reversed')
fig.update_xaxes(title_text='Position', row=2, col=2)
fig.update_yaxes(title_text='Score', row=2, col=2, range=[0, 1.05])

fig.update_layout(
    title=f'{model_name} — Phase 3 Positional Embedding Geometry',
    width=1200,
    height=900,
    template='plotly_dark',
    paper_bgcolor='#151515',
    plot_bgcolor='#151515',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
)

html = fig.to_html(include_plotlyjs=True, full_html=True)
with open(out_path, 'w') as f:
    f.write(inject_dark_mode(html))

print(f"Saved to {out_path}")
