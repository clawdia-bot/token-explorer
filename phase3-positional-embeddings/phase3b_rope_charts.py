"""
Phase 3B visualization: RoPE geometry and first-layer probe effects.

Reads precomputed data from phase3b_rope.py.

Usage: poetry run python phase3-positional-embeddings/phase3b_rope_charts.py [--model MODEL]
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


parser = argparse.ArgumentParser(description="Phase 3B: RoPE charts")
parser.add_argument('--model', default='pythia-70m', help="Model slug (must have run phase3b_rope.py first)")
args = parser.parse_args()

BASE = Path(__file__).parent / 'rope-results' / args.model
if not BASE.exists():
    raise FileNotFoundError(f"No Phase 3B results for '{args.model}' — run phase3b_rope.py --model {args.model} first")

with open(BASE / 'results.json') as f:
    results = json.load(f)

out_path = BASE / 'phase3b_rope_charts.html'
model_name = results.get('model', {}).get('name', args.model)

if not results.get('compatible'):
    skip_reason = results.get('skip_reason', 'This model is not compatible with Phase 3B.')
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{model_name} — Phase 3B</title>
</head>
<body>
  <main style="max-width:800px;margin:48px auto;font-family:SF Mono,Consolas,monospace;">
    <h1>{model_name} — Phase 3B</h1>
    <p>{skip_reason}</p>
    <p>Phase 3B analyzes RoPE models only.</p>
  </main>
</body>
</html>
"""
    with open(out_path, 'w') as f:
        f.write(inject_dark_mode(html))
    print(f"Saved to {out_path}")
    raise SystemExit(0)

score_matrices = np.load(BASE / 'probe_score_matrices.npy')
kernel_gaps = [int(k) for k in results['relative_kernel']['gaps'].keys()]
kernel_values = [results['relative_kernel']['gaps'][str(g)] for g in kernel_gaps]
positions = [int(p) for p in results['qk_rotation']['positions']]
query_drift = [results['qk_rotation']['mean_query_drift_by_position'][str(p)] for p in positions]
key_drift = [results['qk_rotation']['mean_key_drift_by_position'][str(p)] for p in positions]
score_spearman = [results['qk_rotation']['score_spearman_by_position'][str(p)] for p in positions]
probe_labels = [label.strip() for label in results['qk_rotation']['probe_labels']]

period_samples = np.array(results['rope']['periods']['samples'])
rotary_pairs = np.arange(1, len(period_samples) + 1)
far_idx = len(positions) - 1
delta_matrix = score_matrices[far_idx] - score_matrices[0]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "RoPE Relative-Offset Kernel",
        "Fastest Rotary Periods",
        "First-Layer q/k Drift",
        f"Score Delta at Position {positions[far_idx]}",
    ],
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "heatmap"}],
    ],
)

fig.add_trace(
    go.Scatter(
        x=kernel_gaps,
        y=kernel_values,
        mode='lines+markers',
        line=dict(color='#b8a06a', width=2),
        marker=dict(size=5),
        showlegend=False,
        hovertemplate='gap %{x}: %{y:.3f}<extra></extra>',
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=rotary_pairs,
        y=period_samples,
        mode='lines+markers',
        line=dict(color='#78a7a3', width=2),
        marker=dict(size=6),
        showlegend=False,
        hovertemplate='pair %{x}: period %{y:.2f}<extra></extra>',
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Scatter(
        x=positions,
        y=query_drift,
        mode='lines+markers',
        line=dict(color='#e7dfd2', width=2),
        name='query drift',
        hovertemplate='pos %{x}: %{y:.3f}<extra></extra>',
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=positions,
        y=key_drift,
        mode='lines+markers',
        line=dict(color='#3e5c76', width=2),
        name='key drift',
        hovertemplate='pos %{x}: %{y:.3f}<extra></extra>',
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=positions,
        y=score_spearman,
        mode='lines+markers',
        line=dict(color='#5f7f92', width=2, dash='dash'),
        name='score spearman',
        hovertemplate='pos %{x}: %{y:.3f}<extra></extra>',
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Heatmap(
        z=delta_matrix,
        x=probe_labels,
        y=probe_labels,
        colorscale=SOFT_HEATMAP_SCALE,
        zmid=0,
        colorbar=dict(title='Delta', thickness=14, len=0.38, y=0.2, yanchor='middle'),
        hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>',
        showlegend=False,
    ),
    row=2,
    col=2,
)

fig.update_xaxes(title_text='Relative offset', row=1, col=1, type='log')
fig.update_yaxes(title_text='Mean cosine', row=1, col=1, range=[-0.2, 1.05])
fig.update_xaxes(title_text='Rotary pair index', row=1, col=2)
fig.update_yaxes(title_text='Period (tokens)', row=1, col=2, type='log')
fig.update_xaxes(title_text='Query position', row=2, col=1, type='log')
fig.update_yaxes(title_text='Cosine / correlation', row=2, col=1, range=[-0.05, 1.05])
fig.update_xaxes(title_text='Key token', row=2, col=2, tickangle=45)
fig.update_yaxes(title_text='Query token', row=2, col=2, autorange='reversed')

fig.update_layout(
    title=f'{model_name} — Phase 3B RoPE Geometry',
    height=760,
    template='plotly_dark',
    paper_bgcolor='#151515',
    plot_bgcolor='#151515',
    margin=dict(l=60, r=40, t=80, b=60),
    legend=dict(orientation='v', yanchor='top', y=0.46, xanchor='left', x=0.03, bgcolor='rgba(21,21,21,0.65)'),
)

html = fig.to_html(include_plotlyjs=True, full_html=True, config={'responsive': True})
with open(out_path, 'w') as f:
    f.write(inject_dark_mode(html))

print(f"Saved to {out_path}")
