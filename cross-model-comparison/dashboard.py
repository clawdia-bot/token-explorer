"""
Cross-Model Comparison Dashboard

Plotly visualization with 4 panels:
  1. Isotropy Radar Chart
  2. Analogy Scorecard Heatmap
  3. Ghost Cluster Summary
  4. Neighborhood Agreement Heatmap

Reads precomputed data from compare.py — no model loading required.

Usage: poetry run python cross-model-comparison/dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.html import inject_dark_mode

DATA = Path(__file__).parent / 'results'
if not (DATA / 'comparison.json').exists():
    raise FileNotFoundError("No comparison.json found — run compare.py first")

with open(DATA / 'comparison.json') as f:
    comp = json.load(f)

model_slugs = comp['models']
isotropy = comp['isotropy']
ghost = comp['ghost_universality']
scorecard = comp['analogy_scorecard']
jaccard = comp['neighborhood_jaccard']

names = {slug: isotropy[slug]['name'] for slug in model_slugs}

# ============================================================
# Panel 1: Isotropy Radar Chart
# ============================================================

radar_axes = ['Anisotropy', 'PR / dim', 'Eff. dim / dim', 'Mean norm (norm.)']
raw_vals = {}
for slug in model_slugs:
    iso = isotropy[slug]
    raw_vals[slug] = [
        iso['anisotropy'],
        iso['participation_ratio_normalized'],
        iso['entropy_effective_dims_normalized'],
        iso['mean_norm'],
    ]

all_vals = np.array(list(raw_vals.values()))
mins = all_vals.min(axis=0)
maxs = all_vals.max(axis=0)
ranges = maxs - mins
ranges[ranges == 0] = 1

radar_colors = ['#4285f4', '#ea4335', '#34a853', '#fbbc04', '#ab47bc']
fig_radar = go.Figure()
for i, slug in enumerate(model_slugs):
    normed = (np.array(raw_vals[slug]) - mins) / ranges
    vals = list(normed) + [normed[0]]
    axes = radar_axes + [radar_axes[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=axes,
        fill='toself',
        fillcolor=radar_colors[i % len(radar_colors)],
        opacity=0.15,
        line=dict(color=radar_colors[i % len(radar_colors)], width=2),
        name=names[slug],
        hovertemplate=f"{names[slug]}<br>%{{theta}}: %{{r:.3f}}<extra></extra>",
    ))

fig_radar.update_layout(
    title='Embedding Geometry Profile (normalized to [0,1])',
    template='plotly_dark',
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    width=700,
    height=500,
)

# ============================================================
# Panel 2: Analogy Scorecard Heatmap
# ============================================================

analogy_labels = list(list(scorecard.values())[0].keys())
n_models = len(model_slugs)
n_analogies = len(analogy_labels)
z_scores = np.zeros((n_models, n_analogies))
text_matrix = [[''] * n_analogies for _ in range(n_models)]

for i, slug in enumerate(model_slugs):
    for j, label in enumerate(analogy_labels):
        entry = scorecard[slug][label]
        z_scores[i, j] = entry['cosine'] if entry['cosine'] is not None else 0
        marker = '' if entry['success'] else '*'
        text_matrix[i][j] = f"{marker}{entry['top1']}"

fig_analogy = go.Figure(data=go.Heatmap(
    z=z_scores,
    x=analogy_labels,
    y=[names[slug] for slug in model_slugs],
    text=text_matrix,
    texttemplate='%{text}',
    textfont=dict(size=10),
    colorscale=[
        [0.0, '#1a1a2e'],
        [0.3, '#16213e'],
        [0.5, '#0f3460'],
        [0.7, '#e94560'],
        [0.9, '#fbbc04'],
        [1.0, '#34a853'],
    ],
    zmin=0,
    zmax=1,
    colorbar=dict(title='Cosine'),
    hovertemplate='%{y}<br>%{x}<br>cos=%{z:.3f}<br>answer: %{text}<extra></extra>',
))

fig_analogy.update_layout(
    title='Analogy Scorecard (* = wrong answer)',
    template='plotly_dark',
    width=900,
    height=max(300, n_models * 60 + 100),
    xaxis=dict(tickangle=45),
)

# ============================================================
# Panel 3: Ghost Cluster Summary
# ============================================================

ghost_slugs = [slug for slug in model_slugs if slug in ghost]
ghost_names = [names[slug] for slug in ghost_slugs]
ghost_counts = [ghost[slug]['count'] for slug in ghost_slugs]
ghost_cosines = [ghost[slug]['mean_cosine'] if ghost[slug]['mean_cosine'] is not None else 0 for slug in ghost_slugs]
ghost_mins = [ghost[slug]['min_cosine'] if ghost[slug]['min_cosine'] is not None else 0 for slug in ghost_slugs]

fig_ghost = go.Figure()
fig_ghost.add_trace(go.Bar(
    x=ghost_names,
    y=ghost_cosines,
    marker_color='#e94560',
    text=[f"n={count}" for count in ghost_counts],
    textposition='outside',
    name='Mean pairwise cosine',
))
fig_ghost.add_trace(go.Scatter(
    x=ghost_names,
    y=ghost_mins,
    mode='markers+lines',
    marker=dict(size=10, color='#fbbc04', symbol='diamond'),
    line=dict(color='#fbbc04', dash='dash'),
    name='Minimum pairwise cosine',
))

fig_ghost.update_layout(
    title='Ghost Cluster Summary',
    template='plotly_dark',
    width=700,
    height=400,
    yaxis=dict(title='Cosine Similarity', range=[0, 1.05]),
)

# ============================================================
# Panel 4: Neighborhood Agreement Heatmap
# ============================================================

jacc_matrix = np.eye(n_models)
for pair_data in jaccard.values():
    i = model_slugs.index(pair_data['model_a'])
    j = model_slugs.index(pair_data['model_b'])
    jacc_matrix[i, j] = pair_data['mean_jaccard']
    jacc_matrix[j, i] = pair_data['mean_jaccard']

fig_jaccard = go.Figure(data=go.Heatmap(
    z=jacc_matrix,
    x=[names[slug] for slug in model_slugs],
    y=[names[slug] for slug in model_slugs],
    colorscale=[
        [0.0, '#1a1a2e'],
        [0.3, '#0f3460'],
        [0.6, '#e94560'],
        [0.8, '#fbbc04'],
        [1.0, '#34a853'],
    ],
    zmin=0,
    zmax=1,
    colorbar=dict(title='Jaccard'),
    hovertemplate='%{x}<br>%{y}<br>Jaccard=%{z:.3f}<extra></extra>',
    texttemplate='%{z:.2f}',
    textfont=dict(size=11),
))

fig_jaccard.update_layout(
    title=(
        f'Neighborhood Agreement (top-{comp["neighbor_k"]}, '
        f'{comp["neighbor_probe_count"]} probes, '
        f'{comp["concept_inventory_size"]} concepts)'
    ),
    template='plotly_dark',
    width=650,
    height=500,
)

# ============================================================
# Combine into single HTML
# ============================================================

html_parts = []
html_parts.append(f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cross-Model Comparison Dashboard</title>
<style>
  :root {{ color-scheme: dark; }}
  html {{ background: #151515; }}
  body {{ background: #1e1e1e; color: #eee; font-family: 'SF Mono', 'Consolas', monospace; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 8px; }}
  .subtitle {{ text-align: center; color: #888; font-size: 14px; margin-bottom: 30px; }}
  .panel {{ margin-bottom: 40px; display: flex; justify-content: center; }}
  .row {{ display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }}
</style>
</head>
<body>
<h1>Cross-Model Embedding Comparison</h1>
<p class="subtitle">{len(model_slugs)} models: {', '.join(names[slug] for slug in model_slugs)}</p>
""")

html_parts.append('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n')

for i, (fig, panel_id) in enumerate([
    (fig_radar, 'radar'),
    (fig_analogy, 'analogy'),
    (fig_ghost, 'ghost'),
    (fig_jaccard, 'jaccard'),
]):
    if i == 2:
        html_parts.append('<div class="row">')
    html_parts.append(f'<div class="panel"><div id="{panel_id}"></div></div>')
    if i == 3:
        html_parts.append('</div>')
    fig_json = fig.to_json()
    html_parts.append(f"""
<script>
(function() {{
  var data = {fig_json};
  Plotly.newPlot('{panel_id}', data.data, data.layout);
}})();
</script>
""")

html_parts.append('</body></html>')

out_path = DATA / 'comparison_dashboard.html'
with open(out_path, 'w') as f:
    f.write(inject_dark_mode('\n'.join(html_parts)))

print(f"Dashboard saved to {out_path}")
