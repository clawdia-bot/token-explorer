"""
Phase 1 Visualization: Embedding Space Structure
Interactive plotly HTML with:
  1. Norm distribution (histogram)
  2. Cumulative variance explained (PCA scree)
  3. Pairwise cosine similarity distribution (anisotropy)
  4. Mean norm by token category (bar chart)

Reads precomputed data from explore.py — no model loading required.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path

# ── Load results ──────────────────────────────────────────────
OUT = Path(__file__).parent
with open(OUT / 'results.json') as f:
    results = json.load(f)

explained_ratio = np.load(OUT / 'explained_ratio.npy')
cumulative = np.cumsum(explained_ratio)

# ── Build figure ──────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Distance from Origin vs Centroid',
        'Cumulative Variance Explained (PCA)',
        'Pairwise Cosine Similarity (Anisotropy)',
        'Mean Norm by Token Category',
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.10,
)

# ── Panel 1: Norm distribution ────────────────────────────────
norm_hist = results['norms']['histogram']
edges = norm_hist['edges']
counts = norm_hist['counts']
# Bin centers for bar chart
centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
widths = [edges[i+1] - edges[i] for i in range(len(counts))]

# Normalize both histograms to density so they're comparable despite different scales
origin_total = sum(counts)
origin_density = [c / (origin_total * widths[i]) for i, c in enumerate(counts)]

dist_hist = results['centroid']['histogram']
dist_edges = dist_hist['edges']
dist_counts = dist_hist['counts']
dist_centers = [(dist_edges[i] + dist_edges[i+1]) / 2 for i in range(len(dist_counts))]
dist_widths = [dist_edges[i+1] - dist_edges[i] for i in range(len(dist_counts))]
dist_total = sum(dist_counts)
dist_density = [c / (dist_total * dist_widths[i]) for i, c in enumerate(dist_counts)]

fig.add_trace(go.Bar(
    x=centers, y=origin_density, width=widths,
    marker_color='#4285f4', opacity=0.6,
    name='From origin',
    showlegend=True,
    legendgroup='panel1',
), row=1, col=1)

fig.add_trace(go.Bar(
    x=dist_centers, y=dist_density, width=dist_widths,
    marker_color='#ff9800', opacity=0.6,
    name='From centroid',
    showlegend=True,
    legendgroup='panel1',
), row=1, col=1)

fig.update_layout(barmode='overlay')

fig.update_xaxes(title_text='Distance', row=1, col=1)
fig.update_yaxes(title_text='Density', row=1, col=1)

# ── Panel 2: Cumulative variance ──────────────────────────────
components = np.arange(1, len(cumulative) + 1)

fig.add_trace(go.Scatter(
    x=components, y=cumulative * 100,
    mode='lines', line=dict(color='#34a853', width=2),
    name='Cumulative variance',
    showlegend=False,
), row=1, col=2)

# Threshold markers
thresholds = results['pca']['threshold_components']
for pct, label, tpos in [('0.5', '50%', 'bottom right'), ('0.9', '90%', 'top left'), ('0.99', '99%', 'top left')]:
    k = thresholds[pct]
    fig.add_trace(go.Scatter(
        x=[k], y=[float(pct) * 100],
        mode='markers+text',
        marker=dict(size=10, color='#fbbc04', symbol='diamond'),
        text=[f'{label} at {k}'], textposition=tpos,
        textfont=dict(size=10),
        showlegend=False,
    ), row=1, col=2)

fig.update_xaxes(title_text='Number of Components', type='log', row=1, col=2)
fig.update_yaxes(title_text='Variance Explained (%)', row=1, col=2)

# ── Panel 3: Cosine similarity distribution ───────────────────
cos_hist = results['anisotropy']['histogram']
cos_edges = cos_hist['edges']
cos_counts = cos_hist['counts']
cos_centers = [(cos_edges[i] + cos_edges[i+1]) / 2 for i in range(len(cos_counts))]
cos_widths = [cos_edges[i+1] - cos_edges[i] for i in range(len(cos_counts))]

fig.add_trace(go.Bar(
    x=cos_centers, y=cos_counts, width=cos_widths,
    marker_color='#ab47bc', opacity=0.85,
    name='Token pairs',
    showlegend=False,
), row=2, col=1)

# Mark the mean
mean_cos = results['anisotropy']['mean_cosine']
fig.add_vline(
    x=mean_cos, line=dict(color='#fbbc04', width=2, dash='dash'),
    annotation_text=f'mean={mean_cos:.3f}',
    annotation_position='top right',
    annotation_font_size=10,
    row=2, col=1,
)

# Mark where isotropic would be
fig.add_vline(
    x=0, line=dict(color='#9e9e9e', width=1, dash='dot'),
    annotation_text='isotropic',
    annotation_position='bottom right',
    annotation_font_size=9,
    row=2, col=1,
)

fig.update_xaxes(title_text='Cosine Similarity', row=2, col=1)
fig.update_yaxes(title_text='Token Pairs', row=2, col=1)

# ── Panel 4: Norm by category ────────────────────────────────
cats = results['categories']
sorted_cats = sorted(cats.items(), key=lambda x: x[1]['mean_norm'])
cat_names = [c[0] for c in sorted_cats]
cat_means = [c[1]['mean_norm'] for c in sorted_cats]
cat_counts = [c[1]['count'] for c in sorted_cats]

fig.add_trace(go.Bar(
    y=cat_names, x=cat_means,
    orientation='h',
    marker_color='#ea4335', opacity=0.85,
    text=[f'n={c:,}' for c in cat_counts],
    textposition='inside', textfont=dict(size=9, color='white'),
    name='Categories',
    showlegend=False,
), row=2, col=2)

fig.update_xaxes(title_text='Mean L2 Norm', row=2, col=2)

# ── Layout ────────────────────────────────────────────────────
fig.update_layout(
    title='GPT-2 Token Embedding Space — Phase 1 Findings',
    width=1400, height=800,
    template='plotly_dark',
)

out_path = OUT / 'phase1_charts.html'
fig.write_html(str(out_path))
print(f"Saved to {out_path}")
