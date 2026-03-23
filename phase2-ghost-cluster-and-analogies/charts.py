"""
Phase 2 Visualization: Ghost Cluster and Neighbor Graphs
Interactive plotly HTML with:
  1. Ghost cluster cosine similarity heatmap (with reference tokens)
  2. Nearest neighbor radial graphs (2x3 grid)

For the interactive analogy explorer, run analogy_explorer.py instead.
Reads precomputed data from deep_dive.py — no model loading required.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# ── Load data ─────────────────────────────────────────────────
cos_matrix = np.load(OUT / 'ghost_cosine_matrix.npy')
with open(OUT / 'ghost_labels.json') as f:
    ghost_labels = json.load(f)
with open(OUT / 'nn_viz_data.json') as f:
    nn_data = json.load(f)

# ══════════════════════════════════════════════════════════════
# FIGURE 1: Ghost Cluster Heatmap
# ══════════════════════════════════════════════════════════════

# Clean up labels for display
# First 34 are ghost cluster (188-221), rest are reference tokens
n_ghost = 34
display_labels = []
for i, label in enumerate(ghost_labels):
    if i < n_ghost:
        idx = 188 + i
        if idx == 198:
            display_labels.append(f'\\n ({idx})')
        elif idx == 220:
            display_labels.append(f'space ({idx})')
        else:
            short = repr(label).strip("'")
            if len(short) > 10:
                short = short[:8] + '..'
            display_labels.append(f'{short} ({idx})')
    else:
        # Reference tokens — highlight with star prefix
        display_labels.append(f'* {label}')

fig1 = go.Figure(data=go.Heatmap(
    z=cos_matrix,
    x=display_labels,
    y=display_labels,
    colorscale=[
        [0.0, '#1a1a2e'],
        [0.2, '#16213e'],
        [0.4, '#0f3460'],
        [0.6, '#e94560'],
        [0.8, '#fbbc04'],
        [1.0, '#ffffff'],
    ],
    zmin=0.0, zmax=1.0,
    colorbar=dict(title='Cosine'),
    hovertemplate='%{x}<br>%{y}<br>cosine=%{z:.3f}<extra></extra>',
))

# Add a horizontal/vertical line to separate ghost cluster from reference tokens
fig1.add_hline(y=n_ghost - 0.5, line=dict(color='#34a853', width=2))
fig1.add_vline(x=n_ghost - 0.5, line=dict(color='#34a853', width=2))

fig1.update_layout(
    title='Ghost Cluster vs Reference Tokens — Cosine Similarity',
    width=1000, height=900,
    template='plotly_dark',
    xaxis=dict(tickangle=45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=8), autorange='reversed'),
)

fig1.write_html(str(OUT / 'ghost_heatmap.html'))
print(f"Saved ghost_heatmap.html")

# ══════════════════════════════════════════════════════════════
# FIGURE 2: Nearest Neighbor Radial Graphs (2x3)
# ══════════════════════════════════════════════════════════════

nn_keys = list(nn_data.keys())
n_probes = len(nn_keys)
n_cols = 3
n_rows = (n_probes + n_cols - 1) // n_cols

fig2 = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=[nn_data[k]['center']['label'] for k in nn_keys],
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

for idx, key in enumerate(nn_keys):
    row = idx // n_cols + 1
    col = idx % n_cols + 1
    entry = nn_data[key]
    center = entry['center']
    neighbors = entry['neighbors']

    # Arrange neighbors in a circle around center
    n = len(neighbors)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Radius proportional to (1 - cosine) so closer = nearer
    cx, cy = 0, 0

    for i, nb in enumerate(neighbors):
        radius = 1 - nb['cosine']  # closer cosine = shorter radius
        nx = radius * np.cos(angles[i])
        ny = radius * np.sin(angles[i])

        # Edge from center to neighbor
        fig2.add_trace(go.Scatter(
            x=[cx, nx], y=[cy, ny],
            mode='lines',
            line=dict(color='#555', width=1),
            showlegend=False, hoverinfo='skip',
        ), row=row, col=col)

        # Neighbor point
        fig2.add_trace(go.Scatter(
            x=[nx], y=[ny],
            mode='markers+text',
            marker=dict(size=8, color='#7baaf7'),
            text=[nb['label']],
            textposition='middle right' if np.cos(angles[i]) >= 0 else 'middle left',
            textfont=dict(size=9, color='#ccc'),
            showlegend=False,
            hovertemplate=f"{nb['label']}<br>cosine={nb['cosine']:.3f}<extra></extra>",
        ), row=row, col=col)

    # Center point
    fig2.add_trace(go.Scatter(
        x=[cx], y=[cy],
        mode='markers+text',
        marker=dict(size=14, color='#fbbc04', symbol='diamond'),
        text=[center['label']],
        textposition='top center',
        textfont=dict(size=11, color='white'),
        showlegend=False,
        hovertemplate=f"{center['label']}<extra></extra>",
    ), row=row, col=col)

    fig2.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
    fig2.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, scaleanchor=f'x{idx+1}', row=row, col=col)

fig2.update_layout(
    title="Nearest Neighbor Graphs — Top 5 by Cosine Similarity",
    width=1100, height=700,
    template='plotly_dark',
)

fig2.write_html(str(OUT / 'neighbor_graphs.html'))
print(f"Saved neighbor_graphs.html")
