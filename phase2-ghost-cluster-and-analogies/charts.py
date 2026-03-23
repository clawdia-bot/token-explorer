"""
Phase 2 Visualization: Ghost Cluster, Analogies, and Neighbor Graphs
Interactive plotly HTML with:
  1. Ghost cluster cosine similarity heatmap
  2. Analogy parallelogram diagrams (2x2 grid)
  3. Nearest neighbor radial graphs (2x3 grid)

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
with open(OUT / 'analogy_vectors.json') as f:
    analogy_data = json.load(f)
with open(OUT / 'nn_viz_data.json') as f:
    nn_data = json.load(f)

# ══════════════════════════════════════════════════════════════
# FIGURE 1: Ghost Cluster Heatmap
# ══════════════════════════════════════════════════════════════

# Clean up labels for display
display_labels = []
for i, label in enumerate(ghost_labels):
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

fig1 = go.Figure(data=go.Heatmap(
    z=cos_matrix,
    x=display_labels,
    y=display_labels,
    colorscale=[
        [0.0, '#1a1a2e'],
        [0.3, '#16213e'],
        [0.5, '#0f3460'],
        [0.7, '#e94560'],
        [0.9, '#fbbc04'],
        [1.0, '#ffffff'],
    ],
    zmin=-0.1, zmax=1.0,
    colorbar=dict(title='Cosine'),
    hovertemplate='%{x}<br>%{y}<br>cosine=%{z:.3f}<extra></extra>',
))

fig1.update_layout(
    title='Ghost Cluster: Pairwise Cosine Similarity (tokens 188-221)',
    width=900, height=800,
    template='plotly_dark',
    xaxis=dict(tickangle=45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=8), autorange='reversed'),
)

fig1.write_html(str(OUT / 'ghost_heatmap.html'))
print(f"Saved ghost_heatmap.html")

# ══════════════════════════════════════════════════════════════
# FIGURE 2: Analogy Parallelograms (2x2)
# ══════════════════════════════════════════════════════════════

analogy_keys = list(analogy_data.keys())
fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=[k.replace('_', ' ').title() for k in analogy_keys],
    vertical_spacing=0.15,
    horizontal_spacing=0.10,
)

for idx, (label, data) in enumerate(analogy_data.items()):
    row = idx // 2 + 1
    col = idx % 2 + 1

    vecs = np.array(data['vectors'])  # 4 x 768
    token_labels = data['labels']

    # PCA: project 4 points to 2D using SVD of the centered 4-point set
    centered = vecs - vecs.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T  # 4 x 2

    # Points: A, B, C, D (a:b::c:d)
    # Draw parallelogram edges: A->B, C->D (the analogy pairs), A->C, B->D (the relationship)
    colors = ['#4285f4', '#ea4335', '#4285f4', '#ea4335']
    names = ['A', 'B', 'C', 'D=?']

    # Draw edges
    # A->B (source pair)
    fig2.add_trace(go.Scatter(
        x=[proj[0, 0], proj[1, 0]], y=[proj[0, 1], proj[1, 1]],
        mode='lines', line=dict(color='#666', width=1, dash='dash'),
        showlegend=False, hoverinfo='skip',
    ), row=row, col=col)
    # C->D (target pair)
    fig2.add_trace(go.Scatter(
        x=[proj[2, 0], proj[3, 0]], y=[proj[2, 1], proj[3, 1]],
        mode='lines', line=dict(color='#666', width=1, dash='dash'),
        showlegend=False, hoverinfo='skip',
    ), row=row, col=col)
    # A->C (relationship 1)
    fig2.add_trace(go.Scatter(
        x=[proj[0, 0], proj[2, 0]], y=[proj[0, 1], proj[2, 1]],
        mode='lines', line=dict(color='#444', width=1, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ), row=row, col=col)
    # B->D (relationship 2)
    fig2.add_trace(go.Scatter(
        x=[proj[1, 0], proj[3, 0]], y=[proj[1, 1], proj[3, 1]],
        mode='lines', line=dict(color='#444', width=1, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ), row=row, col=col)

    # Draw points
    for i in range(4):
        fig2.add_trace(go.Scatter(
            x=[proj[i, 0]], y=[proj[i, 1]],
            mode='markers+text',
            marker=dict(size=12, color=colors[i]),
            text=[token_labels[i]],
            textposition='top center',
            textfont=dict(size=11, color='white'),
            showlegend=False,
            hovertemplate=f'{token_labels[i]}<extra></extra>',
        ), row=row, col=col)

    fig2.update_xaxes(showticklabels=False, showgrid=False, row=row, col=col)
    fig2.update_yaxes(showticklabels=False, showgrid=False, row=row, col=col)

fig2.update_layout(
    title='Embedding Analogies — Parallelogram Structure in PCA',
    width=900, height=750,
    template='plotly_dark',
)

fig2.write_html(str(OUT / 'analogy_parallelograms.html'))
print(f"Saved analogy_parallelograms.html")

# ══════════════════════════════════════════════════════════════
# FIGURE 3: Nearest Neighbor Radial Graphs (2x3)
# ══════════════════════════════════════════════════════════════

nn_keys = list(nn_data.keys())
n_probes = len(nn_keys)
n_cols = 3
n_rows = (n_probes + n_cols - 1) // n_cols

fig3 = make_subplots(
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
        fig3.add_trace(go.Scatter(
            x=[cx, nx], y=[cy, ny],
            mode='lines',
            line=dict(color='#555', width=1),
            showlegend=False, hoverinfo='skip',
        ), row=row, col=col)

        # Neighbor point
        fig3.add_trace(go.Scatter(
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
    fig3.add_trace(go.Scatter(
        x=[cx], y=[cy],
        mode='markers+text',
        marker=dict(size=14, color='#fbbc04', symbol='diamond'),
        text=[center['label']],
        textposition='top center',
        textfont=dict(size=11, color='white'),
        showlegend=False,
        hovertemplate=f"{center['label']}<extra></extra>",
    ), row=row, col=col)

    fig3.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
    fig3.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, scaleanchor=f'x{idx+1}', row=row, col=col)

fig3.update_layout(
    title="Nearest Neighbor Graphs — Top 5 by Cosine Similarity",
    width=1100, height=700,
    template='plotly_dark',
)

fig3.write_html(str(OUT / 'neighbor_graphs.html'))
print(f"Saved neighbor_graphs.html")
