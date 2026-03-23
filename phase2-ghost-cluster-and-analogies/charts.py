"""
Phase 2 Visualization: Ghost Cluster Heatmap

For interactive exploration, use:
  - analogy_explorer.py (analogies, localhost:8765)
  - neighbor_explorer.py (nearest neighbors, localhost:8766)

Reads precomputed data from deep_dive.py — no model loading required.
"""

import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# ── Load data ─────────────────────────────────────────────────
cos_matrix = np.load(OUT / 'ghost_cosine_matrix.npy')
with open(OUT / 'ghost_labels.json') as f:
    ghost_labels = json.load(f)

# ══════════════════════════════════════════════════════════════
# Ghost Cluster Heatmap (with reference tokens)
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

fig = go.Figure(data=go.Heatmap(
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
fig.add_hline(y=n_ghost - 0.5, line=dict(color='#34a853', width=2))
fig.add_vline(x=n_ghost - 0.5, line=dict(color='#34a853', width=2))

fig.update_layout(
    title='Ghost Cluster vs Reference Tokens — Cosine Similarity',
    width=1000, height=900,
    template='plotly_dark',
    xaxis=dict(tickangle=45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=8), autorange='reversed'),
)

fig.write_html(str(OUT / 'ghost_heatmap.html'))
print(f"Saved ghost_heatmap.html")
