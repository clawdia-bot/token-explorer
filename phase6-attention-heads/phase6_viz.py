"""
Phase 6 Visualization: Pythia L6 Attention Head Analysis
Interactive plotly HTML with:
  1. Residual stream alignment trajectory (all layers)
  2. L6 decomposition (attn vs MLP)
  3. Head properties radar (alignment, norm, entropy)
  4. Attention heatmap for demo sentence
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path

# ── Load results ──────────────────────────────────────────────
results_path = Path(__file__).parent / 'phase6_results.json'
with open(results_path) as f:
    results = json.load(f)

# ── Fig 1: Residual stream trajectory ─────────────────────────
# Hardcoded from the run output (averaged across sentences)
layers = ['emb', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'final_LN']
alignment = [0.478, 0.540, 0.537, 0.549, 0.571, 0.607, 0.923, 0.923]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Residual Stream → Output Alignment Through Layers',
        'L6 Decomposition: Who Drives the Phase Transition?',
        'L6 Head Properties',
        'L6 Head Output Norms (Energy) Across Layers'
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.1,
)

# Plot 1: Alignment trajectory
fig.add_trace(go.Scatter(
    x=layers, y=alignment,
    mode='lines+markers',
    marker=dict(size=10, color=['#636EFA']*6 + ['#EF553B', '#636EFA']),
    line=dict(width=3, color='#636EFA'),
    name='Output alignment',
    hovertemplate='%{x}: %{y:.3f}<extra></extra>',
), row=1, col=1)

# Annotate the cliff
fig.add_annotation(
    x='L6', y=0.923,
    text='Phase transition<br>+0.31 in one layer!',
    showarrow=True, arrowhead=2,
    ax=-60, ay=-40,
    font=dict(size=11, color='#EF553B'),
    row=1, col=1,
)

# Plot 2: L6 decomposition bar chart
decomp = results['decomposition']
components = ['Residual\n(skip L6)', 'Residual\n+ Attn', 'Residual\n+ MLP', 'Full L6']
values = [decomp['residual_only'], decomp['residual_attn'], decomp['residual_mlp'], decomp['full']]
colors = ['#636EFA', '#FFA15A', '#00CC96', '#EF553B']

fig.add_trace(go.Bar(
    x=components, y=values,
    marker_color=colors,
    text=[f'{v:.3f}' for v in values],
    textposition='outside',
    name='Decomposition',
    showlegend=False,
), row=1, col=2)

# Plot 3: Head properties (grouped bar)
heads = [f'H{i}' for i in range(8)]
alignments = [results['head_direct_alignment'][h] for h in heads]
ablation = [results['head_ablation'][h] * 100 for h in heads]  # scale up for visibility

# Norms from the run (averaged from two sentences at L6)
norms = [1.92, 6.03, 1.64, 0.64, 1.05, 4.19, 0.60, 7.80]

fig.add_trace(go.Bar(
    x=heads, y=alignments, name='Output Alignment',
    marker_color='#636EFA', opacity=0.8,
), row=2, col=1)

fig.add_trace(go.Bar(
    x=heads, y=[n/10 for n in norms], name='Norm/10',
    marker_color='#EF553B', opacity=0.8,
), row=2, col=1)

# Head type annotations
head_types = ['content', 'context', 'verb', 'BOS', 'BOS', 'verb', 'BOS', 'prev-tok']
for i, ht in enumerate(head_types):
    fig.add_annotation(
        x=f'H{i}', y=max(alignments[i], norms[i]/10) + 0.05,
        text=ht, showarrow=False,
        font=dict(size=9, color='gray'),
        row=2, col=1,
    )

# Plot 4: Head norms across all layers
layer_norms_sent1 = {
    'L1': [1.76, 1.35, 2.04, 1.49, 1.00, 1.25, 2.30, 0.45],
    'L2': [1.54, 1.39, 0.88, 2.44, 1.77, 1.27, 1.13, 0.98],
    'L3': [1.51, 3.35, 2.43, 1.51, 1.69, 1.96, 0.93, 2.65],
    'L4': [0.64, 0.84, 1.67, 1.48, 2.09, 0.28, 0.49, 1.09],
    'L5': [1.15, 0.70, 0.16, 0.18, 0.16, 0.79, 0.38, 0.27],
    'L6': [2.05, 5.48, 1.59, 0.57, 0.87, 4.28, 0.51, 7.45],
}

colors_8 = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
for hi in range(8):
    fig.add_trace(go.Scatter(
        x=list(layer_norms_sent1.keys()),
        y=[layer_norms_sent1[l][hi] for l in layer_norms_sent1],
        mode='lines+markers',
        name=f'H{hi}',
        line=dict(color=colors_8[hi], width=2),
        marker=dict(size=6),
        legendgroup='norms',
    ), row=2, col=2)

# ── Layout ────────────────────────────────────────────────────
fig.update_layout(
    height=900, width=1200,
    title=dict(
        text='Phase 6: Pythia-70m L6 Attention Head Analysis — The Prediction Layer',
        font=dict(size=16),
    ),
    template='plotly_white',
    showlegend=True,
    legend=dict(font=dict(size=10)),
)

fig.update_yaxes(title_text='Output Alignment', row=1, col=1)
fig.update_yaxes(title_text='Output Alignment', row=1, col=2, range=[0, 1.05])
fig.update_yaxes(title_text='Value', row=2, col=1)
fig.update_yaxes(title_text='Head Output Norm', row=2, col=2)

out = Path(__file__).parent / 'phase6_attention_heads.html'
fig.write_html(str(out), include_plotlyjs='cdn')
print(f"Saved: {out}")
