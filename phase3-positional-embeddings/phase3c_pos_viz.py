"""
Phase 3c: Visualize positional embedding structure.
"""
import torch
import numpy as np
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Loading...")
path = hf_hub_download('gpt2', 'pytorch_model.bin')
sd = torch.load(path, map_location='cpu', weights_only=False)
pos = sd['wpe.weight'].numpy()
norms_pos = np.linalg.norm(pos, axis=1)

# PCA of positions
pos_centered = pos - pos.mean(axis=0)
U, S, Vt = np.linalg.svd(pos_centered, full_matrices=False)
pc_coords = U * S  # [1024, 768]

# Build multi-panel figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Position Norms (U-shaped!)",
        "PC1 vs PC2 (colored by position)",
        "PC2 vs PC3 (the helix)",
        "Cosine Similarity Heatmap"
    ],
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "heatmap"}]]
)

positions = np.arange(1024)
colors = positions

# 1. Norm plot
fig.add_trace(go.Scatter(
    x=positions, y=norms_pos,
    mode='markers', marker=dict(size=3, color=colors, colorscale='Viridis'),
    name='Norm', hovertext=[f"pos {i}: {norms_pos[i]:.3f}" for i in range(1024)]
), row=1, col=1)

# 2. PC1 vs PC2
fig.add_trace(go.Scatter(
    x=pc_coords[:, 0], y=pc_coords[:, 1],
    mode='markers', marker=dict(size=4, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(title="Position")),
    name='PC1-PC2', hovertext=[f"pos {i}" for i in range(1024)]
), row=1, col=2)

# 3. PC2 vs PC3
fig.add_trace(go.Scatter(
    x=pc_coords[:, 1], y=pc_coords[:, 2],
    mode='markers', marker=dict(size=4, color=colors, colorscale='Viridis'),
    name='PC2-PC3', hovertext=[f"pos {i}" for i in range(1024)]
), row=2, col=1)

# 4. Cosine heatmap (subsample every 10th)
step = 10
sample = list(range(0, 1024, step))
sample_embs = pos[sample]
sample_norms = norms_pos[sample]
normed = sample_embs / (sample_norms[:, None] + 1e-10)
cos_mat = normed @ normed.T

fig.add_trace(go.Heatmap(
    z=cos_mat, x=sample, y=sample,
    colorscale='RdBu', zmid=0,
    hovertemplate="pos %{x} vs pos %{y}: %{z:.3f}<extra></extra>"
), row=2, col=2)

fig.update_layout(
    title="GPT-2 Positional Embeddings: Geometry",
    height=900, width=1100,
    showlegend=False,
    template="plotly_dark"
)

fig.write_html("positional_embeddings.html")
print("Saved positional_embeddings.html")

# Also save a 3D PCA view
fig3d = go.Figure(data=[go.Scatter3d(
    x=pc_coords[:, 0], y=pc_coords[:, 1], z=pc_coords[:, 2],
    mode='markers+lines',
    marker=dict(size=3, color=colors, colorscale='Viridis', opacity=0.8),
    line=dict(color='rgba(255,255,255,0.2)', width=1),
    hovertext=[f"pos {i}" for i in range(1024)]
)])
fig3d.update_layout(
    title="GPT-2 Positional Embeddings: 3D PCA (the path through space)",
    scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
    template="plotly_dark", height=800, width=1000
)
fig3d.write_html("positional_3d.html")
print("Saved positional_3d.html")
print("✅ Done")
