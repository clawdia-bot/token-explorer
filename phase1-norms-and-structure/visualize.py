"""
Token Embedding Visualization — UMAP to 2D, interactive Plotly
"""
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import json

import os
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
from tokenutils import token_display, categorize

print("Loading embeddings...")
path = hf_hub_download('gpt2', 'pytorch_model.bin')
sd = torch.load(path, map_location='cpu', weights_only=False)
emb = sd['wte.weight'].numpy()
tok = AutoTokenizer.from_pretrained('gpt2')
labels = [token_display(tok, i) for i in range(emb.shape[0])]
norms = np.linalg.norm(emb, axis=1)

categories = [categorize(tok, i) for i in range(emb.shape[0])]

print("Running UMAP...")
from umap import UMAP

reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42, verbose=False)
coords = reducer.fit_transform(emb)

print("Building plotly figure...")
import plotly.graph_objects as go

# Color map
cat_colors = {
    'word': '#4285f4',
    'alpha_fragment': '#7baaf7',
    'number': '#ea4335',
    'punctuation': '#fbbc04',
    'whitespace': '#34a853',
    'control_char': '#9e9e9e',
    'byte_token': '#bdbdbd',
    'japanese': '#ff6d00',
    'cjk': '#ff9100',
    'cyrillic': '#ab47bc',
    'arabic': '#26a69a',
    'hebrew': '#66bb6a',
    'greek': '#7e57c2',
    'korean': '#ef5350',
    'allcaps': '#0d47a1',
    'other': '#757575',
}

fig = go.Figure()

for cat in sorted(set(categories)):
    mask = [i for i, c in enumerate(categories) if c == cat]
    hover_text = [f"[{i}] {repr(labels[i])}<br>norm={norms[i]:.3f}<br>cat={cat}" for i in mask]
    fig.add_trace(go.Scattergl(
        x=coords[mask, 0],
        y=coords[mask, 1],
        mode='markers',
        marker=dict(
            size=4,
            color=cat_colors.get(cat, '#757575'),
            opacity=0.7,
        ),
        name=f"{cat} ({len(mask)})",
        text=hover_text,
        hoverinfo='text',
    ))

fig.update_layout(
    title="GPT-2 Token Embedding Space (UMAP, cosine distance)",
    width=1400,
    height=1000,
    template='plotly_dark',
    legend=dict(font=dict(size=10)),
)

fig.write_html(f"{OUT}/umap_embeddings.html")
print(f"Saved to {OUT}/umap_embeddings.html")

# Also save coords for later analysis
np.save(f"{OUT}/umap_coords.npy", coords)
np.save(f"{OUT}/categories.npy", np.array(categories))
print("Done!")
