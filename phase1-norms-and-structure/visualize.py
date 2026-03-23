"""
Token Embedding Visualization — UMAP to 2D, interactive Plotly
"""
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

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
        customdata=[labels[i] for i in mask],
    ))

fig.update_layout(
    title="GPT-2 Token Embedding Space (UMAP, cosine distance)",
    width=1400,
    height=800,
    template='plotly_dark',
    legend=dict(font=dict(size=10)),
)

# Write HTML with embedded search functionality
html_str = fig.to_html(include_plotlyjs=True, full_html=True)

search_ui = """
<div id="search-box" style="position:fixed; top:10px; left:50%; transform:translateX(-50%); z-index:9999;
     background:#1e1e1e; padding:12px 16px; border-radius:8px; border:1px solid #444;
     font-family:monospace; box-shadow:0 4px 12px rgba(0,0,0,0.5);">
  <input id="token-search" type="text" placeholder="Search tokens..."
    style="background:#2d2d2d; color:#eee; border:1px solid #555; padding:6px 10px;
    border-radius:4px; width:220px; font-size:14px; font-family:monospace;"
    autocomplete="off">
  <span id="match-count" style="color:#888; font-size:11px; margin-left:8px;"></span>
  <br>
  <label style="color:#999; font-size:11px; cursor:pointer;">
    <input id="exact-match" type="checkbox" style="vertical-align:middle; cursor:pointer;"> exact match
  </label>
  <span style="color:#666; font-size:10px; margin-left:8px;">Enter to search, Esc to clear</span>
</div>
<script>
(function() {
  var input = document.getElementById('token-search');
  var exactBox = document.getElementById('exact-match');
  var countEl = document.getElementById('match-count');
  var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];

  // Store original sizes/opacities per trace
  var origSizes = [], origOpacities = [];
  plotDiv.data.forEach(function(trace) {
    origSizes.push(trace.marker.size);
    origOpacities.push(trace.marker.opacity);
  });

  function doSearch() {
    var query = input.value.toLowerCase();
    if (query === '') {
      // Reset all traces to original appearance
      var sizes = [], opacities = [];
      for (var i = 0; i < plotDiv.data.length; i++) {
        sizes.push(origSizes[i]);
        opacities.push(origOpacities[i]);
      }
      Plotly.restyle(plotDiv, {'marker.size': sizes, 'marker.opacity': opacities});
      countEl.textContent = '';
      return;
    }
    var totalMatches = 0;
    var newSizes = [], newOpacities = [];
    plotDiv.data.forEach(function(trace) {
      var cd = trace.customdata || [];
      var sz = [], op = [];
      for (var j = 0; j < cd.length; j++) {
        var val = cd[j] ? cd[j].toLowerCase() : '';
        var match = exactBox.checked ? val === query : val.indexOf(query) !== -1;
        if (match) {
          sz.push(10);
          op.push(1.0);
          totalMatches++;
        } else {
          sz.push(2);
          op.push(0.05);
        }
      }
      newSizes.push(sz);
      newOpacities.push(op);
    });
    Plotly.restyle(plotDiv, {'marker.size': newSizes, 'marker.opacity': newOpacities});
    countEl.textContent = totalMatches + ' match' + (totalMatches !== 1 ? 'es' : '');
  }

  input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') doSearch();
    if (e.key === 'Escape') { input.value = ''; doSearch(); }
  });
})();
</script>
"""

# Inject search UI before closing </body>
html_str = html_str.replace('</body>', search_ui + '</body>')

out_path = f"{OUT}/umap_embeddings.html"
with open(out_path, 'w') as f:
    f.write(html_str)
print(f"Saved to {out_path}")

print("Done!")
