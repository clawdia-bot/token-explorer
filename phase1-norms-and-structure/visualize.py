"""
Token Embedding Visualization — UMAP to 2D, interactive Plotly

Caches per-model UMAP coordinates so the expensive projection only has to be
computed once for a given model/sample combination.

Usage: poetry run python phase1-norms-and-structure/visualize.py [--model MODEL] [--sample N] [--force] [--all]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.html import inject_dark_mode
from common.models import add_model_arg, load_model, model_slugs
from common.tokenutils import categorize


CAT_COLORS = {
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


def build_html(coords, labels, norms, categories, indices, model_name, sample_note):
    import plotly.graph_objects as go

    fig = go.Figure()
    for cat in sorted(set(categories)):
        mask = [j for j, value in enumerate(categories) if value == cat]
        hover_text = [
            f"[{indices[j]}] {repr(labels[j])}<br>norm={norms[j]:.3f}<br>cat={cat}"
            for j in mask
        ]
        fig.add_trace(go.Scattergl(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode='markers',
            marker=dict(
                size=4,
                color=CAT_COLORS.get(cat, '#757575'),
                opacity=0.7,
            ),
            name=f"{cat} ({len(mask)})",
            text=hover_text,
            hoverinfo='text',
            customdata=[labels[j] for j in mask],
        ))

    fig.update_layout(
        title=f"{model_name} Token Embedding Space (UMAP, cosine distance){sample_note}",
        width=1400,
        height=800,
        template='plotly_dark',
        legend=dict(font=dict(size=10)),
    )

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

  var origSizes = [], origOpacities = [];
  plotDiv.data.forEach(function(trace) {
    origSizes.push(trace.marker.size);
    origOpacities.push(trace.marker.opacity);
  });

  function doSearch() {
    var query = input.value.toLowerCase();
    if (query === '') {
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
    return inject_dark_mode(html_str.replace('</body>', search_ui + '</body>'))


def cache_paths(out_dir):
    return {
        'meta': os.path.join(out_dir, 'umap_cache_meta.json'),
        'arrays': os.path.join(out_dir, 'umap_cache_arrays.npz'),
        'html': os.path.join(out_dir, 'umap_embeddings.html'),
    }


def load_cache(paths, requested_sample):
    if not (os.path.exists(paths['meta']) and os.path.exists(paths['arrays'])):
        return None

    with open(paths['meta']) as f:
        meta = json.load(f)
    if meta.get('sample') != requested_sample:
        return None

    cached = np.load(paths['arrays'], allow_pickle=True)
    return {
        'model_name': meta['model_name'],
        'coords': cached['coords'],
        'indices': cached['indices'],
        'norms': cached['norms'],
        'labels': cached['labels'].tolist(),
        'categories': cached['categories'].tolist(),
    }


def save_cache(paths, sample, model_name, coords, indices, norms, labels, categories):
    with open(paths['meta'], 'w') as f:
        json.dump({'model_name': model_name, 'sample': sample, 'point_count': int(len(indices))}, f, indent=2)
    np.savez_compressed(
        paths['arrays'],
        coords=coords,
        indices=np.asarray(indices),
        norms=np.asarray(norms),
        labels=np.asarray(labels, dtype=object),
        categories=np.asarray(categories, dtype=object),
    )


def generate_umap(model_slug: str, sample: int, force: bool):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', model_slug)
    os.makedirs(out_dir, exist_ok=True)
    paths = cache_paths(out_dir)

    cached = None if force else load_cache(paths, sample)
    if cached is not None:
        print(f"Using cached UMAP for {model_slug} (sample={sample or 'full'})")
        coords = cached['coords']
        indices = cached['indices']
        norms = cached['norms']
        labels = cached['labels']
        categories = cached['categories']
        model_name = cached['model_name']
    else:
        m = load_model(model_slug)
        emb, tok, labels, norms = m.emb, m.tokenizer, m.labels, m.norms
        categories = [categorize(tok, i) for i in range(m.vocab_size)]

        if sample and sample < m.vocab_size:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(m.vocab_size, sample, replace=False)
            sample_idx.sort()
            emb = emb[sample_idx]
            labels = [labels[i] for i in sample_idx]
            norms = norms[sample_idx]
            categories = [categories[i] for i in sample_idx]
            indices = sample_idx
            print(f"Subsampled {sample} tokens from {m.vocab_size}")
        else:
            indices = np.arange(m.vocab_size)

        print("Running UMAP...")
        from umap import UMAP
        reducer = UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
            verbose=False,
        )
        coords = reducer.fit_transform(emb)
        model_name = m.name
        save_cache(paths, sample, model_name, coords, indices, norms, labels, categories)

    print("Building plotly figure...")
    sample_note = f" (sampled {sample})" if sample else ""
    html_str = build_html(coords, labels, norms, categories, indices, model_name, sample_note)

    with open(paths['html'], 'w') as f:
        f.write(html_str)
    print(f"Saved to {paths['html']}")


parser = argparse.ArgumentParser(description="Phase 1: UMAP embedding visualization")
add_model_arg(parser)
parser.add_argument(
    '--sample', type=int, default=0,
    help="Subsample N tokens for UMAP; 0 means use the full vocabulary",
)
parser.add_argument(
    '--force', action='store_true',
    help="Recompute UMAP even if a matching cache already exists",
)
parser.add_argument(
    '--all', action='store_true',
    help="Generate or reuse cached UMAP views for every supported model",
)
args = parser.parse_args()

targets = model_slugs() if args.all else [args.model]
for i, slug in enumerate(targets):
    if i:
        print()
    print(f"=== {slug} ===")
    generate_umap(slug, sample=args.sample, force=args.force)
