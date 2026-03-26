"""
Interactive Nearest Neighbor Explorer
Type a token, see its nearest neighbors in embedding space.
Supports model switching via dropdown.

Usage: poetry run python phase2-ghost-cluster-and-analogies/neighbor_explorer.py [--model MODEL] [--port PORT]
Then open http://localhost:8766 in your browser.
"""

import argparse
import numpy as np
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.models import load_model, add_model_arg, resolve_token_loose, MODEL_REGISTRY

parser = argparse.ArgumentParser(description="Interactive nearest neighbor explorer")
add_model_arg(parser)
parser.add_argument('--port', type=int, default=8766, help="Server port (default: 8766)")
args = parser.parse_args()

# Global state
state = {'cache': {}}


def load_into_state(slug):
    if slug in state['cache']:
        m = state['cache'][slug]
    else:
        m = load_model(slug)
        state['cache'][slug] = m
    state['model'] = m
    state['slug'] = slug


load_into_state(args.model)


def get_neighbors(query, k=15, dedup=True):
    m = state['model']
    idx = resolve_token_loose(m, query)
    if idx is None:
        return {'error': f"Token not found: {query}"}

    cos = m.normed_emb @ m.normed_emb[idx]
    cos[idx] = -2
    nn = np.argsort(cos)[-(k * 5):][::-1]

    neighbors = []
    seen = set()
    for i in nn:
        entry = {
            'token': m.labels[int(i)],
            'cosine': round(float(cos[i]), 4),
            'norm': round(float(m.norms[int(i)]), 4),
            'idx': int(i),
        }
        if dedup:
            key = m.labels[int(i)].strip().lower()
            if key in seen:
                continue
            seen.add(key)
        neighbors.append(entry)
        if len(neighbors) >= k:
            break

    return {
        'center': {
            'token': m.labels[idx],
            'idx': int(idx),
            'norm': round(float(m.norms[idx]), 4),
        },
        'neighbors': neighbors,
    }


def get_presets():
    """Return preset tokens that exist in the current model."""
    m = state['model']
    candidates = ['the', 'king', 'Python', 'dog', 'happy', 'France',
                   '0', 'she', 'war', 'music', 'water', 'hello']
    return [t for t in candidates if resolve_token_loose(m, t) is not None]


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="dark">
<meta name="theme-color" content="#1e1e1e">
<title>Nearest Neighbor Explorer</title>
<style>
  :root { color-scheme: dark; }
  html { background: #1e1e1e; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1e1e1e; color: #eee; font-family: 'SF Mono', 'Consolas', monospace;
    display: flex; flex-direction: column; align-items: center; padding: 40px 20px;
  }
  h1 { font-size: 22px; margin-bottom: 8px; color: #fff; }
  .subtitle { color: #888; font-size: 13px; margin-bottom: 6px; }
  .model-bar {
    display: flex; align-items: center; gap: 10px; margin-bottom: 24px;
  }
  .model-bar select {
    background: #2d2d2d; color: #fff; border: 1px solid #555; padding: 6px 10px;
    border-radius: 4px; font-size: 13px; font-family: inherit; cursor: pointer;
  }
  .model-bar .status { color: #34a853; font-size: 12px; }
  .model-bar .loading { color: #fbbc04; font-size: 12px; }
  .search-row {
    display: flex; align-items: center; gap: 12px; margin-bottom: 16px;
  }
  input[type="text"] {
    background: #2d2d2d; color: #fff; border: 2px solid #444; padding: 10px 14px;
    border-radius: 6px; font-size: 18px; width: 250px; text-align: center;
    font-family: inherit; transition: border-color 0.2s;
  }
  input[type="text"]:focus { border-color: #4285f4; outline: none; }
  button {
    background: #4285f4; color: white; border: none; padding: 10px 24px;
    border-radius: 6px; font-size: 16px; cursor: pointer; font-family: inherit;
    transition: background 0.2s;
  }
  button:hover { background: #3367d6; }
  .help { color: #666; font-size: 12px; margin-bottom: 8px; }
  .presets { margin-bottom: 24px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
  .preset {
    background: #333; color: #aaa; border: 1px solid #444; padding: 4px 10px;
    border-radius: 4px; font-size: 11px; cursor: pointer; transition: all 0.2s;
  }
  .preset:hover { background: #444; color: #fff; border-color: #666; }
  .center-info { margin-bottom: 20px; text-align: center; }
  .center-token { font-size: 28px; color: #fbbc04; font-weight: bold; }
  .center-meta { font-size: 12px; color: #888; margin-top: 4px; }
  .results { width: 100%; max-width: 550px; }
  .neighbor {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 14px; margin-bottom: 3px; border-radius: 4px; position: relative;
  }
  .neighbor:nth-child(odd) { background: #252525; }
  .bar {
    position: absolute; left: 0; top: 0; bottom: 0;
    background: #4285f4; opacity: 0.12; border-radius: 4px;
  }
  .neighbor .token, .neighbor .meta { position: relative; z-index: 1; }
  .neighbor .token { color: #ccc; font-size: 15px; }
  .neighbor:first-child .token { color: #7baaf7; font-weight: bold; }
  .neighbor .meta { color: #666; font-size: 12px; }
  .rank { color: #555; font-size: 11px; width: 24px; text-align: right; margin-right: 8px; position: relative; z-index: 1; }
  .error { color: #ea4335; margin-top: 12px; font-size: 14px; }
</style>
</head>
<body>
  <h1>Nearest Neighbor Explorer</h1>
  <p class="subtitle">Find a token's closest neighbors by cosine similarity</p>
  <p class="subtitle" style="margin-bottom:18px;">Uses heuristic token lookup for interactive exploration, not the strict research probe set.</p>

  <div class="model-bar">
    <label style="color:#999; font-size:13px;">Model:</label>
    <select id="model-select" onchange="switchModel()"></select>
    <span id="model-status" class="status"></span>
  </div>

  <div class="search-row">
    <input type="text" id="query" placeholder="type a token..." value="king">
    <button onclick="search()">Search</button>
    <label style="color:#999; font-size:13px; cursor:pointer;">
      <input id="dedup" type="checkbox" checked style="vertical-align:middle; cursor:pointer;"> collapse variants
    </label>
  </div>
  <p class="help">Press Enter to search.</p>

  <div class="presets" id="presets"></div>

  <div class="center-info" id="center-info"></div>
  <div class="results" id="results"></div>
  <p class="error" id="error"></p>

  <script>
    var currentModel = '';

    async function init() {
      var resp = await fetch('/models');
      var data = await resp.json();
      var sel = document.getElementById('model-select');
      sel.innerHTML = '';
      data.models.forEach(function(m) {
        var opt = document.createElement('option');
        opt.value = m.slug;
        opt.textContent = m.name;
        if (m.slug === data.current) opt.selected = true;
        sel.appendChild(opt);
      });
      currentModel = data.current;
      document.getElementById('model-status').textContent = data.current_name + ' loaded';
      loadPresets();
      search();
    }

    async function switchModel() {
      var slug = document.getElementById('model-select').value;
      if (slug === currentModel) return;
      var statusEl = document.getElementById('model-status');
      statusEl.textContent = 'Loading ' + slug + '...';
      statusEl.className = 'loading';
      var resp = await fetch('/switch?model=' + encodeURIComponent(slug));
      var data = await resp.json();
      if (data.ok) {
        currentModel = slug;
        statusEl.textContent = data.name + ' loaded';
        statusEl.className = 'status';
        loadPresets();
        search();
      } else {
        statusEl.textContent = 'Error: ' + (data.error || 'unknown');
      }
    }

    async function loadPresets() {
      var resp = await fetch('/presets');
      var presets = await resp.json();
      var el = document.getElementById('presets');
      el.innerHTML = '';
      presets.forEach(function(t) {
        var span = document.createElement('span');
        span.className = 'preset';
        span.textContent = t;
        span.onclick = function() { go(t); };
        el.appendChild(span);
      });
    }

    function go(token) {
      document.getElementById('query').value = token;
      search();
    }

    async function search() {
      var query = document.getElementById('query').value.trim();
      var centerEl = document.getElementById('center-info');
      var resultsEl = document.getElementById('results');
      var errorEl = document.getElementById('error');

      if (!query) return;

      errorEl.textContent = '';
      centerEl.innerHTML = '';
      resultsEl.innerHTML = '<div style="color:#666; text-align:center;">searching...</div>';

      var dedup = document.getElementById('dedup').checked ? '1' : '0';
      var resp = await fetch('/search?q=' + encodeURIComponent(query) + '&dedup=' + dedup);
      var data = await resp.json();

      if (data.error) {
        errorEl.textContent = data.error;
        centerEl.innerHTML = '';
        resultsEl.innerHTML = '';
        return;
      }

      centerEl.innerHTML =
        '<div class="center-token">' + escHtml(data.center.token) + '</div>' +
        '<div class="center-meta">idx=' + data.center.idx + ' &middot; norm=' + data.center.norm + '</div>';

      var maxCos = data.neighbors[0].cosine;
      var html = '';
      data.neighbors.forEach(function(nb, i) {
        var pct = (nb.cosine / maxCos * 100).toFixed(0);
        html += '<div class="neighbor">';
        html += '<div class="bar" style="width:' + pct + '%"></div>';
        html += '<span class="rank">' + (i + 1) + '.</span>';
        html += '<span class="token">' + escHtml(nb.token) + '</span>';
        html += '<span class="meta">cos=' + nb.cosine.toFixed(3) + ' &middot; norm=' + nb.norm.toFixed(3) + '</span>';
        html += '</div>';
      });
      resultsEl.innerHTML = html;
    }

    function escHtml(s) {
      var d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }

    document.getElementById('query').addEventListener('keydown', function(e) {
      if (e.key === 'Enter') search();
    });

    init();
  </script>
</body>
</html>"""


class NeighborHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/models':
            models = [{'slug': s, 'name': n} for s, (n, _) in MODEL_REGISTRY.items()]
            resp = {
                'models': models,
                'current': state['slug'],
                'current_name': state['model'].name,
            }
            self._json_response(resp)

        elif parsed.path == '/switch':
            params = parse_qs(parsed.query)
            slug = params.get('model', [''])[0]
            try:
                load_into_state(slug)
                self._json_response({'ok': True, 'name': state['model'].name})
            except Exception as e:
                self._json_response({'ok': False, 'error': str(e)})

        elif parsed.path == '/presets':
            self._json_response(get_presets())

        elif parsed.path == '/search':
            params = parse_qs(parsed.query)
            q = params.get('q', [''])[0]
            dedup = params.get('dedup', ['1'])[0] == '1'
            result = get_neighbors(q, dedup=dedup)
            self._json_response(result)

        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

    def _json_response(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass


port = args.port
print(f"\nNeighbor Explorer running at http://localhost:{port}")
print(f"Current model: {state['model'].name}")
print("Switch models via the dropdown in the browser.")
print("Press Ctrl+C to stop.\n")
HTTPServer(('localhost', port), NeighborHandler).serve_forever()
