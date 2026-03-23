"""
Interactive Nearest Neighbor Explorer
Type a token, see its nearest neighbors in embedding space.

Usage: poetry run python phase2-ghost-cluster-and-analogies/neighbor_explorer.py
Then open http://localhost:8766 in your browser.
"""

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(OUT, '..', 'phase1-norms-and-structure'))
from tokenutils import token_display

print("Loading GPT-2 embeddings...")
path = hf_hub_download('gpt2', 'pytorch_model.bin')
sd = torch.load(path, map_location='cpu', weights_only=False)
emb = sd['wte.weight'].numpy()
tok = AutoTokenizer.from_pretrained('gpt2')
tokens = [tok.decode([i]) for i in range(emb.shape[0])]
labels = [token_display(tok, i) for i in range(emb.shape[0])]
norms = np.linalg.norm(emb, axis=1)
normed_emb = emb / (norms[:, None] + 1e-10)

# Build lookup
token_lookup = {}
for i, t in enumerate(tokens):
    token_lookup[t] = i
    token_lookup[t.strip()] = i


def find_token(query):
    if ' ' + query in token_lookup:
        return token_lookup[' ' + query]
    if query in token_lookup:
        return token_lookup[query]
    q = query.lower()
    for t, i in token_lookup.items():
        if t.lower() == q or t.strip().lower() == q:
            return i
    return None


def get_neighbors(query, k=15, dedup=True):
    idx = find_token(query)
    if idx is None:
        return {'error': f"Token not found: {query}"}

    cos = normed_emb @ normed_emb[idx]
    cos[idx] = -2
    nn = np.argsort(cos)[-(k * 5):][::-1]

    neighbors = []
    seen = set()
    for i in nn:
        entry = {
            'token': labels[int(i)],
            'cosine': round(float(cos[i]), 4),
            'norm': round(float(norms[int(i)]), 4),
            'idx': int(i),
        }
        if dedup:
            key = labels[int(i)].strip().lower()
            if key in seen:
                continue
            seen.add(key)
        neighbors.append(entry)
        if len(neighbors) >= k:
            break

    return {
        'center': {
            'token': labels[idx],
            'idx': int(idx),
            'norm': round(float(norms[idx]), 4),
        },
        'neighbors': neighbors,
    }


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Nearest Neighbor Explorer — GPT-2</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1e1e1e; color: #eee; font-family: 'SF Mono', 'Consolas', monospace;
    display: flex; flex-direction: column; align-items: center; padding: 40px 20px;
  }
  h1 { font-size: 22px; margin-bottom: 8px; color: #fff; }
  .subtitle { color: #888; font-size: 13px; margin-bottom: 30px; }
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
  .center-info {
    margin-bottom: 20px; text-align: center;
  }
  .center-token { font-size: 28px; color: #fbbc04; font-weight: bold; }
  .center-meta { font-size: 12px; color: #888; margin-top: 4px; }
  .results { width: 100%; max-width: 550px; }
  .neighbor {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 14px; margin-bottom: 3px; border-radius: 4px;
    position: relative;
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
  .neighbor .cosine { color: #888; }
  .rank { color: #555; font-size: 11px; width: 24px; text-align: right; margin-right: 8px; position: relative; z-index: 1; }
  .error { color: #ea4335; margin-top: 12px; font-size: 14px; }
</style>
</head>
<body>
  <h1>Nearest Neighbor Explorer</h1>
  <p class="subtitle">GPT-2 static embeddings — find a token's closest neighbors by cosine similarity</p>

  <div class="search-row">
    <input type="text" id="query" placeholder="type a token..." value="king">
    <button onclick="search()">Search</button>
    <label style="color:#999; font-size:13px; cursor:pointer;">
      <input id="dedup" type="checkbox" checked style="vertical-align:middle; cursor:pointer;"> collapse variants
    </label>
  </div>
  <p class="help">Tokens are space-prefixed in GPT-2, so "king" matches " king". Press Enter to search.</p>

  <div class="presets">
    <span class="preset" onclick="go('the')">the</span>
    <span class="preset" onclick="go('king')">king</span>
    <span class="preset" onclick="go('Python')">Python</span>
    <span class="preset" onclick="go('dog')">dog</span>
    <span class="preset" onclick="go('happy')">happy</span>
    <span class="preset" onclick="go('France')">France</span>
    <span class="preset" onclick="go('\\n')">\\n</span>
    <span class="preset" onclick="go('SPONSORED')">SPONSORED</span>
    <span class="preset" onclick="go('0')">0</span>
    <span class="preset" onclick="go('she')">she</span>
    <span class="preset" onclick="go('war')">war</span>
    <span class="preset" onclick="go('music')">music</span>
  </div>

  <div class="center-info" id="center-info"></div>
  <div class="results" id="results"></div>
  <p class="error" id="error"></p>

  <script>
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

    search();
  </script>
</body>
</html>"""


class NeighborHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/search':
            params = parse_qs(parsed.query)
            q = params.get('q', [''])[0]
            dedup = params.get('dedup', ['1'])[0] == '1'
            result = get_neighbors(q, dedup=dedup)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

    def log_message(self, format, *args):
        pass


port = 8766
print(f"\nNeighbor Explorer running at http://localhost:{port}")
print("Press Ctrl+C to stop.\n")
HTTPServer(('localhost', port), NeighborHandler).serve_forever()
