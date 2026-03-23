"""
Interactive Analogy Explorer
Type three tokens, see what the embedding space predicts as the fourth.

Usage: poetry run python phase2-ghost-cluster-and-analogies/analogy_explorer.py
Then open http://localhost:8765 in your browser.
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

# Build lookup: token string -> index (try with and without space prefix)
token_lookup = {}
for i, t in enumerate(tokens):
    token_lookup[t] = i
    token_lookup[t.strip()] = i  # also index without leading space


def find_token(query):
    """Find best token match for a query string."""
    # Try exact match with space prefix first (most GPT-2 words)
    if ' ' + query in token_lookup:
        return token_lookup[' ' + query]
    if query in token_lookup:
        return token_lookup[query]
    # Case-insensitive fallback
    q = query.lower()
    for t, i in token_lookup.items():
        if t.lower() == q or t.strip().lower() == q:
            return i
    return None


def solve_analogy(a_str, b_str, c_str, k=10, dedup=True):
    """a is to b as c is to ? Returns top-k results."""
    ia = find_token(a_str)
    ib = find_token(b_str)
    ic = find_token(c_str)

    missing = []
    if ia is None: missing.append(a_str)
    if ib is None: missing.append(b_str)
    if ic is None: missing.append(c_str)
    if missing:
        return {'error': f"Token(s) not found: {', '.join(missing)}"}

    vec = emb[ib] - emb[ia] + emb[ic]
    vec_norm = np.linalg.norm(vec)
    cos = (emb @ vec) / (norms * vec_norm + 1e-10)
    for idx in (ia, ib, ic):
        cos[idx] = -2

    # Get more candidates than needed, then deduplicate
    nn = np.argsort(cos)[-(k * 5):][::-1]

    results = []
    seen = set()
    for i in nn:
        entry = {'token': labels[int(i)], 'cosine': round(float(cos[i]), 4), 'idx': int(i)}
        if dedup:
            key = labels[int(i)].strip().lower()
            if key in seen:
                continue
            seen.add(key)
        results.append(entry)
        if len(results) >= k:
            break

    return {
        'a': {'token': labels[ia], 'idx': int(ia)},
        'b': {'token': labels[ib], 'idx': int(ib)},
        'c': {'token': labels[ic], 'idx': int(ic)},
        'results': results,
    }


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Embedding Analogy Explorer — GPT-2</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1e1e1e; color: #eee; font-family: 'SF Mono', 'Consolas', monospace;
    display: flex; flex-direction: column; align-items: center; padding: 40px 20px;
  }
  h1 { font-size: 22px; margin-bottom: 8px; color: #fff; }
  .subtitle { color: #888; font-size: 13px; margin-bottom: 30px; }
  .analogy-row {
    display: flex; align-items: center; gap: 12px; margin-bottom: 20px;
    flex-wrap: wrap; justify-content: center;
  }
  input[type="text"] {
    background: #2d2d2d; color: #fff; border: 2px solid #444; padding: 10px 14px;
    border-radius: 6px; font-size: 18px; width: 150px; text-align: center;
    font-family: inherit; transition: border-color 0.2s;
  }
  input[type="text"]:focus { border-color: #4285f4; outline: none; }
  .separator { font-size: 22px; color: #666; font-weight: bold; }
  .result-box {
    background: #2d2d2d; border: 2px solid #34a853; padding: 10px 14px;
    border-radius: 6px; font-size: 18px; width: 150px; text-align: center;
    min-height: 44px; display: flex; align-items: center; justify-content: center;
    color: #34a853; font-weight: bold;
  }
  .result-box.empty { border-color: #444; color: #666; }
  button {
    background: #4285f4; color: white; border: none; padding: 10px 24px;
    border-radius: 6px; font-size: 16px; cursor: pointer; font-family: inherit;
    transition: background 0.2s;
  }
  button:hover { background: #3367d6; }
  .help { color: #666; font-size: 12px; margin-top: 8px; }
  .runners-up { margin-top: 24px; width: 100%; max-width: 500px; }
  .runners-up h3 { font-size: 14px; color: #888; margin-bottom: 10px; }
  .runner {
    display: flex; justify-content: space-between; padding: 6px 12px;
    border-radius: 4px; margin-bottom: 4px;
  }
  .runner:nth-child(odd) { background: #252525; }
  .runner .token { color: #ccc; }
  .runner .score { color: #666; }
  .runner:first-child .token { color: #34a853; font-weight: bold; }
  .runner:first-child .score { color: #34a853; }
  .bar {
    position: absolute; left: 0; top: 0; bottom: 0;
    background: #4285f4; opacity: 0.12; border-radius: 4px;
  }
  .runner { position: relative; }
  .runner .token, .runner .score { position: relative; z-index: 1; }
  .error { color: #ea4335; margin-top: 12px; font-size: 14px; }
  .presets { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
  .preset {
    background: #333; color: #aaa; border: 1px solid #444; padding: 4px 10px;
    border-radius: 4px; font-size: 11px; cursor: pointer; transition: all 0.2s;
  }
  .preset:hover { background: #444; color: #fff; border-color: #666; }
</style>
</head>
<body>
  <h1>Embedding Analogy Explorer</h1>
  <p class="subtitle">GPT-2 static embeddings — A is to B as C is to ?</p>

  <div class="analogy-row">
    <input type="text" id="a" placeholder="king" value="king">
    <span class="separator">:</span>
    <input type="text" id="b" placeholder="queen" value="queen">
    <span class="separator">::</span>
    <input type="text" id="c" placeholder="man" value="man">
    <span class="separator">:</span>
    <div class="result-box empty" id="result">?</div>
  </div>

  <div style="display:flex; align-items:center; gap:16px;">
    <button onclick="solve()">Solve</button>
    <label style="color:#999; font-size:13px; cursor:pointer;">
      <input id="dedup" type="checkbox" checked style="vertical-align:middle; cursor:pointer;"> collapse variants
    </label>
  </div>
  <p class="help">Tokens are space-prefixed in GPT-2, so "king" matches " king". Press Enter in any field to solve.</p>

  <div class="presets">
    <span class="preset" onclick="setPreset('king','queen','man')">king:queen::man:?</span>
    <span class="preset" onclick="setPreset('dog','dogs','cat')">dog:dogs::cat:?</span>
    <span class="preset" onclick="setPreset('France','Paris','Japan')">France:Paris::Japan:?</span>
    <span class="preset" onclick="setPreset('big','bigger','small')">big:bigger::small:?</span>
    <span class="preset" onclick="setPreset('good','best','bad')">good:best::bad:?</span>
    <span class="preset" onclick="setPreset('walk','walked','run')">walk:walked::run:?</span>
    <span class="preset" onclick="setPreset('Spain','Spanish','Germany')">Spain:Spanish::Germany:?</span>
    <span class="preset" onclick="setPreset('hot','cold','up')">hot:cold::up:?</span>
  </div>

  <div class="runners-up" id="runners"></div>
  <p class="error" id="error"></p>

  <script>
    function setPreset(a, b, c) {
      document.getElementById('a').value = a;
      document.getElementById('b').value = b;
      document.getElementById('c').value = c;
      solve();
    }

    async function solve() {
      var a = document.getElementById('a').value.trim();
      var b = document.getElementById('b').value.trim();
      var c = document.getElementById('c').value.trim();
      var resultEl = document.getElementById('result');
      var runnersEl = document.getElementById('runners');
      var errorEl = document.getElementById('error');

      if (!a || !b || !c) return;

      errorEl.textContent = '';
      resultEl.textContent = '...';
      resultEl.className = 'result-box empty';

      var dedup = document.getElementById('dedup').checked ? '1' : '0';
      var resp = await fetch('/solve?a=' + encodeURIComponent(a) + '&b=' + encodeURIComponent(b) + '&c=' + encodeURIComponent(c) + '&dedup=' + dedup);
      var data = await resp.json();

      if (data.error) {
        errorEl.textContent = data.error;
        resultEl.textContent = '?';
        runnersEl.innerHTML = '';
        return;
      }

      var top = data.results[0];
      resultEl.textContent = top.token;
      resultEl.className = 'result-box';

      var maxCos = data.results[0].cosine;
      var html = '<h3>Top 10 results</h3>';
      data.results.forEach(function(r, i) {
        var pct = (r.cosine / maxCos * 100).toFixed(0);
        html += '<div class="runner">';
        html += '<div class="bar" style="width:' + pct + '%"></div>';
        html += '<span class="token">' + r.token + '</span>';
        html += '<span class="score">cos=' + r.cosine.toFixed(3) + '</span>';
        html += '</div>';
      });
      runnersEl.innerHTML = html;
    }

    // Enter key triggers solve
    document.querySelectorAll('input').forEach(function(el) {
      el.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') solve();
      });
    });

    // Auto-solve on load
    solve();
  </script>
</body>
</html>"""


class AnalogyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/solve':
            params = parse_qs(parsed.query)
            a = params.get('a', [''])[0]
            b = params.get('b', [''])[0]
            c = params.get('c', [''])[0]
            dedup = params.get('dedup', ['1'])[0] == '1'
            result = solve_analogy(a, b, c, dedup=dedup)
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
        pass  # Silence request logs


port = 8765
print(f"\nAnalogy Explorer running at http://localhost:{port}")
print("Press Ctrl+C to stop.\n")
HTTPServer(('localhost', port), AnalogyHandler).serve_forever()
