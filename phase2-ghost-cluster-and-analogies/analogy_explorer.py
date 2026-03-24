"""
Interactive Analogy Explorer
Type three tokens, see what the embedding space predicts as the fourth.
Supports model switching via dropdown.

Usage: poetry run python phase2-ghost-cluster-and-analogies/analogy_explorer.py [--model MODEL] [--port PORT]
Then open http://localhost:8765 in your browser.
"""

import argparse
import numpy as np
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.models import load_model, add_model_arg, resolve_token, MODEL_REGISTRY

parser = argparse.ArgumentParser(description="Interactive analogy explorer")
add_model_arg(parser)
parser.add_argument('--port', type=int, default=8765, help="Server port (default: 8765)")
args = parser.parse_args()

# Global state — replaced when model is switched
state = {}


def load_into_state(slug):
    m = load_model(slug)
    state['model'] = m
    state['slug'] = slug


load_into_state(args.model)


def solve_analogy(a_str, b_str, c_str, k=10, dedup=True):
    """a is to b as c is to ? Returns top-k results."""
    m = state['model']

    ia = resolve_token(m, a_str)
    ib = resolve_token(m, b_str)
    ic = resolve_token(m, c_str)

    missing = []
    if ia is None: missing.append(a_str)
    if ib is None: missing.append(b_str)
    if ic is None: missing.append(c_str)
    if missing:
        return {'error': f"Token(s) not found: {', '.join(missing)}"}

    vec = m.emb[ib] - m.emb[ia] + m.emb[ic]
    vec_norm = np.linalg.norm(vec)
    cos = (m.emb @ vec) / (m.norms * vec_norm + 1e-10)
    for idx in (ia, ib, ic):
        cos[idx] = -2

    nn = np.argsort(cos)[-(k * 5):][::-1]

    results = []
    seen = set()
    for i in nn:
        entry = {'token': m.labels[int(i)], 'cosine': round(float(cos[i]), 4), 'idx': int(i)}
        if dedup:
            key = m.labels[int(i)].strip().lower()
            if key in seen:
                continue
            seen.add(key)
        results.append(entry)
        if len(results) >= k:
            break

    return {
        'a': {'token': m.labels[ia], 'idx': int(ia)},
        'b': {'token': m.labels[ib], 'idx': int(ib)},
        'c': {'token': m.labels[ic], 'idx': int(ic)},
        'results': results,
    }


def get_presets():
    """Return preset analogies that work with the current model."""
    m = state['model']
    all_presets = [
        ('king', 'queen', 'man'),
        ('dog', 'dogs', 'cat'),
        ('France', 'Paris', 'Japan'),
        ('big', 'bigger', 'small'),
        ('good', 'best', 'bad'),
        ('walk', 'walked', 'run'),
        ('Spain', 'Spanish', 'Germany'),
        ('hot', 'cold', 'up'),
    ]
    valid = []
    for a, b, c in all_presets:
        if all(resolve_token(m, t) is not None for t in (a, b, c)):
            valid.append({'a': a, 'b': b, 'c': c, 'label': f'{a}:{b}::{c}:?'})
    return valid


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Embedding Analogy Explorer</title>
<style>
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
    border-radius: 4px; margin-bottom: 4px; position: relative;
  }
  .runner:nth-child(odd) { background: #252525; }
  .runner .token { color: #ccc; position: relative; z-index: 1; }
  .runner .score { color: #666; position: relative; z-index: 1; }
  .runner:first-child .token { color: #34a853; font-weight: bold; }
  .runner:first-child .score { color: #34a853; }
  .bar {
    position: absolute; left: 0; top: 0; bottom: 0;
    background: #4285f4; opacity: 0.12; border-radius: 4px;
  }
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
  <p class="subtitle">A is to B as C is to ?</p>

  <div class="model-bar">
    <label style="color:#999; font-size:13px;">Model:</label>
    <select id="model-select" onchange="switchModel()"></select>
    <span id="model-status" class="status"></span>
  </div>

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
  <p class="help">Press Enter in any field to solve.</p>

  <div class="presets" id="presets"></div>

  <div class="runners-up" id="runners"></div>
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
      solve();
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
        solve();
      } else {
        statusEl.textContent = 'Error: ' + (data.error || 'unknown');
      }
    }

    async function loadPresets() {
      var resp = await fetch('/presets');
      var data = await resp.json();
      var el = document.getElementById('presets');
      el.innerHTML = '';
      data.forEach(function(p) {
        var span = document.createElement('span');
        span.className = 'preset';
        span.textContent = p.label;
        span.onclick = function() { setPreset(p.a, p.b, p.c); };
        el.appendChild(span);
      });
    }

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
        html += '<span class="token">' + escHtml(r.token) + '</span>';
        html += '<span class="score">cos=' + r.cosine.toFixed(3) + '</span>';
        html += '</div>';
      });
      runnersEl.innerHTML = html;
    }

    function escHtml(s) {
      var d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }

    document.querySelectorAll('input[type="text"]').forEach(function(el) {
      el.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') solve();
      });
    });

    init();
  </script>
</body>
</html>"""


class AnalogyHandler(BaseHTTPRequestHandler):
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

        elif parsed.path == '/solve':
            params = parse_qs(parsed.query)
            a = params.get('a', [''])[0]
            b = params.get('b', [''])[0]
            c = params.get('c', [''])[0]
            dedup = params.get('dedup', ['1'])[0] == '1'
            result = solve_analogy(a, b, c, dedup=dedup)
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
print(f"\nAnalogy Explorer running at http://localhost:{port}")
print(f"Current model: {state['model'].name}")
print("Switch models via the dropdown in the browser.")
print("Press Ctrl+C to stop.\n")
HTTPServer(('localhost', port), AnalogyHandler).serve_forever()
