"""
Phase 3B browser for switching between precomputed RoPE outputs.

Usage: poetry run python phase3-positional-embeddings/phase3b_rope_browser.py
"""

import json
from pathlib import Path


BASE = Path(__file__).parent
RESULTS = BASE / 'rope-results'
OUT = BASE / 'phase3b_rope_browser.html'


models = []
for model_dir in sorted(RESULTS.iterdir() if RESULTS.exists() else []):
    if not model_dir.is_dir():
        continue
    results_path = model_dir / 'results.json'
    if not results_path.exists():
        continue

    with open(results_path) as f:
        results = json.load(f)

    compatible = results.get('compatible', False)
    model = results.get('model', {})
    rope = results.get('rope', {}) if compatible else {}
    qk = results.get('qk_rotation', {}) if compatible else {}
    positions = qk.get('positions', [])
    last_pos = None if not positions else str(positions[-1])

    models.append({
        'slug': model_dir.name,
        'name': model.get('name', model_dir.name),
        'compatible': compatible,
        'position_type': model.get('position_type'),
        'skip_reason': results.get('skip_reason'),
        'charts': (model_dir / 'phase3b_rope_charts.html').exists(),
        'theta': rope.get('theta'),
        'rotary_dim': rope.get('rotary_dim'),
        'kernel_end': None if not compatible else results['relative_kernel']['gaps'][last_pos],
        'query_drift_end': None if not compatible else qk['mean_query_drift_by_position'][last_pos],
        'score_spearman_end': None if not compatible else qk['score_spearman_by_position'][last_pos],
    })

if not models:
    raise FileNotFoundError("No Phase 3B results found. Run phase3b_rope.py first.")

payload = json.dumps(models)
first = models[0]['slug']

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="dark">
<meta name="theme-color" content="#151515">
<title>Phase 3B Browser</title>
<style>
  :root {{ color-scheme: dark; }}
  html {{ background:#151515; }}
  body {{ background:#151515; color:#eee; font-family:'SF Mono','Consolas',monospace; margin:0; padding:24px; }}
  h1 {{ margin:0 0 8px 0; font-size:28px; }}
  .sub {{ color:#888; margin-bottom:20px; }}
  .toolbar {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:18px; }}
  select, button {{ background:#262626; color:#eee; border:1px solid #444; border-radius:6px; padding:8px 10px; font:inherit; }}
  .summary {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px; }}
  .card {{ background:#202020; border:1px solid #333; border-radius:8px; padding:12px 14px; min-width:180px; }}
  .label {{ color:#8a8a8a; font-size:12px; margin-bottom:4px; }}
  .value {{ font-size:20px; color:#fff; }}
  .panel {{ background:#202020; border:1px solid #333; border-radius:8px; overflow:hidden; }}
  .panel-header {{ padding:10px 12px; border-bottom:1px solid #333; color:#aaa; }}
  iframe {{ width:100%; height:760px; border:0; background:#111; }}
  .empty {{ padding:24px; color:#888; }}
</style>
</head>
<body>
  <h1>Phase 3B Browser</h1>
  <div class="sub">Compare RoPE geometry and first-layer positional drift across saved models.</div>
  <div class="toolbar">
    <label for="model-select">Model:</label>
    <select id="model-select"></select>
    <button id="open-results">Open results.json</button>
    <button id="open-charts">Open charts in new tab</button>
  </div>
  <div class="summary">
    <div class="card"><div class="label">Position type</div><div class="value" id="position-type">-</div></div>
    <div class="card"><div class="label">RoPE theta</div><div class="value" id="theta">-</div></div>
    <div class="card"><div class="label">Rotary dim</div><div class="value" id="rotary-dim">-</div></div>
    <div class="card"><div class="label">End score corr</div><div class="value" id="score-end">-</div></div>
  </div>
  <div class="panel">
    <div class="panel-header">Phase 3B charts</div>
    <div id="missing" class="empty" style="display:none;"></div>
    <iframe id="charts-frame"></iframe>
  </div>
<script>
const models = {payload};
const bySlug = Object.fromEntries(models.map(m => [m.slug, m]));
const select = document.getElementById('model-select');
const chartsFrame = document.getElementById('charts-frame');
const missing = document.getElementById('missing');

models.forEach(model => {{
  const opt = document.createElement('option');
  opt.value = model.slug;
  opt.textContent = model.name;
  select.appendChild(opt);
}});

function update() {{
  const slug = select.value;
  const model = bySlug[slug];
  document.getElementById('position-type').textContent = model.position_type || '—';
  document.getElementById('theta').textContent = model.compatible ? model.theta.toFixed(0) : '—';
  document.getElementById('rotary-dim').textContent = model.compatible ? model.rotary_dim : '—';
  document.getElementById('score-end').textContent = model.compatible ? model.score_spearman_end.toFixed(3) : '—';

  if (model.charts) {{
    missing.style.display = 'none';
    chartsFrame.style.display = 'block';
    chartsFrame.src = `rope-results/${{slug}}/phase3b_rope_charts.html`;
  }} else {{
    chartsFrame.style.display = 'none';
    chartsFrame.removeAttribute('src');
    missing.style.display = 'block';
    missing.textContent = model.skip_reason || `No charts saved yet. Run: poetry run python phase3-positional-embeddings/phase3b_rope_charts.py --model ${{slug}}`;
  }}

  document.getElementById('open-results').onclick = () => window.open(`rope-results/${{slug}}/results.json`, '_blank');
  document.getElementById('open-charts').onclick = () => {{
    if (model.charts) {{
      window.open(`rope-results/${{slug}}/phase3b_rope_charts.html`, '_blank');
    }}
  }};
}}

select.value = "{first}";
select.addEventListener('change', update);
update();
</script>
</body>
</html>
"""

with open(OUT, 'w') as f:
    f.write(html)

print(f"Saved to {OUT}")
