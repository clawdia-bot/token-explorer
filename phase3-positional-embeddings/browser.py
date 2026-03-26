"""
Phase 3 browser for switching between precomputed positional embedding outputs.

Usage: poetry run python phase3-positional-embeddings/browser.py
"""

import json
from pathlib import Path


BASE = Path(__file__).parent
RESULTS = BASE / 'results'
OUT = BASE / 'phase3_browser.html'


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
    norm_summary = results.get('position_norms', {}) if compatible else {}
    similarity = results.get('position_similarity', {}) if compatible else {}
    interaction = results.get('token_position_interaction', {}) if compatible else {}

    models.append({
        'slug': model_dir.name,
        'name': model.get('name', model_dir.name),
        'compatible': compatible,
        'position_type': model.get('position_type'),
        'max_positions': model.get('max_positions'),
        'skip_reason': results.get('skip_reason'),
        'charts': (model_dir / 'phase3_charts.html').exists(),
        'max_position': norm_summary.get('max_position'),
        'max_norm': norm_summary.get('max'),
        'adjacent_mean': similarity.get('adjacent_mean'),
        'probe_count': interaction.get('probe_count'),
        'mean_jaccard_end': None if not compatible else interaction['mean_jaccard_by_position'][str(interaction['positions'][-1])],
    })

if not models:
    raise FileNotFoundError("No Phase 3 model results found. Run explore.py first.")

payload = json.dumps(models)
first = models[0]['slug']

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="dark">
<meta name="theme-color" content="#151515">
<title>Phase 3 Browser</title>
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
  iframe {{ width:100%; height:680px; border:0; background:#111; }}
  .empty {{ padding:24px; color:#888; }}
</style>
</head>
<body>
  <h1>Phase 3 Browser</h1>
  <div class="sub">Switch between precomputed positional embedding results without rerunning analysis.</div>
  <div class="toolbar">
    <label for="model-select">Model:</label>
    <select id="model-select"></select>
    <button id="open-results">Open results.json</button>
    <button id="open-charts">Open charts in new tab</button>
  </div>
  <div class="summary">
    <div class="card"><div class="label">Position type</div><div class="value" id="position-type">-</div></div>
    <div class="card"><div class="label">Peak norm pos</div><div class="value" id="peak-position">-</div></div>
    <div class="card"><div class="label">Adjacent cosine</div><div class="value" id="adjacent-cosine">-</div></div>
    <div class="card"><div class="label">End stability</div><div class="value" id="end-jaccard">-</div></div>
  </div>
  <div class="panel">
    <div class="panel-header">Phase 3 charts</div>
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
  document.getElementById('peak-position').textContent =
    model.compatible ? `${{model.max_position}} (${{model.max_norm.toFixed(2)}})` : '—';
  document.getElementById('adjacent-cosine').textContent =
    model.compatible ? model.adjacent_mean.toFixed(3) : '—';
  document.getElementById('end-jaccard').textContent =
    model.compatible ? model.mean_jaccard_end.toFixed(3) : '—';

  if (model.charts) {{
    missing.style.display = 'none';
    chartsFrame.style.display = 'block';
    chartsFrame.src = `results/${{slug}}/phase3_charts.html`;
  }} else {{
    chartsFrame.style.display = 'none';
    chartsFrame.removeAttribute('src');
    missing.style.display = 'block';
    missing.textContent = model.skip_reason || `No charts saved yet. Run: poetry run python phase3-positional-embeddings/charts.py --model ${{slug}}`;
  }}

  document.getElementById('open-results').onclick = () => window.open(`results/${{slug}}/results.json`, '_blank');
  document.getElementById('open-charts').onclick = () => {{
    if (model.charts) {{
      window.open(`results/${{slug}}/phase3_charts.html`, '_blank');
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
