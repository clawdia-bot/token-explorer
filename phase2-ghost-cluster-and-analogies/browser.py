"""
Phase 2 browser for switching between precomputed model summaries and heatmaps.

Usage: poetry run python phase2-ghost-cluster-and-analogies/browser.py
"""

import json
from pathlib import Path


BASE = Path(__file__).parent
RESULTS = BASE / 'results'
OUT = BASE / 'phase2_browser.html'


models = []
for model_dir in sorted(RESULTS.iterdir()):
    if not model_dir.is_dir():
        continue
    results_path = model_dir / 'results.json'
    if not results_path.exists():
        continue

    with open(results_path) as f:
        results = json.load(f)

    ghost = results['ghost_cluster']
    newline_neighbors = [item['token'] for item in results['newline_neighbors'][:3]]
    analogies = {
        key: value[0]['token'] if value else None
        for key, value in results['analogies'].items()
    }
    models.append({
        'slug': model_dir.name,
        'name': results.get('model', {}).get('name', model_dir.name),
        'ghost_count': ghost['count'],
        'ghost_mean': ghost['pairwise_cosine']['mean'],
        'ghost_min': ghost['pairwise_cosine']['min'],
        'newline': newline_neighbors,
        'analogies': analogies,
        'heatmap': (model_dir / 'ghost_heatmap.html').exists(),
    })

if not models:
    raise FileNotFoundError("No Phase 2 model results found. Run deep_dive.py first.")

payload = json.dumps(models)
first = models[0]['slug']

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Phase 2 Browser</title>
<style>
  body {{ background:#151515; color:#eee; font-family:'SF Mono','Consolas',monospace; margin:0; padding:24px; }}
  h1 {{ margin:0 0 8px 0; font-size:28px; }}
  .sub {{ color:#888; margin-bottom:20px; }}
  .toolbar {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:18px; }}
  select, button {{ background:#262626; color:#eee; border:1px solid #444; border-radius:6px; padding:8px 10px; font:inherit; }}
  .summary {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px; }}
  .card {{ background:#202020; border:1px solid #333; border-radius:8px; padding:12px 14px; min-width:180px; }}
  .label {{ color:#8a8a8a; font-size:12px; margin-bottom:4px; }}
  .value {{ font-size:20px; color:#fff; }}
  .layout {{ display:flex; gap:18px; flex-wrap:wrap; }}
  .left, .right {{ flex:1 1 420px; }}
  .panel {{ background:#202020; border:1px solid #333; border-radius:8px; margin-bottom:18px; overflow:hidden; }}
  .panel-header {{ padding:10px 12px; border-bottom:1px solid #333; color:#aaa; }}
  .panel-body {{ padding:14px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:10px; }}
  .mini {{ background:#1a1a1a; border:1px solid #2d2d2d; border-radius:6px; padding:10px; }}
  iframe {{ width:100%; height:720px; border:0; background:#111; }}
  .empty {{ padding:24px; color:#888; }}
  ul {{ margin:0; padding-left:18px; }}
  li {{ margin-bottom:6px; }}
</style>
</head>
<body>
  <h1>Phase 2 Browser</h1>
  <div class="sub">Switch between precomputed deep-dive summaries and ghost-cluster heatmaps without rerunning scripts.</div>
  <div class="toolbar">
    <label for="model-select">Model:</label>
    <select id="model-select"></select>
    <button id="open-results">Open results.json</button>
    <button id="open-heatmap">Open heatmap in new tab</button>
  </div>
  <div class="summary">
    <div class="card"><div class="label">Ghost cluster size</div><div class="value" id="ghost-count">-</div></div>
    <div class="card"><div class="label">Mean cosine</div><div class="value" id="ghost-mean">-</div></div>
    <div class="card"><div class="label">Min cosine</div><div class="value" id="ghost-min">-</div></div>
  </div>
  <div class="layout">
    <div class="left">
      <div class="panel">
        <div class="panel-header">Newline neighbors</div>
        <div class="panel-body"><ul id="newline-list"></ul></div>
      </div>
      <div class="panel">
        <div class="panel-header">Analogy winners</div>
        <div class="panel-body"><div id="analogy-grid" class="grid"></div></div>
      </div>
    </div>
    <div class="right">
      <div class="panel">
        <div class="panel-header">Ghost heatmap</div>
        <div id="heatmap-missing" class="empty" style="display:none;">This model has no saved ghost-cluster heatmap because the strict detector found no cluster.</div>
        <iframe id="heatmap-frame"></iframe>
      </div>
    </div>
  </div>
<script>
const models = {payload};
const bySlug = Object.fromEntries(models.map(m => [m.slug, m]));
const select = document.getElementById('model-select');
const heatmapFrame = document.getElementById('heatmap-frame');
const heatmapMissing = document.getElementById('heatmap-missing');

models.forEach(model => {{
  const opt = document.createElement('option');
  opt.value = model.slug;
  opt.textContent = model.name;
  select.appendChild(opt);
}});

function update() {{
  const slug = select.value;
  const model = bySlug[slug];
  document.getElementById('ghost-count').textContent = model.ghost_count;
  document.getElementById('ghost-mean').textContent = model.ghost_mean === null ? '—' : model.ghost_mean.toFixed(3);
  document.getElementById('ghost-min').textContent = model.ghost_min === null ? '—' : model.ghost_min.toFixed(3);

  const newlineList = document.getElementById('newline-list');
  newlineList.innerHTML = '';
  model.newline.forEach(token => {{
    const li = document.createElement('li');
    li.textContent = token;
    newlineList.appendChild(li);
  }});

  const grid = document.getElementById('analogy-grid');
  grid.innerHTML = '';
  Object.entries(model.analogies).forEach(([label, winner]) => {{
    const card = document.createElement('div');
    card.className = 'mini';
    card.innerHTML = `<div class="label">${{label}}</div><div>${{winner || '—'}}</div>`;
    grid.appendChild(card);
  }});

  if (model.heatmap) {{
    heatmapMissing.style.display = 'none';
    heatmapFrame.style.display = 'block';
    heatmapFrame.src = `results/${{slug}}/ghost_heatmap.html`;
  }} else {{
    heatmapFrame.style.display = 'none';
    heatmapFrame.removeAttribute('src');
    heatmapMissing.style.display = 'block';
  }}

  document.getElementById('open-results').onclick = () => window.open(`results/${{slug}}/results.json`, '_blank');
  document.getElementById('open-heatmap').onclick = () => {{
    if (model.heatmap) {{
      window.open(`results/${{slug}}/ghost_heatmap.html`, '_blank');
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
