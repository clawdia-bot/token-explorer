"""
Phase 1 browser for switching between precomputed model outputs.

Reads per-model results from results/<model>/ and builds a single HTML page with
dropdown-based switching for charts and optional cached UMAP views.

Usage: poetry run python phase1-norms-and-structure/browser.py
"""

import json
from pathlib import Path


BASE = Path(__file__).parent
RESULTS = BASE / 'results'
OUT = BASE / 'phase1_browser.html'


models = []
for model_dir in sorted(RESULTS.iterdir()):
    if not model_dir.is_dir():
        continue
    results_path = model_dir / 'results.json'
    if not results_path.exists():
        continue

    with open(results_path) as f:
        results = json.load(f)

    slug = model_dir.name
    umap_meta_path = model_dir / 'umap_cache_meta.json'
    umap_meta = None
    if umap_meta_path.exists():
        with open(umap_meta_path) as f:
            umap_meta = json.load(f)

    models.append({
        'slug': slug,
        'name': results.get('model', {}).get('name', slug),
        'charts': (model_dir / 'phase1_charts.html').exists(),
        'umap': (model_dir / 'umap_embeddings.html').exists(),
        'umap_sample': None if umap_meta is None else umap_meta.get('sample'),
        'umap_points': None if umap_meta is None else umap_meta.get('point_count'),
        'anisotropy': results['anisotropy']['mean_cosine'],
        'token_rank_r': results['token_rank']['pearson_r'],
        'pr': results['pca']['participation_ratio'],
        'hidden_dim': results['model']['hidden_dim'],
    })

if not models:
    raise FileNotFoundError("No Phase 1 model results found. Run explore.py first.")

payload = json.dumps(models)
first = models[0]['slug']

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="dark">
<meta name="theme-color" content="#151515">
<title>Phase 1 Browser</title>
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
  .row {{ display:flex; gap:18px; flex-wrap:wrap; }}
  .pane {{ flex:1 1 640px; background:#202020; border:1px solid #333; border-radius:8px; overflow:hidden; min-height:560px; }}
  .pane-header {{ padding:10px 12px; border-bottom:1px solid #333; color:#aaa; }}
  iframe {{ width:100%; height:560px; border:0; background:#111; }}
  .empty {{ padding:24px; color:#888; }}
</style>
</head>
<body>
  <h1>Phase 1 Browser</h1>
  <div class="sub">Switch between precomputed charts and cached UMAP views without rerunning scripts.</div>
  <div class="toolbar">
    <label for="model-select">Model:</label>
    <select id="model-select"></select>
    <button id="open-charts">Open charts in new tab</button>
    <button id="open-umap">Open UMAP in new tab</button>
  </div>
  <div class="summary">
    <div class="card"><div class="label">Anisotropy</div><div class="value" id="anisotropy">-</div></div>
    <div class="card"><div class="label">Token rank r</div><div class="value" id="token-rank">-</div></div>
    <div class="card"><div class="label">Participation ratio</div><div class="value" id="pr">-</div></div>
    <div class="card"><div class="label">UMAP cache</div><div class="value" id="umap-cache">-</div></div>
  </div>
  <div class="row">
    <div class="pane">
      <div class="pane-header">Phase 1 charts</div>
      <iframe id="charts-frame"></iframe>
    </div>
    <div class="pane">
      <div class="pane-header">UMAP view</div>
      <div id="umap-missing" class="empty" style="display:none;">No cached UMAP for this model yet. Run:<br><br><code>poetry run python phase1-norms-and-structure/visualize.py --model MODEL</code></div>
      <iframe id="umap-frame"></iframe>
    </div>
  </div>
<script>
const models = {payload};
const bySlug = Object.fromEntries(models.map(m => [m.slug, m]));
const select = document.getElementById('model-select');
const chartsFrame = document.getElementById('charts-frame');
const umapFrame = document.getElementById('umap-frame');
const umapMissing = document.getElementById('umap-missing');

models.forEach(model => {{
  const opt = document.createElement('option');
  opt.value = model.slug;
  opt.textContent = model.name;
  select.appendChild(opt);
}});

function update() {{
  const slug = select.value;
  const model = bySlug[slug];
  document.getElementById('anisotropy').textContent = model.anisotropy.toFixed(3);
  document.getElementById('token-rank').textContent = model.token_rank_r.toFixed(3);
  document.getElementById('pr').textContent = `${{model.pr.toFixed(0)}} / ${{model.hidden_dim}}`;
  if (!model.umap) {{
    document.getElementById('umap-cache').textContent = 'missing';
  }} else if (model.umap_sample) {{
    document.getElementById('umap-cache').textContent = `${{model.umap_points}} pts`;
  }} else if (model.umap_points) {{
    document.getElementById('umap-cache').textContent = `full (${{model.umap_points}})`;
  }} else {{
    document.getElementById('umap-cache').textContent = 'legacy';
  }}
  chartsFrame.src = `results/${{slug}}/phase1_charts.html`;
  if (model.umap) {{
    umapMissing.style.display = 'none';
    umapFrame.style.display = 'block';
    umapFrame.src = `results/${{slug}}/umap_embeddings.html`;
  }} else {{
    umapFrame.style.display = 'none';
    umapFrame.removeAttribute('src');
    umapMissing.innerHTML = `No cached UMAP for this model yet. Run:<br><br><code>poetry run python phase1-norms-and-structure/visualize.py --model ${{slug}}</code><br><br>For a faster exploratory view on larger vocabularies:<br><br><code>poetry run python phase1-norms-and-structure/visualize.py --model ${{slug}} --sample 25000</code>`;
    umapMissing.style.display = 'block';
  }}
  document.getElementById('open-charts').onclick = () => window.open(`results/${{slug}}/phase1_charts.html`, '_blank');
  document.getElementById('open-umap').onclick = () => {{
    if (model.umap) {{
      window.open(`results/${{slug}}/umap_embeddings.html`, '_blank');
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
