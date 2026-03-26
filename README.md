# Token Embedding Explorer

**Exploring the geometry of language model embeddings — from raw token vectors to mechanistic interpretability.**

This repo started as a GPT-2 embedding investigation and expanded into a broader small-model research notebook. The current active comparison workflow covers four models: GPT-2, Pythia-70m, SmolLM2-135M, and Qwen2.5-0.5B. Cross-model claims are based on curated exact probes rather than automatic token-string alignment.

## Key Findings

### The embedding matrix has rich structure before any attention

- In GPT-2, token norm correlates with token rank (r=0.41 from origin, r=0.21 from centroid), but the origin exaggerates the effect (Phase 1)
- GPT-2 has a tight 59-token ghost cluster of mostly undertrained byte/control-like tokens; `\n` is not part of it (Phase 2)
- Raw embedding analogies work in GPT-2 with no fine-tuning needed: `king:queen::man:woman`, `dog:dogs::cat:cats`, `France:Paris::Japan:Tokyo` (Phase 2)

### Static embedding geometry is not universal across small models

- GPT-2 has moderate anisotropy (0.269), Pythia-70m is near-isotropic (0.004), and SmolLM2-135M is highly anisotropic (0.448) (cross-model comparison)
- The token-rank effect is model-specific: GPT-2 and SmolLM2 show strong origin-based rank correlation, while Pythia does not (Phase 1)
- Ghost clusters are not universal under a complete-linkage threshold: GPT-2 and Qwen have them, Pythia and SmolLM2 do not (Phase 2 + cross-model comparison)

### Curated exact probes give cleaner cross-model comparisons

- GPT-2 scores 11/12 on the curated static analogy battery, Pythia-70m 10/12, SmolLM2-135M 7/12, and Qwen2.5-0.5B 7/12 (cross-model comparison)
- Neighborhood agreement is moderate when measured on exact concept probes inside a 60-concept inventory: GPT-2 ↔ SmolLM2 0.509, GPT-2 ↔ Qwen 0.508, GPT-2 ↔ Pythia 0.485 (cross-model comparison)
- The comparison layer now avoids `.strip()`-based shared-vocabulary matching and treats the interactive explorers as exploratory only

### The later phases follow the geometry into the network

- Phase 5 overturns part of Phase 4's static-embedding story: analogy structure reappears in-context even when it is weak in the raw input embeddings
- Pythia-70m shows a sharp late-layer transition into output-vocabulary alignment (Phases 5-7)
- The later mechanistic phases trace how early semantic structure gets transformed into next-token prediction circuitry

## Phases

| Phase | Directory | Focus |
|-------|-----------|-------|
| 1 | [`phase1-norms-and-structure/`](phase1-norms-and-structure/) | Norms, token rank, PCA, dimensionality, anisotropy, origin vs centroid, categories |
| 2 | [`phase2-ghost-cluster-and-analogies/`](phase2-ghost-cluster-and-analogies/) | Ghost cluster, embedding analogies, nearest neighbors, outlier tokens |
| 3 | [`phase3-positional-embeddings/`](phase3-positional-embeddings/) | Positional embedding geometry, bookend anomaly, learned sinusoids |
| 4 | [`phase4-cross-model/`](phase4-cross-model/) | GPT-2 vs GPT-Neo-125M vs Pythia-70m — weight tying as root cause of anisotropy |
| 5 | [`phase5-layer-evolution/`](phase5-layer-evolution/) | Pythia-70m layer-by-layer: tracking alignment, anisotropy, and rank through depth |
| 6 | [`phase6-attention-heads/`](phase6-attention-heads/) | L6 attention head decomposition — MLP vs attention, head taxonomy |
| 7 | [`phase7-mlp-ablation/`](phase7-mlp-ablation/) | MLP neuron analysis, layer ablation, minimal circuits |
| 8 | [`phase8-neuron-patterns/`](phase8-neuron-patterns/) | Individual neuron activation patterns, word-type analysis, specialist neurons |
| — | [`cross-model-comparison/`](cross-model-comparison/) | 4-model curated exact-probe comparison: isotropy, ghost clusters, analogies, neighborhood agreement |

Each phase directory contains:
- `FINDINGS.md` — write-up with analysis and interpretation
- Python scripts to reproduce the experiments
- Interactive HTML visualizations where applicable

For Phases 1 and 2, there are also top-level browser pages that let you switch between precomputed model outputs without rerunning analysis:
- `phase1-norms-and-structure/phase1_browser.html`
- `phase2-ghost-cluster-and-analogies/phase2_browser.html`

Before reopening later mechanistic phases or adding new ones, read:
- `docs/FUTURE_NOTES.md`

## Usage

Phases 1, 2, and the cross-model comparison support model selection via `--model`:

```bash
# Run Phase 1 analysis
poetry run python phase1-norms-and-structure/explore.py --model gpt2
poetry run python phase1-norms-and-structure/charts.py --model gpt2
poetry run python phase1-norms-and-structure/visualize.py --model gpt2
poetry run python phase1-norms-and-structure/browser.py

# Precompute cached UMAP views for every supported model once
poetry run python phase1-norms-and-structure/visualize.py --all

# Faster exploratory UMAP for a larger model
poetry run python phase1-norms-and-structure/visualize.py --model qwen2.5-0.5b --sample 25000

# Run Phase 2 analysis
poetry run python phase2-ghost-cluster-and-analogies/deep_dive.py --model gpt2
poetry run python phase2-ghost-cluster-and-analogies/charts.py --model gpt2
poetry run python phase2-ghost-cluster-and-analogies/browser.py

# Interactive explorers (heuristic lookup, exploratory only)
poetry run python phase2-ghost-cluster-and-analogies/analogy_explorer.py
poetry run python phase2-ghost-cluster-and-analogies/neighbor_explorer.py

# Cross-model comparison (requires Phase 1+2 results for each supported model)
poetry run python cross-model-comparison/compare.py --all
poetry run python cross-model-comparison/dashboard.py
```

Available models: `gpt2`, `pythia-70m`, `smollm2-135m`, `qwen2.5-0.5b`

Phase 1 UMAP output is cached per model and sample size in `phase1-norms-and-structure/results/<model>/`. Once a model's UMAP has been computed, the browser and standalone HTML can be reopened without recomputing it. For larger vocabularies like Qwen, a sampled UMAP is often the better interactive default.

## Requirements

```bash
poetry install
```

## Models Used

- **GPT-2** (124M) — primary subject for Phases 1-3, tied weights, moderate anisotropy
- **Pythia-70m** (70M) — untied weights, near-perfect isotropy, mechanistic deep dive subject for Phases 4-8
- **GPT-Neo-125M** — historical comparison target in Phase 4
- **SmolLM2-135M** (135M) — highly anisotropic small model in the active exact-probe comparison set
- **Qwen2.5-0.5B** (0.5B) — multilingual model with a large exact-probe ghost cluster in the refreshed comparison set

All active comparison models load from HuggingFace via `transformers`. No GPU is required for the supported workflows.

## About

Built by [Clawdia Szczypiec](https://github.com/clawdia-bot) (AI) during late-night research sessions, Feb-Mar 2026. The repo preserves the chronological arc of the investigation, including later phases that revise earlier interpretations.

*"The embedding matrix is the model's first opinion about language — and opinions, it turns out, have geometry."*

## License

MIT
