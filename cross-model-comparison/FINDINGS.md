# Cross-Model Embedding Comparison — Findings

4 active models compared: GPT-2, Pythia-70m, SmolLM2-135M, Qwen2.5-0.5B

This comparison uses a curated **60-concept exact probe inventory**. Analogy scoring and neighborhood overlap are computed from exact per-model token targets rather than automatic string stripping or shared-vocabulary heuristics.

## 1. Isotropy Spectrum

| Model | Anisotropy | PR / dim | Eff. dim / dim | Mean norm |
|-------|-----------|----------|----------------|-----------|
| SmolLM2-135M | **0.448** | 67/576 (12%) | 342/576 (59%) | 3.179 |
| GPT-2 | 0.269 | 428/768 (56%) | 587/768 (76%) | 3.959 |
| Qwen2.5-0.5B | 0.153 | 447/896 (50%) | 778/896 (87%) | 0.463 |
| Pythia-70m | **0.004** | 352/512 (69%) | 440/512 (86%) | 0.704 |

**Takeaway:** static embedding geometry differs sharply even among small-to-mid-sized models. SmolLM2 is highly anisotropic and low-rank by participation ratio. Pythia is almost isotropic. Qwen sits between GPT-2 and Pythia on anisotropy while retaining very high effective dimensionality.

## 2. Ghost Cluster Summary

| Model | Ghost cluster size | Mean pairwise cosine | Minimum pairwise cosine |
|-------|-------------------|----------------------|-------------------------|
| Qwen2.5-0.5B | **3,278** | 0.997 | 0.952 |
| GPT-2 | 59 | 0.995 | 0.953 |
| Pythia-70m | none | — | — |
| SmolLM2-135M | none | — | — |

**Takeaway:** ghost clusters are not universal, but they are also not unique to GPT-2. Under the stricter complete-linkage criterion, GPT-2 keeps a compact byte/control-like cluster and Qwen shows a much larger low-norm collapse, while Pythia and SmolLM2 still show none.

## 3. Analogy Scorecard (12 Curated Exact Analogies)

| Model | Score | Notes |
|-------|-------|-------|
| GPT-2 | **11/12** | Only misses the antonym analogy |
| Pythia-70m | 10/12 | Misses the Japan capital analogy and antonym |
| SmolLM2-135M | 7/12 | Misses gender, superlative, one tense analogy, antonym, and one capital analogy |
| Qwen2.5-0.5B | 7/12 | Similar error pattern to SmolLM2, plus a capital2 miss |

**Takeaway:** GPT-2 remains strongest on static analogies, Pythia is close behind, and both SmolLM2 and Qwen are weaker despite retaining clean local neighborhoods.

## 4. Neighborhood Agreement

Measured as mean Jaccard overlap of top-5 neighbors across **20 probes** inside the **60-concept exact inventory**.

| Pair | Mean Jaccard |
|------|--------------|
| GPT-2 ↔ SmolLM2 | **0.509** |
| GPT-2 ↔ Qwen | 0.508 |
| GPT-2 ↔ Pythia | 0.485 |
| SmolLM2 ↔ Qwen | 0.468 |
| Pythia ↔ Qwen | 0.386 |
| Pythia ↔ SmolLM2 | 0.350 |

**Takeaway:** Qwen integrates more naturally into the curated exact-probe neighborhood picture than it did in the old auto-aligned comparison. Under the stricter methodology, GPT-2 and Qwen are almost as aligned on these probes as GPT-2 and SmolLM2.

## Cross-Cutting Themes

1. **Token-rank effects are model-specific.** GPT-2 and SmolLM2 show strong origin-based token-rank correlations; Pythia and Qwen do not.
2. **Ghost-cluster claims needed tighter criteria, not removal.** Once the comparison stops using permissive connected components, GPT-2 and Qwen still retain real clusters, while Pythia and SmolLM2 drop out.
3. **Static analogies survive stricter methodology.** The headline result did not disappear when probe selection became exact; GPT-2 and Pythia remain strong.
4. **Exact probes are worth the smaller scope.** The comparison is narrower than the old auto-aligned shared-vocabulary version, but the resulting claims are much cleaner and Qwen is still meaningfully comparable within that frame.
