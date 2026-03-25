# Phase 2: Ghost Cluster, Analogies, and Nearest Neighbors

**Model:** GPT-2 (50,257 tokens x 768 dimensions)

## Summary

Phase 1 measured global geometry. This phase zooms in on concrete local phenomena: a tight low-norm ghost cluster, newline as a meaningful exception, nearest-neighbor structure, and raw embedding analogies. The saved research outputs now use strict exact probes; the browser explorers remain heuristic and exploratory.

## 1. The Ghost Cluster: A Tight Low-Norm Collapse

GPT-2 has a **59-token ghost cluster** with mean pairwise cosine **0.995**, minimum pairwise cosine **0.953**, and diameter **0.047** under a complete-linkage criterion. This is a strong claim: every pair inside the reported cluster is highly similar, not just connected through a chain.

Representative examples include byte/control-like tokens such as `À`, `Á`, `õ`, `ö`, and `÷`.

These tokens sit at low norm relative to the full vocabulary:

- Ghost-cluster mean norm: **3.092**
- Global mean norm: **3.959**

The picture is consistent with undertrained or weakly differentiated tokens collapsing into a near-shared vector.

## 2. Newline's Identity

Newline is **not** part of the ghost cluster.

Its nearest neighbors are:

1. `'\n\n'`
2. `' The'`
3. `'<|endoftext|>'`

That is a strong sign that GPT-2 treats newline as a boundary or context-reset marker rather than generic whitespace.

## 3. Nearest Neighbors: The Model's Taxonomy

Using strict exact probe tokens:

| Token | Top Neighbors | Category |
|-------|---------------|----------|
| `' the'` | `' a'`, `' an'`, `' it'` | Determiners/pronouns |
| `' at'` | `' in'`, `' on'`, `' with'` | Prepositions |
| `' king'` | `' kings'`, `' King'`, `' queen'` | Royalty |
| `' queen'` | `' queens'`, `' Queen'`, `' king'` | Royalty (gendered) |
| `' dog'` | `' dogs'`, `' Dog'`, `'Dog'` | Animals |
| `' Python'` | `' python'`, `'Python'`, `'python'` | Programming languages |

The static embeddings already carry clean local structure before any attention is applied.

## 4. Embedding Analogies Work

GPT-2 still passes the core raw-embedding analogies:

| Analogy | Top Result | Cosine |
|---------|------------|--------|
| king:queen::man:? | **` woman`** | 0.662 |
| dog:dogs::cat:? | **` cats`** | 0.817 |
| france:paris::japan:? | **` Tokyo`** | 0.688 |
| big:bigger::small:? | **` smaller`** | 0.797 |

These are static vocabulary vectors, not in-context activations. The result is real for GPT-2, but later phases and the curated cross-model comparison show that this behavior is not universal across all models.

## 5. Weird Tokens: Training Data Archaeology

High-norm outliers still look like dataset artifacts:

- `SPONSORED`
- `soDeliveryDate`
- `BuyableInstoreAndOnline`

But the stronger old claim that weird tokens form a sharply coherent region does **not** hold up well under the refreshed numbers:

- Outlier pairwise cosine: **0.279**
- Global mean cosine: **0.270**

So the safer interpretation is that GPT-2 contains obvious rare or artifact-heavy outliers, not that all weird tokens cluster together tightly.

## Key Takeaways

1. **GPT-2 has a real ghost cluster.** Under a complete-linkage threshold, 59 low-norm tokens remain tightly collapsed.
2. **Newline is semantically distinct.** It behaves like a boundary token, not a dead control token.
3. **Static analogies are genuinely present in GPT-2.** The classic examples still work in the raw embedding matrix.
4. **Nearest-neighbor structure is clean.** Grammar and lexical categories appear locally in the embedding space before context.
5. **Outlier tokens are real, but their group structure is weaker than it first looked.** The artifacts are obvious individually; the collective cluster claim needed to be softened.

## Files

| File | Description |
|------|-------------|
| `deep_dive.py` | Analysis: ghost cluster, analogies, nearest neighbors, weird tokens |
| `charts.py` | Ghost cluster heatmap with reference tokens |
| `analogy_explorer.py` | Interactive analogy explorer (heuristic lookup, localhost:8765) |
| `neighbor_explorer.py` | Interactive nearest neighbor explorer (heuristic lookup, localhost:8766) |
| `results.json` | Detailed data: exact-probe neighbor lists, analogy results, ghost cluster stats |
