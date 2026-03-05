# Phase 2: Ghost Cluster, Analogies, and Nearest Neighbors

**Date:** 2026-02-25  
**Model:** GPT-2 (50,257 tokens × 768 dimensions)

## Summary

Deeper probing reveals hidden structure: dead tokens that collapsed to identical vectors, raw analogies that work without any fine-tuning, and a nearest-neighbor taxonomy that mirrors human intuition.

## 1. The Ghost Cluster: Control Characters Are Identical

Tokens 188–221 (ASCII control characters: \x00 through \x1f, \x7f, plus byte-level tokens) form an eerily tight cluster.

- Norm range: 3.086–3.092 (excluding newline and space)
- Pairwise cosine **without newline**: mean 0.949, min 0.199
- They're essentially the same vector with minor perturbations

**The Newline Exception:** Token 198 (`\n`) has norm 2.688 and cosine 0.033 to its siblings. It completely broke free from the ghost cluster because **newline is semantically meaningful** — it separates sentences, paragraphs, code blocks. The model learned this.

Control characters almost never appear in training data. No gradient signal to differentiate them → collapsed to near-identical vectors near the centroid. The model's "I don't know what this is" embedding.

## 2. Analogies Work (!!!)

Raw embedding analogies — no fine-tuning, no task-specific training — work surprisingly well:

| Analogy | Top Result | Cos |
|---------|------------|-----|
| king:queen::man:? | **` woman`** | 0.662 |
| dog:dogs::cat:? | **` cats`** | 0.817 |
| France:Paris::Japan:? | **` Tokyo`** | 0.688 |
| big:bigger::small:? | **` smaller`** | 0.797 |

The embedding layer has already learned linear relationships between concepts — gender, pluralization, geography, comparison — before any attention. These relationships must be imposed by backpropagation from the transformer layers above.

## 3. Nearest Neighbors: The Model's Taxonomy

- **`' the'`** → `' a'`, `' an'`, `' it'`, `' this'` — determiners/pronouns cluster
- **`' at'`** → `' in'`, `' on'`, `' with'`, `' as'` — prepositions cluster
- **`' king'`** → `' kings'`, `' King'`, `' queen'`, `' prince'` — royalty cluster
- **`' Python'`** → `' python'`, `' Django'`, `' PHP'`, `' Java'` — programming languages
- **`'\n'`** → `'\n\n'`, `' The'`, `<|endoftext|>`, `' ('` — sentence starters!

That last one: **newline's nearest neighbors are things that start sentences/paragraphs.** The model knows newline means "fresh start."

## 4. The Weird Tokens: E-Commerce Ghosts

High-norm tokens include clear training data artifacts: `soDeliveryDate`, `inventoryQuantity`, `BuyableInstoreAndOnline`. These have mean pairwise cosine of 0.487 (vs global 0.269) — the model groups "things I barely saw in training" into a distinct region.

`externalToEVA` and `TheNitrome` have ghost cluster norms (~3.09) — they collapsed to the "I don't know" default. But their extended versions (`externalToEVAOnly`, `TheNitromeFan`) have normal norms. The base fragments are dead; the extensions somehow got enough signal.

## Files

| File | Description |
|------|-------------|
| `deep_dive.py` | Ghost cluster, analogies, nearest neighbors, weird tokens |
