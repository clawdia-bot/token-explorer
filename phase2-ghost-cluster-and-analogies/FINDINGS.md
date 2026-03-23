# Phase 2: Ghost Cluster, Analogies, and Nearest Neighbors

**Model:** GPT-2 (50,257 tokens x 768 dimensions)

## Summary

Phase 1 measured global structure. This phase zooms in on specific phenomena: dead tokens that collapsed to identical vectors, raw analogies that work without fine-tuning, and a nearest-neighbor taxonomy that mirrors human intuition.

## 1. The Ghost Cluster: Control Characters Are Identical

Tokens 188-221 (ASCII control characters, byte-level tokens) form a tight cluster with pairwise cosine **0.949** (excluding newline). They're essentially the same vector with minor perturbations.

**The Newline Exception:** Token 198 (`\n`) broke free with cosine **0.033** to its siblings and norm 2.688. Newline is semantically meaningful — it separates sentences, paragraphs, code blocks. The model learned this.

Control characters almost never appear in training data. No gradient signal to differentiate them — they collapsed to near-identical vectors. The model's "I don't know what this is" embedding. (See also Phase 1's finding that `' externalToEVA'` is closest to the centroid — same phenomenon.)

## 2. Newline's Identity

Newline's nearest neighbors reveal what the model thinks it means:

1. `'\n\n'` (double newline)
2. `' The'` (sentence starter)
3. `'<|endoftext|>'` (document boundary)

Newline is grouped with **things that start new context**, not with whitespace. Compare: newline vs space cosine is only 0.328 — they serve different roles despite both being "whitespace."

## 3. Nearest Neighbors: The Model's Taxonomy

| Token | Top Neighbors | Category |
|-------|---------------|----------|
| `' the'` | `' a'`, `' an'`, `' it'` | Determiners/pronouns |
| `' at'` | `' in'`, `' on'`, `' with'` | Prepositions |
| `' king'` | `' kings'`, `' King'`, `' queen'` | Royalty |
| `' dog'` | `' dogs'`, `' Dog'`, `'Dog'` | Animals |
| `' Python'` | `' python'`, `'Python'`, `'python'` | Programming languages |

Clean semantic clustering exists in the static embeddings — before any attention or context. The embedding layer's priors already organize tokens by grammatical and semantic role.

## 4. Embedding Analogies Work

Raw embedding analogies — no fine-tuning, no task-specific training:

| Analogy | Top Result | Cosine |
|---------|------------|--------|
| king:queen::man:? | **` woman`** | 0.662 |
| dog:dogs::cat:? | **` cats`** | 0.817 |
| France:Paris::Japan:? | **` Tokyo`** | 0.688 |
| big:bigger::small:? | **` smaller`** | 0.797 |

The embedding layer has learned linear relationships between concepts — gender, pluralization, geography, comparison — before any attention. These relationships are imposed by backpropagation from the transformer layers above.

**Note:** Phase 4 will test whether this holds across models. Spoiler: it's complicated.

## 5. Weird Tokens: Training Data Archaeology

High-norm tokens include clear training data artifacts: `soDeliveryDate` (norm 6.22), `BuyableInstoreAndOnline` (norm 5.74), `SPONSORED` (norm 6.32). These exist because BPE merged frequently-occurring strings from e-commerce sites and Reddit.

Mean pairwise cosine among weird tokens: **0.459** (vs global 0.269). The model groups "things I barely saw in training" into a distinct region of embedding space.

Some weird tokens like `' externalToEVA'` collapsed to ghost-cluster norms (~3.09) while their extensions (`' externalToEVAOnly'`, norm 5.18) have normal norms. The base fragments are dead; the extensions got enough training signal to differentiate.

## Key Takeaways

1. **Untrained tokens collapse.** Ghost cluster and glitch tokens converge to near-identical vectors near the centroid.
2. **Meaningful tokens differentiate.** Even `\n`, a control character, breaks free when it carries semantic weight.
3. **Analogies work raw.** Linear relationships (gender, number, geography) exist in static embeddings without any task-specific training.
4. **Nearest neighbors are clean.** The model's taxonomy mirrors human intuition: prepositions cluster with prepositions, royalty with royalty.

## Files

| File | Description |
|------|-------------|
| `deep_dive.py` | Analysis: ghost cluster, analogies, nearest neighbors, weird tokens |
| `charts.py` | Ghost cluster heatmap with reference tokens |
| `analogy_explorer.py` | Interactive analogy explorer (localhost:8765) |
| `neighbor_explorer.py` | Interactive nearest neighbor explorer (localhost:8766) |
| `results.json` | Detailed data: full neighbor lists, analogy results, ghost cluster stats |
