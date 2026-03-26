# Phase 3: Positional Embeddings

## Question

Phase 1 covered global token-embedding geometry. Phase 2 covered exact local token phenomena. Phase 3 now asks a narrower question:

1. What structure lives in GPT-2's learned positional embedding matrix?
2. How separate are token and position subspaces?
3. How much does adding position perturb the exact Phase 2 token probes?

## Refactor Note

This phase was rewritten to match the Phase 1-2 architecture:

- `explore.py` produces structured `results/<model>/results.json`
- `charts.py` renders HTML from precomputed outputs
- `browser.py` switches between saved model outputs
- models without learned absolute position matrices are skipped explicitly

The older Phase 3 scripts mixed in Phase 1 and Phase 2 follow-ups such as PC1 removal and ghost-cluster interchangeability. Those topics are now treated as belonging to their original phases rather than to Phase 3.

## Representation Scope

- **Representation type:** learned absolute positional embedding matrix
- **Primary compatible model in the current registry:** GPT-2
- **Skipped models:** Pythia-70m, SmolLM2-135M, and Qwen2.5-0.5B use RoPE rather than a learned absolute position matrix

## Direct Observations

### 1. The norm profile is sharply asymmetric

- Position `0` has norm `9.88`, the largest in the matrix
- Position `1023` has norm `0.12`, effectively collapsing at the context edge
- Most interior positions sit on a much flatter plateau near `3.35`

This is the main structural fact of GPT-2's learned position table: the beginning is loud, the end is faint.

### 2. Nearby positions are extremely similar

- Gap `1` cosine: `0.997`
- Gap `32` cosine: `0.913`
- Gap `128` cosine: `0.259`
- Gap `256` cosine: `-0.446`

The path through position space is smooth locally, then rotates far enough that medium-range positions become weakly aligned or opposed.

### 3. Token and position occupy unusually different subspaces

- Top-5 principal subspace alignment: `0.055` vs random baseline `0.081`
- Top-10 alignment: `0.079` vs random baseline `0.114`
- Top-50 alignment: `0.143` vs random baseline `0.255`
- Per-dimension variance Spearman correlation is strongly negative in the saved results

The position matrix does not merely avoid the most important token directions. It is less aligned with them than a random subspace would be.

### 4. Exact Phase 2 probes stay fairly stable, but not perfectly

Using the same exact concept probes introduced in Phase 2:

- Mean nearest-neighbor Jaccard overlap with position `0` is `0.885` at position `1`
- It falls to about `0.77` through much of the middle context
- It drops to `0.606` at position `1023`

So position perturbs local token neighborhoods, but usually does not destroy them until the context edge.

### 5. The curated analogy battery stays intact under a shared position shift

For GPT-2, the Phase 2 four-item analogy battery remains `4/4` top-1 across all sampled positions in the current implementation.

That is a direct observation about the static embedding arithmetic after adding the same position vector to all candidates. It does **not** imply that contextual analogy behavior is unchanged deeper in the network.

## Interpretations

- The very large position-0 norm is consistent with GPT-2 learning a strong "sequence start" bias without an explicit BOS token.
- The near-zero final position is consistent with weak training pressure at the context limit.
- The low subspace alignment suggests GPT-2 partitioned the residual stream into partly separate "what token is this?" and "where am I?" directions before any attention layer runs.
- The fact that exact probe neighborhoods mostly survive positional addition helps explain why the semantic structure seen in Phases 1 and 2 remains recognizable after the first embedding sum.

## Limitations

- These claims apply to **learned absolute** position matrices, not to RoPE or ALiBi.
- The current active registry only has one learned-absolute compatible model, so Phase 3 is descriptive rather than comparative.
- The analogy result here is about static vector arithmetic after adding position, not about contextual hidden states or logits.

## Files

- `explore.py`: structured positional analysis and exact-probe interaction metrics
- `charts.py`: precomputed charts HTML
- `browser.py`: multi-model results browser
- `results/<model>/results.json`: saved metrics and compatibility status

## Next Useful Extensions

1. Add a learned-absolute comparison model back into the active registry so Phase 3 can become genuinely cross-model.
2. Compare learned absolute position matrices against RoPE-derived effective position directions rather than skipping them entirely.
3. Carry the same exact probe set into Phase 5-style layer tracking to measure where position-token separation gets merged downstream.
