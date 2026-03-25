# Phase 1: Norms, Structure, and Dimensionality

**Model:** GPT-2 (50,257 tokens x 768 dimensions)

## Summary

The embedding matrix is the model's first opinion about language — assigned before any attention or context. This phase measures the geometry of GPT-2's token space: how norms, directions, and principal components organize the vocabulary. The current framing is deliberately conservative: the generic metric is **token rank**, not frequency. For GPT-2, token rank is still frequency-like because of how the BPE vocabulary was built, but that interpretation does not automatically transfer to other models.

## 1. Norms and Token Rank

Token embedding norm (distance from the origin) correlates with GPT-2 token rank.

| Token | Norm | Note |
|-------|------|------|
| ` at` | 2.454 (min) | Extremely common preposition |
| ` in` | 2.465 | Extremely common preposition |
| ` the` | 2.670 | Very early GPT-2 token rank |
| `SPONSORED` | 6.316 (max) | Rare advertising label |
| `soDeliveryDate` | 6.215 | E-commerce artifact |

**Correlation (token rank vs norm):** r=0.409. In GPT-2, earlier-ranked tokens tend to have lower norms, later-ranked tokens higher norms. Token length vs norm is r=-0.166, which is consistent with short, common fragments appearing earlier in the GPT-2 vocabulary.

## 2. Anisotropy

Mean pairwise cosine similarity is **0.269** on a 5K random token sample. GPT-2 is moderately anisotropic: embeddings prefer the same hemisphere, but the space is far from collapsed.

## 3. Effective Dimensionality

| Metric | Value |
|--------|-------|
| Participation ratio | 428 / 768 |
| Entropy-based effective dims | 587 / 768 |
| Dims for 50% variance | 182 |
| Dims for 90% variance | 558 |
| Dims for 99% variance | 712 |
| Top PC explains | 1.81% |

No single direction dominates. Even with moderate anisotropy, GPT-2 uses most of its 768 dimensions.

## 4. Origin vs Centroid

The zero vector in R^768 has no linguistic meaning. To test whether the token-rank effect is real or just a consequence of where the cloud happens to sit relative to the origin, we re-ran the same correlations from the centroid.

| Correlation | From origin | From centroid |
|-------------|-------------|---------------|
| Token rank vs distance | r=0.409 | r=0.207 |
| Token length vs distance | r=-0.166 | r=-0.256 |

**The token-rank correlation roughly halves** when measured from the centroid. That means the origin-based number is real but inflated by centroid displacement. In GPT-2's case, the interpretation is still frequency-like because token rank reflects the tokenizer's merge order, but the safer claim is about token rank first and frequency only as GPT-2-specific context.

The token-length correlation strengthens from the centroid (-0.166 to -0.256), suggesting that short tokens are genuinely closer to the middle of the embedding cloud, not just closer to an arbitrary origin.

**Closest to centroid:** `' externalToEVA'` (dist=1.531)  
**Farthest from centroid:** `'SPONSORED'` (dist=5.569)  
**Most dissimilar from mean direction:** `' the'` (cosine to mean=0.135)

## 5. Token Categories

Sixteen categories were classified by script and token type. Non-Latin scripts have higher norms in GPT-2, which fits its English-dominant training mix.

| Category | Count | Mean Norm |
|----------|-------|-----------|
| word | 32,090 | 3.818 |
| alpha_fragment | 13,643 | 4.302 |
| number | 1,692 | 3.411 |
| allcaps | 1,270 | 4.310 |
| punctuation | 774 | 4.070 |
| byte_token | 209 | 4.186 |
| other | 169 | 3.999 |
| japanese | 126 | 4.398 |
| cjk | 42 | 4.634 |
| whitespace | 59 | 3.824 |
| control_char | 31 | 4.175 |
| cyrillic | 47 | 4.128 |
| arabic | 41 | 4.074 |
| greek | 24 | 3.957 |
| hebrew | 11 | 4.085 |
| korean | 9 | 4.364 |

Most distant category pair: `cyrillic x number` (cosine 0.628).

## Key Takeaways

1. **GPT-2 norm tracks token rank.** The effect is strong from the origin and weaker from the centroid, so the safe claim is about token rank rather than universal frequency.
2. **The space is genuinely high-dimensional.** GPT-2 uses hundreds of effective dimensions, so any 2D view is a projection tool, not a faithful map.
3. **GPT-2 is moderately anisotropic.** Mean cosine 0.269 is substantial but not degenerate.
4. **Near the centroid means undertrained or generic.** Tokens like `' externalToEVA'` sit there because they barely moved, not because they are semantically central.
5. **`' the'` is directionally unusual.** It is the most anti-average token by cosine to the mean direction, which is a different claim from being far from every other token.

## Files

| File | Description |
|------|-------------|
| `explore.py` | Analysis: norms, token rank, PCA, anisotropy, centroid comparison, categories |
| `visualize.py` | UMAP 2D interactive Plotly visualization with token search |
| `charts.py` | 4-panel chart dashboard (distance distributions, PCA scree, anisotropy, categories) |
| `tokenutils.py` | Shared token display and categorization (16 categories, handles byte tokens) |
