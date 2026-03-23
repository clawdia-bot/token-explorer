# Phase 1: Norms, Structure, and Dimensionality

**Model:** GPT-2 (50,257 tokens x 768 dimensions)

## Summary

The embedding matrix is the model's first opinion about language — assigned before any attention, before any context. These 768-dimensional vectors are where the model places its raw priors about what each token *is*. This phase measures the geometry of that space: how tokens are distributed by distance, direction, and dimensionality.

## 1. Norms and Frequency

Token embedding norm (distance from origin) correlates with rarity.

| Token | Norm | Note |
|-------|------|------|
| ` at` | 2.454 (min) | Extremely common preposition |
| ` in` | 2.465 | Extremely common preposition |
| ` the` | 2.670 | Most common English word |
| `SPONSORED` | 6.316 (max) | Rare advertising label |
| `soDeliveryDate` | 6.215 | E-commerce artifact |

**Correlation (BPE index vs norm):** r=0.409. BPE token indices roughly reflect merge frequency, so rarer tokens have higher norms. Token length vs norm: r=-0.166 (shorter tokens, which tend to be more frequent, have lower norms).

**But the origin is arbitrary.** See section 4.

## 2. Anisotropy

Mean pairwise cosine similarity is **0.269** (5K random token sample). This is moderate anisotropy — embeddings prefer the same hemisphere but don't collapse into a narrow cone. For reference, an isotropic (uniform) sphere would give ~0, and severe representation degeneration pushes above 0.5.

## 3. Effective Dimensionality

| Metric | Value |
|--------|-------|
| Participation ratio | 428 / 768 |
| Entropy-based effective dims | 587 / 768 |
| Dims for 50% variance | 182 |
| Dims for 90% variance | 558 |
| Dims for 99% variance | 712 |
| Top PC explains | 1.81% |

No single dimension dominates. Despite the moderate anisotropy, the space genuinely uses most of its 768 dimensions.

## 4. Origin vs Centroid

The zero vector in R^768 has no special meaning — it's where the weights were initialized, not a linguistically meaningful landmark. To test whether the norm-frequency relationship is real or an artifact of centroid displacement, we re-ran the same correlations measuring distance from the centroid (mean embedding) instead of the origin.

| Correlation | From origin | From centroid |
|-------------|-------------|---------------|
| BPE index vs distance | r=0.409 | r=0.207 |
| Token length vs distance | r=-0.166 | r=-0.256 |

**The BPE-frequency correlation halves** when measured from the centroid. This means the origin-based measurement amplifies the effect — common tokens dominate the mean and pull it toward them, inflating the apparent correlation. The radial structure is real (r=0.207 is still significant) but weaker than the origin-based number suggests.

The token-length correlation actually **strengthens** from the centroid (-0.166 to -0.256), suggesting that short tokens (which tend to be frequent) are genuinely closer to the center of the embedding cloud, not just closer to an arbitrary origin.

**Closest to centroid:** `' externalToEVA'` (dist=1.531) — a "glitch token" that barely moved from initialization. Being near the centroid means "generic/undertrained," not "semantically central."

**Farthest from centroid:** `'SPONSORED'` (dist=5.569) — the same token that has the highest origin-based norm. This outlier is robust to reference point.

**Most directionally unique:** `' the'` (cosine to mean=0.135) — more directionally distinct from the average token than any other in the vocabulary.

## 5. Token Categories

16 categories classified by script and token type. Non-Latin scripts have higher norms, reflecting their rarity in GPT-2's English-dominated training data.

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

Most distant category pair: cyrillic x number (cosine=0.628).

## Key Takeaways

1. **Norm encodes rarity**, but the effect is amplified when measured from the arbitrary origin. From the centroid, the correlation halves. The structure is real but overstated by naive measurement.
2. **The space is genuinely high-dimensional** (428-587 effective dims out of 768). No 2D visualization can capture this — UMAP shows neighborhoods, not the space.
3. **Moderate anisotropy** (mean cosine 0.269). Embeddings cluster directionally but aren't degenerate.
4. **"Near the centroid" means undertrained**, not semantically central. Glitch tokens like `' externalToEVA'` live there.
5. **`' the'` is directionally unique** (lowest cosine to mean at 0.135). The most common English word occupies a direction shared by nothing else.

## Files

| File | Description |
|------|-------------|
| `explore.py` | Analysis: norms, PCA, anisotropy, centroid comparison, categories |
| `visualize.py` | UMAP 2D interactive Plotly visualization with token search |
| `charts.py` | 4-panel chart dashboard (norm distribution, PCA scree, anisotropy, categories) |
| `tokenutils.py` | Shared token display and categorization (16 categories, handles byte tokens) |
