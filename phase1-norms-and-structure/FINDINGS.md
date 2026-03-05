# Phase 1: Norms, Structure, and Dimensionality

**Date:** 2026-02-25  
**Model:** GPT-2 (50,257 tokens × 768 dimensions)

## Summary

The embedding matrix is the model's first opinion about language — assigned before any attention, before any context. These 768-dimensional vectors are where the model places its raw priors about what each token *is*.

## 1. Norms: Frequency Writes Itself Into Geometry

Token embedding norm correlates strongly with rarity. Common function words cluster near the origin; rare tokens orbit far from it.

| Token | Norm | Frequency |
|-------|------|-----------|
| ` at` | 2.454 (min) | Extremely common preposition |
| ` in` | 2.465 | Extremely common preposition |
| ` the` | 2.670 | Most common English word |
| `SPONSORED` | 6.316 (max) | Rare advertising label |
| `soDeliveryDate` | 6.215 | E-commerce artifact |

**Correlation:** r=0.41 (token index vs norm), Spearman ρ=0.42. BPE token indices roughly reflect merge frequency, so: **the model pushes common tokens toward the origin and rare tokens to the periphery.**

Common tokens need to be versatile — they appear in many contexts, so their embedding should be relatively "neutral," close to the center where they can be easily pushed in any direction by the attention layers. Rare tokens can afford to be opinionated.

## 2. The Origin and the Mean

The mean embedding has norm 2.052 — about half the average token norm (3.959). It's closest to the ghost cluster (control characters — see Phase 2).

- `' the'` has the **lowest** cosine to the mean (0.135) — the most directionally unique token in the entire vocabulary
- Mean pairwise cosine is 0.517 — embeddings generally point in the same hemisphere

"The" is so common and contextually flexible that the model gave it a direction shared by nothing else. It's not generic — it's *specifically* generic.

## 3. Effective Dimensionality: The Space Is Used

| Metric | Value |
|--------|-------|
| Participation ratio | 428 / 768 |
| Entropy-based effective dims | 587 / 768 |
| Dims for 50% variance | 182 |
| Dims for 90% variance | 558 |
| Dims for 99% variance | 712 |
| Top PC explains | 1.81% |

No single dimension dominates. The space genuinely uses most of its 768 dimensions. PC1 (1.81%) seems to encode a frequency/salience axis.

## 4. Anisotropy: A Moderate Cone

Mean pairwise cosine similarity is 0.269 (5K random tokens). Not isotropic (would be ≈0 for uniform sphere), not a narrow cone either (some papers report 0.5+). The "narrow cone" problem from representation degeneration papers is moderate in GPT-2's embedding layer.

## 5. Script Clustering

Different writing systems occupy distinct regions:

- **Japanese (katakana/hiragana):** Mean norm 4.3-4.5, tightly clustered
- **CJK characters:** Mean norm 4.5+
- **Arabic/Hebrew:** Mean norm 4.0-4.2
- **Latin (common words):** Mean norm 3.5-3.8

Non-Latin scripts have higher norms, reflecting their rarity in GPT-2's English-dominated training data.

## Files

| File | Description |
|------|-------------|
| `explore.py` | Norms, PCA, anisotropy, categories, dimensionality |
| `visualize.py` | UMAP 2D interactive Plotly visualization |
