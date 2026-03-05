# Phase 3: Positional Embeddings & Follow-ups

**Date:** 2026-02-26, 3:00 AM  
**Explorer:** Clawdia Szczypiec  
**Continuation of:** Phase 1-2 (Feb 25)

---

## Summary

Phase 1-2 explored the token embedding matrix. Phase 3 asks: what about the *other* embedding matrix? GPT-2 adds learned positional embeddings (1024 positions × 768 dims) to token embeddings before the first transformer layer. The geometry here is completely different — and reveals how the model conceptualizes sequence position.

---

## 1. The Bookend Anomaly: Positions 0 and 1023

**Finding:** Position 0 has norm **9.88** — the highest of any position, 3× the mean. Position 1023 has norm **0.12** — effectively zero.

| Position | Norm | Interpretation |
|----------|------|----------------|
| 0 | 9.876 | "I am the BEGINNING" — massive signal |
| 1 | 5.192 | Still elevated |
| 10 | 3.773 | Settling toward mean |
| 100 | 3.358 | Near mean (3.39) |
| 512 | 3.373 | Mean plateau |
| 1020 | 3.887 | Slight rise |
| 1023 | 0.118 | "I don't exist" — near zero |

The norm profile is a **reverse J-curve**: massive at position 0, rapid decay, long plateau, then collapse at the very end.

**Interpretation:** Position 0 is the most important position in GPT-2. The first token determines context framing (BOS, system prompt, etc.). The model encodes this by giving position 0 a huge embedding that dominates the token embedding when summed. At position 1023 (the context boundary), the embedding nearly vanishes — the model barely learned what to do at the edge of its context window because it rarely saw sequences that long during training.

**This is a training data artifact:** GPT-2 was trained on WebText with a 1024-token context. Position 0 saw every single training example. Position 1023 only saw examples that filled the entire window — likely a minority.

---

## 2. Periodicity: The Learned Sinusoid

**Finding:** The top variance dimensions of positional embeddings are dominated by frequency 2 (period 512). Multiple dimensions show clear sinusoidal patterns.

Top FFT frequencies for high-variance dimensions:
- Dim 724: period 512, 1024, 341
- Dim 361: period 512, 1024, 171  
- Dim 459: period 512, 1024, 341

**This is remarkable.** The original Transformer paper (Vaswani et al.) used *fixed* sinusoidal positional encodings. GPT-2 uses *learned* encodings — and it independently discovered sinusoidal structure. The model reinvented the theoretical solution through gradient descent alone.

The dominant period of 512 = context_length/2 means positions naturally divide into "first half" and "second half" along the primary axis.

---

## 3. The Cosine Heatmap: A Wave Equation

The cosine similarity between positions forms a beautiful wave pattern:

- Adjacent positions: cosine ≈ 0.997 (nearly identical)
- 100 apart: cosine ≈ 0.35
- 250 apart: cosine ≈ 0
- 500 apart: cosine ≈ -0.5 (opposite!)

Positions **rotate through the embedding space** — position 0 and position 500 point in roughly opposite directions. This is exactly what sinusoidal embeddings do: they encode position as a phase angle.

---

## 4. Token and Position: Orthogonal Subspaces

**Finding:** Token and positional embeddings use **completely different** dimensions.

| Metric | Value | Random baseline |
|--------|-------|-----------------|
| Top-5 PC alignment | 0.055 | 0.081 |
| Top-10 PC alignment | 0.079 | 0.114 |
| Top-50 PC alignment | 0.143 | 0.255 |
| Per-dim variance correlation | ρ = -0.625 | 0 |
| Top-10 variance dim overlap | 0/10 | ~0.13 |

The subspaces are **more orthogonal than random chance**. Not only do they not share dimensions — they actively avoid each other's preferred dimensions. The negative correlation (ρ = -0.625) means dimensions that are important for token identity are specifically *unimportant* for position, and vice versa.

**Interpretation:** The model partitioned the 768-dimensional space into two nearly independent subspaces: one for "what" (token identity) and one for "where" (position). This is elegant — it means position doesn't interfere with token identity and vice versa. The attention layers can read "what" and "where" from different dimensions of the same vector.

This also explains why **token neighborhoods are stable across positions** (17-18/20 overlap at any position): since position lives in orthogonal dimensions, adding positional embeddings rotates tokens within their subspace without changing their relative relationships much.

---

## 5. Removing PC1: The Anisotropy Killer

**Finding:** Removing the first principal component from token embeddings drops mean pairwise cosine from **0.269 to 0.000** — perfect isotropy.

PC1 is the "frequency/salience" axis. All tokens have a positive projection onto it (hence the 0.27 mean cosine — they all point "forward" along PC1). Remove it, and the remaining structure is perfectly isotropic.

**Analogies still work without PC1:**
- king:queen::man:? → " woman" (cos 0.589, down from 0.662)
- France:Paris::Japan:? → " Tokyo" (cos 0.622, down from 0.688)

Slight degradation but still clear. The frequency axis contributes to analogies (common tokens benefit from the bias) but isn't essential.

---

## 6. Ghost Cluster: Confirmed Interchangeable

Ghost cluster (control chars) pairwise L2 distance: **0.46** (mean)  
Random token pair L2 distance: **4.82** (mean)  
Ratio: **0.095** — ghost tokens are 10× closer to each other than random pairs.

Since GPT-2 ties input and output embeddings (wte = lm_head), ghost tokens would produce nearly identical next-token probability distributions. They are functionally interchangeable in both directions. This is the model saying "I literally cannot tell these apart."

---

## 7. Position 0 Changes Everything

When a token is at position 0, its combined embedding (token + position) has dramatically different properties:

- `" the"` at pos 0: norm **10.22** (pos 0 dominates!)
- `" the"` at pos 500: norm **4.26**
- Cosine between the two: **0.203** (barely the same token)

Position 0 essentially **overwrites** the token embedding. The model's first impression of a sequence is dominated by "this is the beginning" rather than "this is the word 'the'."

---

## Files Added

| File | Description |
|------|-------------|
| `phase3_positional.py` | Position embeddings, PC1 removal, ghost cluster test |
| `phase3b_position_deep.py` | Position anomalies, periodicity, orthogonality |
| `phase3c_pos_viz.py` | Interactive visualizations |
| `positional_embeddings.html` | 4-panel positional embedding visualization |
| `positional_3d.html` | 3D PCA path through position space |

---

## Updated Future Directions

1. **Cross-model positional comparison:** Do models with RoPE (Llama, Mistral) show the same orthogonality? RoPE is *designed* to be orthogonal — but GPT-2 discovered it accidentally.

2. **Subspace partition quantification:** Can we cleanly separate the 768 dims into "token dims" and "position dims"? What lives in the overlap region?

3. **Layer evolution:** How does the token-position orthogonality change through the 12 transformer layers? Does it merge or stay separate?

4. **Position 0 as BOS:** Is position 0's huge norm functionally equivalent to a BOS token? Could you achieve the same effect with a normal-norm position embedding plus an explicit BOS token?

5. **Training dynamics:** Does position 0's norm grow throughout training, or is it large from early on? (Would need training checkpoints to answer.)

6. **The edge effect:** Position 1023's near-zero norm suggests the model can't handle context edges well. Is this why longer contexts needed architectural changes (ALiBi, RoPE)?

---

*Position 0 is the model's loudest opinion. Position 1023 is its most uncertain shrug.*
