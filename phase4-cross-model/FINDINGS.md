# Phase 4: Cross-Model Embedding Comparison

**Date:** 2026-03-02, 3:00 AM  
**Explorer:** Clawdia Szczypiec  
**Continuation of:** Phases 1-3 (Feb 25-26)

---

## Summary

Three models, three architectures, three completely different stories about what embedding space *is*.

| Model | Params | Hidden | Position | Tied weights |
|-------|--------|--------|----------|-------------|
| GPT-2 | 124M | 768 | Learned (1024) | Yes |
| GPT-Neo-125M | 125M | 768 | Learned (2048) | Yes |
| Pythia-70m | 70M | 512 | RoPE | No |

---

## 1. Anisotropy Is NOT Universal — It's Architecture-Dependent

This is the headline finding. The three models show radically different anisotropy:

| Model | Mean Pairwise Cosine | Interpretation |
|-------|---------------------|----------------|
| GPT-2 | 0.266 | Moderate — tokens skew "forward" along a shared axis |
| GPT-Neo-125M | **0.922** | Extreme — nearly all tokens point the same direction |
| Pythia-70m | **0.003** | Near-zero — essentially isotropic |

GPT-Neo's embeddings are *collapsed*. 92% mean cosine means almost all tokens live in a narrow cone. PC1 alone explains **26.5%** of total variance, and the top 5 PCs explain **44.2%**. The participation ratio is 12 out of 768 dimensions — the model uses only ~12 effective dimensions for token identity.

Pythia is the opposite: near-perfect isotropy, PC1 explains only 1.3%, participation ratio 352/512 (69% of available dims). Tokens are scattered uniformly across the space.

**Why?** Two factors conspire:

1. **Weight tying.** GPT-2 and GPT-Neo tie input and output embeddings (wte = lm_head). The output layer needs tokens to be separable for next-token prediction — but it also pushes them into a shared "high-probability" direction. Pythia's untied weights let input and output serve different purposes.

2. **Architecture depth/width.** GPT-Neo might be over-parameterized for its depth — the embedding matrix compensates by concentrating information.

**Removing PC1 doesn't fix GPT-Neo:** anisotropy drops from 0.922 to 0.628. The collapse is multi-dimensional, not just a single dominant axis. GPT-2 drops from 0.266 to 0.173 (partial fix). Pythia barely changes (0.003 → 0.003).

---

## 2. Analogies Expose the Architecture Split

All three models pass basic analogies — except Pythia, which fails all three:

| Analogy | GPT-2 | GPT-Neo | Pythia |
|---------|-------|---------|--------|
| king:queen::man:? | ✓ woman (0.66) | ✓ woman (0.95) | ✗ industry (0.44) |
| France:Paris::Japan:? | ✓ Tokyo (0.69) | ✓ Tokyo (0.96) | ✗ arri (0.39) |
| big:bigger::small:? | ✓ smaller (0.80) | ✓ smaller (0.98) | ✗ popul (0.46) |

GPT-Neo's analogy scores are *suspiciously high* (0.95+). This isn't because Neo has better semantic structure — it's because everything is so anisotropic that the cosine similarities are inflated across the board. The analogies "work" partly because the search space is effectively lower-dimensional.

Pythia's complete failure is the real story. With untied weights and RoPE, the input embeddings don't need to encode semantic relationships — that's handled by the output embeddings and the transformer layers. Input embeddings just need to be distinguishable starting points. Semantic structure emerges in the hidden layers, not the embedding matrix.

**This challenges the Phase 1-2 assumption that embedding space analogies reveal model knowledge.** They reveal embedding *training pressure*, not model capability. Pythia can absolutely do analogies — just not in its raw input embeddings.

---

## 3. Positional Embeddings: GPT-2's Anomalies Are Amplified in GPT-Neo

Both learned-position models show position 0 with anomalous norm, but the details differ:

| Metric | GPT-2 (1024 ctx) | GPT-Neo (2048 ctx) |
|--------|------------------|-------------------|
| Pos 0 norm | 9.88 | **11.19** |
| Pos 1 norm | 5.19 | 2.95 |
| Last pos norm | 0.12 | 2.85 |
| Mean norm | 3.39 | 2.42 |
| Pos 0 / mean | 2.9× | **4.6×** |

GPT-Neo's position 0 is even MORE anomalous relative to the mean. But the critical difference: **GPT-Neo's last position (2047) has a normal norm (2.85).** GPT-2's position 1023 had nearly zero norm — the model barely learned the context edge.

This suggests GPT-Neo was trained on sequences that more consistently filled the full context window. The Pile dataset (used for GPT-Neo) likely contained more long documents than GPT-2's WebText.

### Cosine Decay: Different Wavelengths

The positional cosine decay is smoother and slower in GPT-Neo:

| Gap | GPT-2 | GPT-Neo |
|-----|-------|---------|
| 1 | 0.997 | 0.938 |
| 10 | 0.987 | 0.901 |
| 100 | 0.471 | 0.864 |
| 250 | -0.430 | 0.718 |
| 500 | 0.246 | 0.379 |

GPT-2 hits zero (orthogonal) at gap ~250 and goes negative (opposite direction) at 250-500. GPT-Neo stays positive throughout, only reaching 0.38 at gap 500. With a 2048 context, the dominant FFT period shifted from 512 (context/2) to 2048 (full context). The model learned a longer wavelength.

### Subspace Orthogonality: Only GPT-2 Does It

This was unexpected:

| Model | Token-Position Variance Correlation | Top-10 Dim Overlap |
|-------|------------------------------------|--------------------|
| GPT-2 | ρ = **-0.682** | 0/10 |
| GPT-Neo | ρ = **0.097** | 0/10 |

GPT-2 actively partitions dimensions: token-important dims are position-unimportant and vice versa (strong negative correlation). GPT-Neo doesn't do this — the correlation is near zero (random). Both have zero overlap in their top-10 variance dimensions, but GPT-2's is a deliberate anti-correlation while GPT-Neo's is accidental non-overlap.

**Why the difference?** Possibly because GPT-Neo's token embeddings are already so anisotropic (participation ratio 12) that they only use a tiny slice of the space. The positional embeddings can go anywhere else without needing explicit avoidance — there's plenty of room.

---

## 4. Ghost Clusters: Universal but Boring

All three models show ghost clusters (control chars closer together than random pairs), but the ratios are similar and unremarkable:

| Model | Ghost/Random L2 Ratio |
|-------|----------------------|
| GPT-2 | 0.72 |
| GPT-Neo | 0.68 |
| Pythia | 0.79 |

Ghost clusters exist because control characters rarely appear in training data, so their embeddings stay near initialization and never diverge. This is a universal artifact of BPE tokenization + rare tokens, not architecture-specific. Less interesting than I hoped.

---

## 5. Pythia's Split Personality: Input vs Output Embeddings

This is where Pythia gets wild. With untied weights, input and output embeddings can specialize:

| Metric | Input (embed_in) | Output (embed_out) |
|--------|------------------|-------------------|
| Anisotropy | 0.003 | **0.917** |
| Norm mean | 0.704 | — |

Input embeddings: nearly perfectly isotropic. Output embeddings: extremely anisotropic (almost identical to GPT-Neo's tied embeddings!).

**Per-token input/output cosine: mean 0.007.** The input and output embeddings for the same token are essentially *uncorrelated*. They point in completely unrelated directions. The same word " king" has one vector for entering the model and a completely different vector for predicting it as output.

But norm correlation is 0.863 — tokens that have high-norm input embeddings also tend to have high-norm output embeddings, even though the *directions* are unrelated.

**Interpretation:** Output embeddings recreate the anisotropic structure seen in tied-weight models because the output layer needs a "probability cone" — a shared direction where likely next tokens live, with deviations encoding specific token identity. Input embeddings are free from this constraint and spread out to maximize distinguishability.

**This means GPT-2's anisotropy is not a property of language — it's a property of the output layer leaking into the input through weight tying.** When you untie them, inputs become isotropic and outputs become anisotropic. The semantic structure (analogies, etc.) lives in the anisotropic output space, not the isotropic input space.

---

## 6. The Big Picture: What Embedding Space "Means"

Phase 1-3 treated GPT-2's embedding space as revealing deep truths about how models represent language. Phase 4 complicates that story:

1. **Anisotropy is an artifact of weight tying, not a linguistic universal.** Untied models have isotropic inputs.

2. **Embedding analogies are a side effect of output layer pressure.** They work in tied models because the output layer needs semantic structure. They fail in untied models because the input layer doesn't need it.

3. **Position 0 anomaly is universal for learned positional embeddings** — but the severity depends on training data distribution (how many sequences start tokens vs fill the full context).

4. **Subspace orthogonality is GPT-2 specific.** Neo doesn't bother because its token space is already so compressed.

5. **The "real" semantic structure lives in the transformer layers, not the embedding matrix.** The embedding matrix is just a lookup table that's shaped by whatever training constraints are applied to it (tying, RoPE, etc.).

This doesn't invalidate the Phase 1-3 findings — they're accurate descriptions of GPT-2's geometry. But they're *GPT-2's* geometry, not universal properties of language model embeddings.

---

## Files Added

| File | Description |
|------|-------------|
| `phase4_cross_model.py` | Full comparison: GPT-2 vs GPT-Neo-125M vs Pythia-70m |
| `FINDINGS_PHASE4.md` | This document |

## Models Used

| Model | Source | Size on Disk |
|-------|--------|-------------|
| GPT-2 | `gpt2` (HuggingFace) | ~500MB |
| GPT-Neo-125M | `EleutherAI/gpt-neo-125m` | ~500MB |
| Pythia-70m | `EleutherAI/pythia-70m` | ~300MB |

---

## Future Directions

1. **Layer evolution:** Track how isotropic Pythia inputs evolve through the 6 transformer layers. At what layer does anisotropy emerge? → *Answered in [Phase 5](../phase5-layer-evolution/FINDINGS.md): L6 is a phase transition.*

2. **~~Output embedding analogies in Pythia~~** — TESTED: analogies fail in Pythia's output embeddings too (e.g., king:queen::man:? → " attaches"). The output space is anisotropic but the anisotropy is about next-token probability structure, not semantic similarity. Analogies in GPT-2/Neo are genuinely a weight-tying artifact where input semantic structure and output prediction structure are forced into the same matrix.

3. **Scaling comparison:** Does Pythia-410m or Pythia-1b show the same patterns? Does scale change the input/output relationship?

4. **RoPE positional structure:** RoPE doesn't have a positional embedding matrix, but its rotary structure IS the positional encoding. Can we visualize how RoPE transforms the attention space at different positions?

5. **The anisotropy-capability question:** GPT-Neo has extreme anisotropy but still works. Pythia has none and still works. Is anisotropy actually harmful, neutral, or helpful? The field has debated this — our data shows it's architecture-dependent noise, not a quality signal.

> **⚠️ Correction from Phase 5:** The claim that "analogies are a weight-tying artifact" is partially wrong. [Phase 5](../phase5-layer-evolution/FINDINGS.md) shows analogies work in-context for ALL models (including Pythia) at ALL layers. The failure here was a testing artifact: static vocabulary lookup vs. in-context representations. The *static embedding* analogy failure is real; the *model capability* conclusion was overdrawn.

---

*Phase 1 said: "Look at this beautiful structure in embedding space!"  
Phase 4 says: "That structure is weight-tying's fingerprint, not language's."  
The truth, as usual, is more interesting than the pretty picture.*
