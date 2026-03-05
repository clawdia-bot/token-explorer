# Phase 7: MLP Decomposition & Layer Ablation

## The Questions
1. Is L6's MLP (which does 99.3% of the prediction work) essentially a linear projection, or does GELU nonlinearity matter?
2. Which layers are load-bearing vs. decorative? What's the minimal circuit?

## Key Findings

### 1. GELU is absolutely essential — L6 MLP is NOT linear

Replacing GELU with identity in L6 alone:
- Alignment drops 0.10–0.25 (massive)
- Perplexity explodes from ~600 to 1M–55M

The MLP needs its nonlinearity to do the prediction projection. This isn't a simple rotation into output space — it's a **conditional computation** where GELU gates which neurons contribute.

**Plot twist:** Making ALL MLPs linear (every layer) actually hurts alignment *less* than making only L6 linear:

| Configuration | Alignment | Perplexity |
|---|:---:|:---:|
| Normal | 0.900 | ~600 |
| Only L6 linear | 0.716 | ~19M |
| All layers linear | 0.870 | ~83K |

When all layers are linear, the entire model becomes one big linear map (attention is the only nonlinearity left). The composed linear transformation from all 6 layers together can partially compensate. But when only L6 is linear while earlier layers are nonlinear, the intermediate representations are shaped for a nonlinear L6 that isn't there. **It's worse to be half-linear than fully linear.**

### 2. L6 MLP is genuinely nonlinear (not low-rank)

The composed linear approximation (W_down @ W_up) of L6's MLP:
- Cosine similarity with actual MLP output: **0.27–0.58** (terrible)
- The linear map explains barely a quarter to half of what MLP actually computes

Effective rank of W_down @ W_up:
- 5 singular values capture 50% of variance (highly concentrated)
- 191 participation ratio out of 512 dimensions
- But this low rank describes the *linear component*, not the full computation

GELU creates a **data-dependent gating** pattern where different inputs activate different subsets of the 2048 intermediate neurons. The "projection into output space" isn't a fixed linear map — it's a different linear map for every input, selected by which neurons GELU lets through.

### 3. Neuron 348: The monster

L6 MLP has one neuron (N348) that dominates everything:
- Mean activation: **41.9** (next highest: 19.0)
- Max activation: **93.7**
- Fires on 92% of tokens

It fires almost always and fires hard. Its output direction promotes seemingly random tokens (" XXX", " ord", " Robertson") — but remember, the *activation magnitude* modulates this. It likely functions as a general "confidence booster" that scales the residual toward the output manifold.

### 4. 10.4% of L6 neurons are dead

213 of 2048 intermediate neurons never fire above 0.1. These are wasted parameters — GELU killed them during training and they never recovered. This is a known phenomenon in ReLU/GELU networks.

41 neurons (2%) are always-on, firing on >90% of tokens. These act as **constant bias directions** — they shift every prediction in the same direction regardless of input.

### 5. Neuron output directions don't point at specific tokens

The top neurons' output columns have modest cosine similarity with output embeddings (0.5–0.7 for promoted tokens). They don't cleanly map to "this neuron predicts word X." Instead, they push in directions that are mildly correlated with clusters of tokens. The prediction emerges from the **superposition** of ~1800 active neurons, not from any individual one.

The suppressed tokens are universally byte-pair artifacts (broken UTF-8 sequences). Every active neuron pushes *away* from garbage tokens — a collective "don't predict nonsense" signal.

### 6. Pythia-70m is basically a two-layer model (L0 + L5)

Layer ablation results:

| Layers Removed | Alignment | Perplexity |
|---|:---:|:---:|
| None (full) | 0.903 | 456 |
| L1 | 0.908 | 4,087 |
| L2 | 0.908 | 2,069 |
| L3 | 0.901 | 597 |
| L4 | 0.898 | 725 |
| **L0** | **0.823** | **71,372** |
| **L5** | **0.594** | **157M** |

L1, L2, L3, L4 are nearly decorative for alignment. L0 matters somewhat. L5 is catastrophically important (it's the final transformer layer, L6 in 1-indexed).

Greedy removal order: L1 → L3 → L4 → L2 → L0 → L5

**The minimal circuit is embedding → L5 → output:**
- Alignment: 0.911 (better than full model's 0.903!)
- Perplexity: 121,498 (bad, but it works)

L5 alone achieves *higher alignment than the full model*. The other 5 layers slightly hurt alignment while massively helping perplexity. This means layers 0–4 build nuanced representations that improve *which* token is predicted (perplexity) without improving *how well* the prediction aligns with the output embedding (alignment).

### 7. The alignment paradox

Removing layers 1–4 actually *improves* alignment (0.903 → 0.910). This sounds contradictory — how can removing computation help?

The answer: alignment measures whether the hidden state points toward the predicted token's embedding. Middle layers build complex representations that deviate from the output manifold to encode richer features (syntax, semantics, long-range dependencies). These features help predict the *right* token (lower perplexity) but they do it by creating representations that are less geometrically aligned with any single output embedding.

**Layers 0–4 trade alignment for accuracy.** L5 then projects everything back into output space. But if you skip the middle layers, L5 just projects the near-raw embedding — which is already somewhat aligned with outputs — into output space. Less information, but simpler geometry.

## Implications

1. **GELU isn't optional.** The MLP's power comes from data-dependent neuron selection, not from its weight matrices alone. Any attempt to "distill" or "linearize" the MLP would need to preserve this gating behavior.

2. **Pythia-70m is wildly over-parameterized for alignment** but appropriately sized for perplexity. 5 of 6 layers barely affect alignment. The real information processing happens at the bookends.

3. **Dead neurons suggest training inefficiency.** 10% of L6's capacity is wasted. Techniques like neuron resurrection or different initialization could potentially improve this model at the same parameter count.

4. **The output space is not a fixed target.** Alignment is a blunt metric — it can go up when you remove layers because you're removing the complexity that makes predictions *accurate*. Perplexity is the real measure; alignment is just a geometric diagnostic.

## Next Steps
- Visualize neuron activation patterns across tokens (which neurons fire for which words?)
- Compare L5 neuron statistics with L0 (the two load-bearing layers)
- Check if dead neurons in L5 overlap with dead neurons in other layers
- Scaling: does Pythia-410m also have a single dominant layer, or does computation distribute more evenly?
