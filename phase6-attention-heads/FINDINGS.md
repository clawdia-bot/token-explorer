# Phase 6: Attention Head Analysis in Pythia L6

## The Question
Phase 5 found that Pythia-70m's output alignment jumps from 0.16 to 0.95 in a single layer (L6). What drives this phase transition?

## Key Findings

### 1. MLP is the quiet hero, attention gets the headlines

Decomposing L6's contribution:

| Component | Output Alignment | Δ from residual |
|-----------|:---:|:---:|
| Residual only (skip L6) | 0.614 | — |
| Residual + attn only | 0.875 | +0.261 |
| Residual + MLP only | **0.922** | **+0.308** |
| Full (residual + attn + MLP) | 0.924 | +0.310 |

The MLP alone gets you 99.3% of L6's total boost. Attention contributes +0.261 in isolation but only +0.002 on top of MLP. The MLP has learned to project the residual stream into output space almost perfectly. Attention is redundant when MLP is present — but when MLP is absent, attention can do most of the job too.

This is **functional redundancy**: both components independently learned the prediction-projection skill.

### 2. No single head matters (distributed computation)

Ablating any single head drops alignment by less than 0.001. The information is spread across all 8 heads, and the MLP provides a safety net. This is a robustly engineered system — no single point of failure.

### 3. All L6 heads point toward output space

Every L6 head output aligns positively with next-token output embeddings:

| Head | Output Alignment | Norm (energy) | Pattern |
|------|:---:|:---:|:---|
| H0 | 0.627 | 1.9 | Self-attention + verb focus |
| **H1** | **0.916** | **6.0** | **Conjunction/context aggregation** |
| H2 | 0.707 | 1.6 | Verb attention |
| H3 | 0.758 | 0.6 | BOS sink (83% to pos 0) |
| H4 | 0.805 | 1.0 | BOS sink (75% to pos 0) |
| **H5** | **0.886** | **4.2** | **Verb/predicate focus** |
| H6 | 0.789 | 0.6 | BOS sink (92% to pos 0) |
| **H7** | **0.904** | **7.8** | **Previous-token (bigram) head** |

The power trio: **H7** (highest energy, previous-token pattern), **H1** (highest alignment, context aggregation), and **H5** (high alignment, verb focus). Together they dominate L6's attention contribution.

### 4. Clear functional specialization

Three head types in L6:

- **BOS sink heads** (H3, H4, H6): 75-92% attention to position 0. Low energy output. These dump attention mass into a "garbage" position — a known phenomenon in transformers where the model needs somewhere to put attention when there's nothing useful to attend to.

- **Content heads** (H0, H1, H2, H5): Attend to semantically relevant positions. H1 locks onto conjunctions ("and"), H0/H2/H5 attend to verbs. These gather context for prediction.

- **Previous-token head** (H7): Highest entropy (most distributed), but primarily attends to the immediately preceding token. Classic bigram feature — it provides "what came right before" which is the strongest single signal for next-token prediction.

### 5. L5 is nearly a no-op

Head output norms in L5 are tiny (0.15-0.79) compared to L6 (0.51-8.14). L5 barely modifies the residual stream. The model effectively uses only 5 active layers — L5 might as well not exist.

### 6. The residual stream is a slow build, then a cliff

Layer-by-layer alignment progression:
```
emb → L1:  +0.06 (initial feature extraction)
L1  → L2:  -0.00 (nothing)
L2  → L3:  +0.01 (slight refinement)
L3  → L4:  +0.02 (building)
L4  → L5:  +0.02 (building)
L5  → L6:  +0.31 (THE prediction layer)
```

Layers 1-5 spend their budget building a representation. L6 spends its budget projecting that representation into prediction space. The architecture naturally separates "understand" from "predict."

## Implications

1. **For mechanistic interpretability**: The MLP-attention redundancy in L6 suggests the prediction projection is more like a learned linear map than a complex circuit. Both components converge on the same function independently.

2. **For model efficiency**: L5 does almost nothing. L2 does almost nothing. A 4-layer Pythia might work nearly as well — worth testing with layer ablation.

3. **For scaling**: Does this "cliff" pattern persist in larger Pythia models (410m, 1b)? If the final layer always handles projection, it would mean the "understanding" layers scale but the "prediction" layer stays fixed.

## Next Steps
- Layer ablation: Remove L2 and L5 entirely, measure perplexity impact
- Scaling comparison: Run same analysis on Pythia-410m — does the cliff still exist at the last layer?
- MLP decomposition: What's the MLP actually doing? Is it a simple linear projection or something nonlinear?
- Induction head search: Are there induction heads in the earlier layers that feed L6?
