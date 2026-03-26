# Phase 3B: RoPE Geometry and First-Layer Positional Effects

## Question

The original Phase 3 question only worked for GPT-2 because it asked about a learned absolute position matrix. The active comparison set is mostly RoPE models, so Phase 3B asks a different question:

1. What relative-position kernel is hard-coded by each model's rotary frequencies?
2. How much of each head is actually rotary?
3. How much does RoPE alone perturb first-layer `q/k` geometry and exact-probe score structure?

## Scope

- **Compatible models:** Pythia-70m, SmolLM2-135M, Qwen2.5-0.5B
- **Skipped model:** GPT-2, because it uses learned absolute position embeddings rather than RoPE
- **Probe set:** the 20 exact comparison probes already used elsewhere in the repo
- **Layer:** first attention layer only

## Direct Observations

### 1. The three RoPE models do not allocate position the same way

| Model | RoPE theta | Rotary dim | Head dim |
|------|------------:|-----------:|---------:|
| Pythia-70m | 10,000 | 16 | 64 |
| SmolLM2-135M | 100,000 | 64 | 64 |
| Qwen2.5-0.5B | 1,000,000 | 64 | 64 |

Pythia only rotates one quarter of each head. SmolLM2 and Qwen rotate the full head.

### 2. The operator-level kernels are all smooth near zero, but their long-range behavior differs

- Pythia starts with the fastest local decay because only a small rotary subspace is carrying the positional phase.
- SmolLM2 stays broadly positive across the sampled offsets but its first-layer probe structure degrades sharply later.
- Qwen keeps a comparatively smooth long-range kernel despite very large sampled offsets.

None of the three sampled kernels crossed below zero on the stored offset grid.

### 3. Qwen preserves first-layer probe-score ordering extremely well

At the largest sampled offset:

- Qwen query drift: `0.317`
- Qwen key drift: `0.924`
- Qwen probe-score Spearman correlation: `0.977`

So Qwen's queries rotate substantially, but the relative compatibility structure over the exact probe inventory is still mostly preserved.

### 4. SmolLM2's first-layer score structure breaks much harder

At the largest sampled offset:

- SmolLM2 query drift: `0.047`
- SmolLM2 key drift: `-0.102`
- SmolLM2 probe-score Spearman correlation: `-0.700`

This is not just mild degradation. On this metric, the first-layer probe-score ordering is substantially overturned.

### 5. Pythia sits in between

At the largest sampled offset:

- Pythia query drift: `0.282`
- Pythia key drift: `0.597`
- Pythia probe-score Spearman correlation: `0.496`

Pythia preserves some of the baseline structure, but much less than Qwen and more than SmolLM2.

## Interpretations

- **RoPE is not one thing.** The architectural label hides meaningful choices: partial vs full-head rotation and dramatically different `theta` values.
- **Operator smoothness does not fully predict probe stability.** SmolLM2's kernel looks respectable at the operator level, yet its first-layer probe-score ordering collapses harder than Pythia's.
- **Qwen appears to combine large-context RoPE with unusually stable first-layer score structure.** That makes it the clearest counterexample to the idea that long-range rotary phase must scramble early semantic compatibility.

## Limitations

- This phase measures **first-layer projected `q/k` behavior**, not whole-model forward behavior.
- The exact probe inventory is English-centric and small.
- The long-offset samples are logarithmic rather than exhaustive over the whole context window.
- These findings compare RoPE models to each other; GPT-2 belongs to the absolute-position Phase 3 track instead.

## Files

- `phase3b_rope.py`: RoPE analysis and first-layer probe metrics
- `phase3b_rope_charts.py`: per-model charts
- `phase3b_rope_browser.py`: browser across saved models
- `rope-results/<model>/results.json`: structured outputs

## What This Adds To Phase 3

Phase 3A says: "GPT-2 learned a positional geometry in an explicit matrix."  
Phase 3B says: "RoPE models embed positional geometry in a rotational operator, and those operators have materially different downstream effects even before the first attention pattern is formed."
