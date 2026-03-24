# Cross-Model Embedding Comparison — Findings

4 models compared: GPT-2, Pythia-70m, SmolLM2-135M, Qwen2.5-0.5B

Larger models (Gemma 2-2B, Llama 3.2-1B, Phi-3.5-mini) were also tested but dropped — they don't surface findings beyond what these four cover, and they take 10-50x longer to process.

## 1. Isotropy Spectrum

| Model | Anisotropy | PR / dim | Eff. dim / dim | Mean norm |
|-------|-----------|----------|----------------|-----------|
| SmolLM2-135M | **0.448** | 67/576 (12%) | 342/576 (59%) | 3.179 |
| GPT-2 | 0.269 | 428/768 (56%) | 587/768 (76%) | 3.959 |
| Qwen2.5-0.5B | 0.153 | 447/896 (50%) | 778/896 (87%) | 0.463 |
| Pythia-70m | **0.004** | 352/512 (69%) | 440/512 (86%) | 0.704 |

**Takeaway:** Anisotropy varies wildly across models of similar size. SmolLM2 is extremely anisotropic (nearly half the pairwise cosines are positive), while Pythia is near-perfectly isotropic. No correlation with model size — SmolLM2 (135M) is worse than GPT-2 (124M).

SmolLM2 has a notably low participation ratio (12%) despite decent entropy-based effective dims — a few PCs dominate but the long tail is spread out.

## 2. Ghost Cluster Universality

| Model | Ghost cluster size | Mean pairwise cosine | Vocab size |
|-------|-------------------|---------------------|------------|
| Qwen2.5-0.5B | **4,340** | 0.983 | 151,936 |
| GPT-2 | 61 | 0.993 | 50,257 |
| Pythia-70m | none | — | 50,304 |
| SmolLM2-135M | none | — | 49,152 |

**Takeaway:** Ghost clusters are NOT universal. Qwen's massive multilingual vocabulary (151K tokens) means ~2.9% of tokens are rarely seen and collapse to near-identical vectors. GPT-2 has a smaller cluster at byte-token indices. Pythia and SmolLM2 have none — their training likely covers more of the vocabulary, or their optimization prevents collapse.

Newline is never in the ghost cluster in any model.

## 3. Analogy Scorecard (12 standardized analogies)

| Model | Score | Notes |
|-------|-------|-------|
| GPT-2 | **11/12** | Only misses antonym (hot:cold::up:?) |
| Pythia-70m | 10/12 | Misses capital (Japan) and antonym |
| SmolLM2-135M | 7/12 | Case errors, past tense failures |
| Qwen2.5-0.5B | 7/12 | Similar pattern to SmolLM2 |

**Takeaway:** GPT-2's tied-weight embeddings are best at static analogies — the oldest, smallest model wins. Pythia (untied weights) is close behind. SmolLM2 and Qwen struggle with gender and tense analogies.

Hypothesis: tied weights force the embedding layer to do double duty (input + output), which creates stronger linear structure. The antonym analogy (hot:cold::up:?) fails everywhere — antonymy doesn't produce clean linear offsets.

## 4. Neighborhood Jaccard Overlap (top-10 neighbors, 20 probe tokens)

| Pair | Jaccard |
|------|---------|
| GPT-2 ↔ Pythia | 0.570 |
| GPT-2 ↔ SmolLM2 | 0.558 |
| Pythia ↔ SmolLM2 | 0.513 |
| SmolLM2 ↔ Qwen | 0.255 |
| GPT-2 ↔ Qwen | 0.250 |
| Pythia ↔ Qwen | 0.236 |

**Takeaway:** GPT-2, Pythia, and SmolLM2 agree strongly on neighborhoods (0.51-0.57) — they share the same tokenizer family. Qwen's different tokenizer produces different neighborhoods (0.24-0.26 with everyone).

## 5. Outlier Migration (Spearman rank correlation of token norms)

| Pair | Spearman rho |
|------|-------------|
| GPT-2 ↔ SmolLM2 | **0.777** |
| GPT-2 ↔ Qwen | 0.618 |
| SmolLM2 ↔ Qwen | 0.540 |
| Pythia ↔ everyone | **~0** (uncorrelated) |

**Takeaway:** GPT-2 and SmolLM2 agree strongly on which tokens are "weird" (high norm) — rho 0.777. Pythia is uncorrelated with everyone because its near-isotropy compresses the norm distribution — there are no meaningful outliers.

## Cross-Cutting Themes

1. **Tied vs untied weights matters more than model size.** Tied-weight models (GPT-2, SmolLM2) have stronger static embedding structure: better analogies, more anisotropy, clearer outliers. Pythia's untied weights yield a flatter, more isotropic space.

2. **Tokenizer family determines neighborhood structure.** GPT-2/Pythia/SmolLM2 agree on neighborhoods (same tokenizer family). Qwen's different tokenizer produces different neighborhoods. The tokenizer is a stronger predictor of embedding geometry than model size.

3. **Large multilingual vocabs produce ghost clusters.** Qwen's 151K vocab has 4,340 ghost tokens — rarely-seen multilingual tokens that collapse. Models with ~50K vocabs have small or no ghost clusters.

4. **Anisotropy is not obviously harmful.** SmolLM2 is highly anisotropic (0.448) but still passes 7/12 analogies. Pythia is perfectly isotropic (0.004) and gets 10/12. The relationship between isotropy and embedding quality is not straightforward.
