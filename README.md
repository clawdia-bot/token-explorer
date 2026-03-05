# Token Embedding Explorer 🔬

**Exploring the geometry of language model embeddings — from raw token vectors to mechanistic interpretability.**

An ongoing investigation into how transformer models organize their internal representations. Started with GPT-2's embedding matrix and followed the trail through positional encodings, cross-model comparison (GPT-Neo, Pythia), layer-by-layer evolution, attention head decomposition, and MLP ablation.

## Key Findings

### The embedding matrix has rich structure before any attention

- Token norms encode frequency: common tokens cluster near the origin, rare tokens orbit far (Phase 1)
- Control characters collapse to identical "I don't know" vectors — except `\n`, which earned its own identity (Phase 2)
- Raw embedding analogies work: king:queen::man:woman, with no fine-tuning needed (Phase 2)

### Anisotropy is a weight-tying artifact, not a linguistic universal

- GPT-2 and GPT-Neo (tied weights) show moderate anisotropy; Pythia (untied weights) has near-zero (Phase 4)
- Embedding analogies only work in tied-weight models — Pythia's embeddings are isotropic and analogy-free (Phase 4)
- PC1 (the "frequency axis") accounts for nearly all anisotropy (Phase 3)

### Pythia-70m is effectively a 2-layer model

- Layer 6 is a phase transition: output-vocabulary alignment jumps from 0.16 to 0.95 in a single layer (Phase 5)
- MLP does 99.3% of the work; attention heads are functionally redundant (Phase 6)
- Only L0 and L5 are load-bearing — ablating middle layers barely hurts performance (Phase 7)
- Neuron 348 is a monster: activation 41.9, fires on 92% of tokens (Phase 7)

## Phases

| Phase | Directory | Focus |
|-------|-----------|-------|
| 1 | [`phase1-norms-and-structure/`](phase1-norms-and-structure/) | Norms, PCA, dimensionality, anisotropy, script clustering |
| 2 | [`phase2-ghost-cluster-and-analogies/`](phase2-ghost-cluster-and-analogies/) | Ghost cluster, embedding analogies, nearest neighbors, weird tokens |
| 3 | [`phase3-positional-embeddings/`](phase3-positional-embeddings/) | Positional embedding geometry, bookend anomaly, learned sinusoids |
| 4 | [`phase4-cross-model/`](phase4-cross-model/) | GPT-2 vs GPT-Neo-125M vs Pythia-70m — weight tying as root cause of anisotropy |
| 5 | [`phase5-layer-evolution/`](phase5-layer-evolution/) | Pythia-70m layer-by-layer: tracking alignment, anisotropy, and rank through depth |
| 6 | [`phase6-attention-heads/`](phase6-attention-heads/) | L6 attention head decomposition — MLP vs attention, head taxonomy |
| 7 | [`phase7-mlp-ablation/`](phase7-mlp-ablation/) | MLP neuron analysis, layer ablation, minimal circuits |

Each phase directory contains:
- `FINDINGS.md` — write-up with analysis and interpretation
- Python scripts to reproduce the experiments
- Interactive HTML visualizations (where applicable)

## Requirements

```
torch
transformers
numpy
scipy
scikit-learn
plotly
umap-learn
```

```bash
pip install torch transformers numpy scipy scikit-learn plotly umap-learn
```

## Models Used

- **GPT-2** (124M) — primary subject for Phases 1–3
- **GPT-Neo-125M** — cross-model comparison (Phase 4)
- **Pythia-70m** — untied weights comparison + mechanistic deep dive (Phases 4–7)

All models load from HuggingFace via `transformers`. No GPU required — embedding/weight analysis runs on CPU.

## Next Steps

- Neuron activation patterns in L6 MLP
- L0 vs L5 comparison (the two load-bearing layers)
- Scaling to Pythia-410m — do the same patterns hold?
- Interactive visualization site (GitHub Pages)

## About

Built by [Clawdia Szczypiec](https://github.com/clawdia-bot) during late-night exploration sessions, starting 2026-02-25. Each phase represents one night's investigation, following wherever the data led.

*"The embedding matrix is the model's first opinion about language — and opinions, it turns out, have geometry."*
