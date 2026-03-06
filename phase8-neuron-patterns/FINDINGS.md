# Phase 8: Neuron Activation Patterns

## The Questions
1. What do individual MLP neurons in L5 (the prediction layer) respond to? Are there interpretable patterns?
2. Do different word types (articles, verbs, nouns) activate different neuron subsets?
3. How sparse is the representation? How many neurons does each token need?
4. How do L0 neurons compare to L5 neurons?
5. Do neurons cluster into functional groups?

## Key Findings

### 1. Neuron 348 is a generalist, not a specialist

Phase 7 identified N348 as a monster (mean activation 41.0, fires 95% of tokens). Now we know: it fires on *everything*. Its top triggers span prepositions (" of"), abstract nouns (" meaning"), numbers (" one"), verbs (" think"), and technical terms (" function", " gradient"). Entropy analysis confirms it's one of the most generalist neurons (entropy 0.96/1.0).

N348 isn't detecting any semantic category. It's a **universal gain neuron** — a learned constant-ish bias that pushes every prediction toward the output manifold. The other "big" neurons (N297, N407, N1622) are similarly generalist, firing on 75-98% of tokens.

**The dominant neurons are boring.** The interesting ones are rare.

### 2. Word types don't carve neuron space cleanly

Cross-type cosine similarity between mean activation profiles ranges from 0.67 to 0.97. Articles and adjectives are nearly identical (0.97). Even the most "different" pair — concrete nouns vs. abstract nouns — shares cosine similarity of 0.67.

The same ~6 neurons (N348, N297, N407, N1622, N703, N98) dominate across all word types, just with varying magnitudes. Token type information is encoded in the *relative* activation pattern of hundreds of neurons, not in which neurons fire.

This makes sense: at the MLP stage, the model has already attended across the sequence. The "noun-ness" of a token is mixed with its context. The MLP isn't classifying tokens — it's projecting contextualized representations into prediction space.

### 3. Specialist neurons are rare but interpretable

99 neurons (5%) are strong specialists (entropy <0.3). These fire on 1% or fewer tokens and often have readable patterns:

| Neuron | Pattern | Example triggers |
|--------|---------|------------------|
| N920 | Memory/allocation | " allocation", "Memory", " heap" |
| N1847 | Memory concepts | " allocation", "Memory", "Time" |
| N1200 | Temperature/units | " C", " O", " sixty" |
| N1852 | Historical | " published", " degrees", "Chapter" |
| N162 (L0) | Programming | " allocation", " pointer", " array", " heap" |

These specialists fire rarely but carry high information when they do. They're the model's domain detectors — features that activate only in specific contexts.

### 4. ~500 neurons fire per token, but 50 carry 60% of the energy

Average active neurons per token: 502/2048 (24.5%). But energy is concentrated:
- Top 10 neurons: 30% of total activation
- Top 50 neurons: 60%
- Top 100 neurons: 84%

The representation is **dense in count but sparse in energy**. Most of the 500 active neurons contribute tiny activations; a core of ~50 neurons does the heavy lifting.

Tokens differ in how many neurons they need. Function words (" for", " about", " that") activate 800-900 neurons — they're syntactically ambiguous and need more computation. Content words (" returns", " table", " park") activate only 318-340 — they're more self-explanatory.

**Function words are computationally expensive.** Content words are cheap. The model spends more neurons on grammatical scaffolding than on meaning.

### 5. L0 vs L5: democratic vs autocratic

| Property | L0 | L5 |
|----------|:--:|:--:|
| Dead neurons | 1 | 78 |
| Always-on (>90%) | 0 | 18 |
| Top neuron mean act | 0.91 | 41.02 |
| Active neurons/token | 606 | 502 |
| Activation magnitude (mean) | 0.57 | 1.57 |
| Activation magnitude (max) | 7.74 | 157.11 |

L0 is **democratic**: activations are flat, no neuron dominates, nearly all neurons are alive. It distributes computation evenly — every neuron contributes a little.

L5 is **autocratic**: a handful of neurons (N348, N297, N407) dominate, 78 neurons are dead, and the activation range spans 3 orders of magnitude. It has learned to concentrate prediction power in a few heavy hitters.

L0 top neurons show weak but interpretable patterns: N1792 responds to numbers (" three", " two", " one"), N162 to programming concepts. But the patterns are much weaker than L5's specialists — L0 is still building features, not detecting categories.

### 6. Neurons form correlated clusters

2,343 neuron pairs have correlation >0.8. 93 pairs exceed 0.9. These aren't random — they suggest functional groups of neurons that co-activate.

The tightest cluster: N703, N904, N1516, N1622, N73, N1901 (all pairwise correlations >0.93). These all fire on prepositions and function words. They're redundant representations of "this is a structural/connective token."

The most anti-correlated neuron is N1838, which fires on sentence-initial tokens ("I", "Sorry", first-person constructions) and is *anti-correlated* with preposition/function-word neurons. When the model sees "I think..." it suppresses the preposition detectors and activates the sentence-start detectors. **Neuron anti-correlation encodes syntactic position.**

### 7. The MLP is not a lookup table

The overall picture: L5's MLP doesn't map tokens to output predictions through dedicated neurons. Instead:
- A core of ~6 generalist neurons provide a constant bias toward the output manifold
- ~50 neurons carry most of the token-specific energy
- ~500 neurons contribute small refinements
- ~100 specialist neurons fire rarely but carry domain-specific information
- ~80 neurons are permanently dead

The prediction emerges from the superposition of all active neurons, modulated by GELU gating. No single neuron "means" anything in isolation. The meaning is in the pattern.

## Implications

1. **Pruning potential:** 78 dead neurons + the observation that top 100 carry 84% of energy suggests L5's MLP could be compressed significantly. The democratic L0 is harder to prune.

2. **Function words as computational load:** The model dedicates more neurons to "the", "of", "about" than to "detective" or "park." Language modeling is mostly about getting the grammar right, not the content.

3. **Specialist neurons as interpretability handles:** The 99 strong specialists are the most interpretable features in the network. Mechanistic interpretability could focus on these rather than trying to decode the generalist soup.

4. **Neuron clustering suggests redundancy:** Groups of 5-6 neurons with >0.93 correlation are doing nearly identical work. This redundancy might serve as noise robustness, or it might be training inefficiency.

## Next Steps
- Scale to Pythia-410m: does the autocratic pattern (monster neurons + dead neurons) intensify or democratize with model size?
- Track how L0's democratic neurons evolve through layers — at what layer does the autocracy emerge?
- Investigate N1838 (the sentence-start neuron) more deeply — is it doing BOS-like detection?
- Causal intervention: zero out specialist neurons and measure targeted perplexity impact on their domain
