# Future Notes

Read this before reopening later phases or starting new ones.

These notes are the useful leftovers from the old `codex-research-repo-cleanup` branch, condensed into one place.

## Working Rules

1. Keep static embedding claims separate from contextual representation claims.
2. When a later phase revises an earlier claim, preserve the correction explicitly instead of silently overwriting the earlier story.
3. Treat manual forward-pass code as exploratory until it is validated against a reference implementation.
4. Do not use a metric name loosely if the underlying computation changed. Give materially different computations different names.
5. Prefer claims that clearly say whether they are direct observations, interpretations, or broader hypotheses.

## Metric Discipline

For later phases, especially mechanistic ones, keep these distinctions explicit:

- `true_next_token_alignment`
  Mean cosine between hidden state at position `i` and the output embedding of token `i+1`.
- `self_prediction_alignment`
  Mean cosine between a hidden state and the output embedding of the model's own top-1 predicted token.

Do not collapse both into a generic "output alignment" label.

## Validation Priorities

Before leaning too hard on Phases 5+:

1. Run the same prompts through the reference Hugging Face model and the manual implementation.
2. Compare hidden states checkpoint-by-checkpoint.
3. Compare final logits.
4. Treat any substantial mismatch as a blocker on downstream strong claims.

Until that validation is done, later-phase findings should be treated as strong exploratory work rather than fully validated mechanistic truth.

## Phase Structure

For any future phase, try to write it in this order:

1. What exact question is being asked?
2. What model, prompts/data, and representation type are being used?
3. What metrics are being computed?
4. What are the direct observations?
5. What are the interpretations?
6. What alternative explanations remain plausible?
7. What limitations matter?
8. What result would falsify the main takeaway?

## Best Next Directions

If this repo gets reopened for deeper work, the highest-value directions are:

1. Validate manual forward passes against Hugging Face references.
2. Standardize recurring metrics across later phases.
3. Compare more Pythia checkpoints instead of inferring too much from 70m alone.
4. Move from descriptive geometry to causal interventions.

## What Not To Do

- Do not spend large amounts of time polishing Phase 1 and Phase 2 unless a later phase forces a revisit.
- Do not treat exploratory browser tools as claim-bearing evidence.
- Do not generalize from one prompt set or one checkpoint family without saying that is what happened.
