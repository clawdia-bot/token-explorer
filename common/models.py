"""
Shared model registry and loader for token-explorer.

Loads embedding matrices from any supported HuggingFace model via
AutoModelForCausalLM, so no model-specific weight keys are needed.
"""

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .tokenutils import token_display

# slug -> (display_name, huggingface_id)
# Kept to the small models used by the active comparison workflow.
MODEL_REGISTRY = {
    'gpt2':          ('GPT-2',          'gpt2'),
    'pythia-70m':    ('Pythia-70m',     'EleutherAI/pythia-70m'),
    'smollm2-135m':  ('SmolLM2-135M',  'HuggingFaceTB/SmolLM2-135M'),
    'qwen2.5-0.5b':  ('Qwen2.5-0.5B',  'Qwen/Qwen2.5-0.5B'),
}


def model_slugs() -> list[str]:
    """Return supported model slugs in registry order."""
    return list(MODEL_REGISTRY.keys())


@dataclass
class ModelData:
    name: str               # Display name: "GPT-2", "Pythia-70m"
    slug: str               # Filesystem-safe: "gpt2", "pythia-70m"
    hf_id: str              # HuggingFace ID
    emb: np.ndarray         # [vocab_size, hidden_dim] input embeddings
    emb_out: np.ndarray | None  # Output embeddings if untied, else None
    pos_emb: np.ndarray | None   # Learned absolute position embeddings, if present
    tokenizer: object       # AutoTokenizer instance
    vocab_size: int
    hidden_dim: int
    tied: bool
    max_positions: int | None
    position_type: str
    has_learned_positions: bool
    tokens: list            # Decoded token strings for all indices
    labels: list            # Display-friendly labels via token_display()
    norms: np.ndarray       # Precomputed L2 norms
    normed_emb: np.ndarray  # Unit-normalized embeddings


def _extract_position_data(model, config) -> tuple[np.ndarray | None, int | None, str, bool]:
    """Extract learned position embeddings when the architecture exposes them."""
    max_positions = getattr(config, 'max_position_embeddings', None)

    if hasattr(config, 'rope_theta') and getattr(config, 'rope_theta') is not None:
        return None, max_positions, 'rope', False

    if config.model_type == 'gpt2' and hasattr(model, 'transformer') and hasattr(model.transformer, 'wpe'):
        pos_emb = model.transformer.wpe.weight.detach().numpy().copy()
        return pos_emb, pos_emb.shape[0], 'learned_absolute', True

    return None, max_positions, 'none', False


def load_model(slug: str) -> ModelData:
    """Load a model's embeddings, tokenizer, and precomputed quantities.

    Uses AutoModelForCausalLM to extract embeddings universally —
    no model-specific weight key handling needed.
    """
    if slug not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{slug}'. Available: {', '.join(MODEL_REGISTRY)}"
        )

    name, hf_id = MODEL_REGISTRY[slug]
    print(f"Loading {name} ({hf_id})...")

    # Load config to check weight tying
    config = AutoConfig.from_pretrained(hf_id)
    tied = getattr(config, 'tie_word_embeddings', True)

    # Load full model, extract embeddings, then free model memory
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32)
    emb = model.get_input_embeddings().weight.detach().numpy().copy()
    pos_emb, max_positions, position_type, has_learned_positions = _extract_position_data(model, config)

    emb_out = None
    if not tied:
        out_layer = model.get_output_embeddings()
        if out_layer is not None:
            emb_out = out_layer.weight.detach().numpy().copy()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vocab_size, hidden_dim = emb.shape
    print(f"  Embedding matrix: {vocab_size} tokens x {hidden_dim} dimensions")
    if not tied and emb_out is not None:
        print(f"  Output embeddings: {emb_out.shape[0]} x {emb_out.shape[1]} (untied)")
    if has_learned_positions and pos_emb is not None:
        print(f"  Position embeddings: {pos_emb.shape[0]} positions x {pos_emb.shape[1]} dimensions")
    elif max_positions is not None:
        print(f"  Positional scheme: {position_type} (max positions: {max_positions})")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    # Decode all tokens
    tokens = [tokenizer.decode([i]) for i in range(vocab_size)]
    labels = [token_display(tokenizer, i) for i in range(vocab_size)]

    # Precompute norms and normalized embeddings
    norms = np.linalg.norm(emb, axis=1)
    normed_emb = emb / (norms[:, None] + 1e-10)

    print(f"  {name} loaded successfully.")
    return ModelData(
        name=name, slug=slug, hf_id=hf_id,
        emb=emb, emb_out=emb_out, pos_emb=pos_emb,
        tokenizer=tokenizer,
        vocab_size=vocab_size, hidden_dim=hidden_dim, tied=tied,
        max_positions=max_positions,
        position_type=position_type,
        has_learned_positions=has_learned_positions,
        tokens=tokens, labels=labels,
        norms=norms, normed_emb=normed_emb,
    )


def _ensure_token_lookups(model: ModelData):
    """Build exact and heuristic token lookups lazily on the model object."""
    if hasattr(model, '_token_lookup_exact') and hasattr(model, '_token_lookup_loose'):
        return

    exact = {}
    loose = {}
    for i, token in enumerate(model.tokens):
        exact.setdefault(token, []).append(i)
        loose[token] = i
        stripped = token.strip()
        if stripped not in loose:
            loose[stripped] = i

    model._token_lookup_exact = exact
    model._token_lookup_loose = loose


def resolve_token_exact(model: ModelData, token: str) -> int | None:
    """Resolve an exact decoded token string.

    Returns None if the token string is missing or ambiguous.
    """
    _ensure_token_lookups(model)
    matches = model._token_lookup_exact.get(token, [])
    if len(matches) != 1:
        return None
    return matches[0]


def resolve_token_loose(model: ModelData, query: str) -> int | None:
    """Find a token index using exploration-friendly heuristics.

    Tries: space-prefixed exact, exact, case-insensitive stripped variants.
    """
    _ensure_token_lookups(model)
    lu = model._token_lookup_loose

    if ' ' + query in lu:
        return lu[' ' + query]
    if query in lu:
        return lu[query]

    q = query.lower()
    for token, idx in lu.items():
        if token.lower() == q or token.strip().lower() == q:
            return idx

    return None


def resolve_token(model: ModelData, query: str) -> int | None:
    """Backward-compatible alias for the loose exploratory resolver."""
    return resolve_token_loose(model, query)


def add_model_arg(parser: argparse.ArgumentParser):
    """Add --model argument to an argparse parser."""
    parser.add_argument(
        '--model', default='gpt2',
        choices=model_slugs(),
        help=f"Model to analyze (default: gpt2)",
    )
