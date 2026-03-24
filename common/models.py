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
MODEL_REGISTRY = {
    'gpt2':          ('GPT-2',          'gpt2'),
    'pythia-70m':    ('Pythia-70m',     'EleutherAI/pythia-70m'),
    'smollm2-135m':  ('SmolLM2-135M',  'HuggingFaceTB/SmolLM2-135M'),
    'qwen2.5-0.5b':  ('Qwen2.5-0.5B',  'Qwen/Qwen2.5-0.5B'),
    'gemma-2-2b':    ('Gemma 2-2B',    'google/gemma-2-2b'),
    'llama-3.2-1b':  ('Llama 3.2-1B',  'meta-llama/Llama-3.2-1B'),
    'phi-3.5-mini':  ('Phi-3.5-mini',  'microsoft/Phi-3.5-mini-instruct'),
}


@dataclass
class ModelData:
    name: str               # Display name: "GPT-2", "Pythia-70m"
    slug: str               # Filesystem-safe: "gpt2", "pythia-70m"
    hf_id: str              # HuggingFace ID
    emb: np.ndarray         # [vocab_size, hidden_dim] input embeddings
    emb_out: np.ndarray | None  # Output embeddings if untied, else None
    tokenizer: object       # AutoTokenizer instance
    vocab_size: int
    hidden_dim: int
    tied: bool
    tokens: list            # Decoded token strings for all indices
    labels: list            # Display-friendly labels via token_display()
    norms: np.ndarray       # Precomputed L2 norms
    normed_emb: np.ndarray  # Unit-normalized embeddings


def list_models():
    """Print available models."""
    print("Available models:")
    for slug, (name, hf_id) in MODEL_REGISTRY.items():
        print(f"  {slug:20s}  {name:20s}  ({hf_id})")


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
        emb=emb, emb_out=emb_out,
        tokenizer=tokenizer,
        vocab_size=vocab_size, hidden_dim=hidden_dim, tied=tied,
        tokens=tokens, labels=labels,
        norms=norms, normed_emb=normed_emb,
    )


def resolve_token(model: ModelData, query: str) -> int | None:
    """Find the best token index for a query string.

    Tries: space-prefixed exact, exact, case-insensitive variants.
    Returns None if no match found.
    """
    # Build lookup on first call (cached on the model object)
    if not hasattr(model, '_token_lookup'):
        lookup = {}
        for i, t in enumerate(model.tokens):
            lookup[t] = i
            stripped = t.strip()
            if stripped not in lookup:
                lookup[stripped] = i
        model._token_lookup = lookup

    lu = model._token_lookup

    # Try space-prefixed (most tokenizers use leading space for word tokens)
    if ' ' + query in lu:
        return lu[' ' + query]
    if query in lu:
        return lu[query]

    # Case-insensitive fallback
    q = query.lower()
    for t, i in lu.items():
        if t.lower() == q or t.strip().lower() == q:
            return i

    return None


def add_model_arg(parser: argparse.ArgumentParser):
    """Add --model argument to an argparse parser."""
    parser.add_argument(
        '--model', default='gpt2',
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Model to analyze (default: gpt2)",
    )
