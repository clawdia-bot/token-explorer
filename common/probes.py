"""
Curated probe definitions for exact, reproducible cross-model comparisons.
"""

from .models import MODEL_REGISTRY, resolve_token_exact


SUPPORTED_MODELS = tuple(MODEL_REGISTRY.keys())


def _same(token: str) -> dict[str, str]:
    return {slug: token for slug in SUPPORTED_MODELS}


CONCEPTS = {
    'the': {'description': 'determiner', 'tokens': _same(' the')},
    'at': {'description': 'preposition', 'tokens': _same(' at')},
    'king': {'description': 'royalty', 'tokens': _same(' king')},
    'queen': {'description': 'royalty', 'tokens': _same(' queen')},
    'dog': {'description': 'animal', 'tokens': _same(' dog')},
    'python': {'description': 'programming language', 'tokens': _same(' Python')},
    'good': {'description': 'positive adjective', 'tokens': _same(' good')},
    'france': {'description': 'country', 'tokens': _same(' France')},
    'water': {'description': 'substance', 'tokens': _same(' water')},
    'big': {'description': 'adjective', 'tokens': _same(' big')},
    'run': {'description': 'verb', 'tokens': _same(' run')},
    'she': {'description': 'pronoun', 'tokens': _same(' she')},
    'one': {'description': 'number', 'tokens': _same(' one')},
    'day': {'description': 'time noun', 'tokens': _same(' day')},
    'time': {'description': 'time noun', 'tokens': _same(' time')},
    'new': {'description': 'adjective', 'tokens': _same(' new')},
    'old': {'description': 'adjective', 'tokens': _same(' old')},
    'man': {'description': 'person', 'tokens': _same(' man')},
    'world': {'description': 'world noun', 'tokens': _same(' world')},
    'just': {'description': 'adverb', 'tokens': _same(' just')},
    'make': {'description': 'verb', 'tokens': _same(' make')},
    'think': {'description': 'verb', 'tokens': _same(' think')},
    'back': {'description': 'direction', 'tokens': _same(' back')},
    'woman': {'description': 'person', 'tokens': _same(' woman')},
    'dogs': {'description': 'plural animal', 'tokens': _same(' dogs')},
    'cat': {'description': 'animal', 'tokens': _same(' cat')},
    'cats': {'description': 'plural animal', 'tokens': _same(' cats')},
    'paris': {'description': 'city', 'tokens': _same(' Paris')},
    'japan': {'description': 'country', 'tokens': _same(' Japan')},
    'tokyo': {'description': 'city', 'tokens': _same(' Tokyo')},
    'bigger': {'description': 'comparative adjective', 'tokens': _same(' bigger')},
    'small': {'description': 'adjective', 'tokens': _same(' small')},
    'smaller': {'description': 'comparative adjective', 'tokens': _same(' smaller')},
    'best': {'description': 'superlative adjective', 'tokens': _same(' best')},
    'bad': {'description': 'negative adjective', 'tokens': _same(' bad')},
    'worst': {'description': 'superlative adjective', 'tokens': _same(' worst')},
    'walk': {'description': 'verb', 'tokens': _same(' walk')},
    'walked': {'description': 'past tense verb', 'tokens': _same(' walked')},
    'ran': {'description': 'past tense verb', 'tokens': _same(' ran')},
    'spain': {'description': 'country', 'tokens': _same(' Spain')},
    'spanish': {'description': 'demonym', 'tokens': _same(' Spanish')},
    'germany': {'description': 'country', 'tokens': _same(' Germany')},
    'german': {'description': 'demonym', 'tokens': _same(' German')},
    'hot': {'description': 'adjective', 'tokens': _same(' hot')},
    'cold': {'description': 'adjective', 'tokens': _same(' cold')},
    'up': {'description': 'direction', 'tokens': _same(' up')},
    'down': {'description': 'direction', 'tokens': _same(' down')},
    'boy': {'description': 'person', 'tokens': _same(' boy')},
    'girl': {'description': 'person', 'tokens': _same(' girl')},
    'eat': {'description': 'verb', 'tokens': _same(' eat')},
    'ate': {'description': 'past tense verb', 'tokens': _same(' ate')},
    'drink': {'description': 'verb', 'tokens': _same(' drink')},
    'drank': {'description': 'past tense verb', 'tokens': _same(' drank')},
    'italy': {'description': 'country', 'tokens': _same(' Italy')},
    'rome': {'description': 'city', 'tokens': _same(' Rome')},
    'berlin': {'description': 'city', 'tokens': _same(' Berlin')},
    'fast': {'description': 'adjective', 'tokens': _same(' fast')},
    'faster': {'description': 'comparative adjective', 'tokens': _same(' faster')},
    'slow': {'description': 'adjective', 'tokens': _same(' slow')},
    'slower': {'description': 'comparative adjective', 'tokens': _same(' slower')},
}


PHASE2_NEIGHBOR_PROBES = [
    ('the', 'determiners/pronouns'),
    ('at', 'prepositions'),
    ('king', 'royalty'),
    ('queen', 'royalty (gendered)'),
    ('dog', 'animals'),
    ('python', 'programming languages'),
]


PHASE2_ANALOGIES = [
    ('gender', 'king', 'queen', 'man', 'woman'),
    ('pluralization', 'dog', 'dogs', 'cat', 'cats'),
    ('capital_city', 'france', 'paris', 'japan', 'tokyo'),
    ('comparative', 'big', 'bigger', 'small', 'smaller'),
]


COMPARISON_ANALOGIES = [
    ('gender', 'king', 'queen', 'man', 'woman'),
    ('plural', 'dog', 'dogs', 'cat', 'cats'),
    ('capital', 'france', 'paris', 'japan', 'tokyo'),
    ('comparative', 'big', 'bigger', 'small', 'smaller'),
    ('superlative', 'good', 'best', 'bad', 'worst'),
    ('past_tense', 'walk', 'walked', 'run', 'ran'),
    ('demonym', 'spain', 'spanish', 'germany', 'german'),
    ('antonym', 'hot', 'cold', 'up', 'down'),
    ('gender2', 'man', 'woman', 'boy', 'girl'),
    ('past_tense2', 'eat', 'ate', 'drink', 'drank'),
    ('capital2', 'italy', 'rome', 'germany', 'berlin'),
    ('comparative2', 'fast', 'faster', 'slow', 'slower'),
]


COMPARISON_NEIGHBOR_PROBES = [
    'the', 'king', 'dog', 'good', 'france', 'water', 'big', 'run',
    'she', 'one', 'day', 'time', 'new', 'old', 'man', 'world',
    'just', 'make', 'think', 'back',
]


ALL_REQUIRED_CONCEPTS = tuple(dict.fromkeys(
    [concept_id for concept_id, _ in PHASE2_NEIGHBOR_PROBES]
    + [item for _, a, b, c, d in PHASE2_ANALOGIES for item in (a, b, c, d)]
    + [item for _, a, b, c, d in COMPARISON_ANALOGIES for item in (a, b, c, d)]
    + COMPARISON_NEIGHBOR_PROBES
))


def token_for_concept(slug: str, concept_id: str) -> str:
    return CONCEPTS[concept_id]['tokens'][slug]


def resolve_concept(model, concept_id: str) -> int:
    token = token_for_concept(model.slug, concept_id)
    idx = resolve_token_exact(model, token)
    if idx is None:
        raise ValueError(
            f"Exact token {token!r} for concept '{concept_id}' is missing or ambiguous in {model.slug}"
        )
    return idx


def validate_model_probes(model, concept_ids=None):
    """Resolve and validate exact probe tokens for a single model."""
    concept_ids = ALL_REQUIRED_CONCEPTS if concept_ids is None else concept_ids
    return {concept_id: resolve_concept(model, concept_id) for concept_id in concept_ids}


def validate_probe_pack(models: dict):
    """Resolve and validate exact probe tokens for multiple loaded models."""
    resolved = {}
    for slug, model in models.items():
        resolved[slug] = validate_model_probes(model)
    return resolved
