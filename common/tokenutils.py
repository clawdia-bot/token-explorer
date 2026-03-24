"""
Shared token display and categorization for any HuggingFace tokenizer.

Generalized from the GPT-2-specific version. Works with byte-level BPE (GPT-2),
SentencePiece (Llama, Gemma), and other tokenizer types.

This module provides:
  - token_display(): readable label for any token (falls back to raw vocab form)
  - categorize(): unified 16-category classification
"""

REPLACEMENT_CHAR = '\ufffd'

# Word-boundary prefixes used by different tokenizer families
_SPACE_PREFIXES = (
    'Ġ',   # GPT-2 byte-level BPE (U+0120)
    '▁',   # SentencePiece (Llama, Gemma, etc.) (U+2581)
)


def token_display(tok, idx):
    """Return a readable string for token `idx`.

    Uses the decoded form when it's clean. Falls back to the raw vocab
    representation when the decoded form contains replacement characters.
    """
    decoded = tok.decode([idx])
    if REPLACEMENT_CHAR not in decoded:
        return decoded
    raw = tok.convert_ids_to_tokens(idx)
    return f"{raw} [raw]"


def categorize(tok, idx):
    """Classify a token into one of 16 categories.

    Works with any tokenizer — detects byte tokens and control chars by
    character properties rather than index ranges.

    Categories: control_char, byte_token, whitespace, number, punctuation,
    japanese, cjk, cyrillic, arabic, hebrew, greek, korean, word,
    allcaps, alpha_fragment, other.
    """
    decoded = tok.decode([idx])
    raw = tok.convert_ids_to_tokens(idx)
    has_space = any(raw.startswith(p) for p in _SPACE_PREFIXES) if raw else False

    # --- Byte-level / control tokens (model-agnostic) ---
    if REPLACEMENT_CHAR in decoded and len(decoded) == 1:
        return 'byte_token'
    if len(decoded) == 1 and (ord(decoded) < 32 or ord(decoded) == 127):
        return 'control_char'

    # --- Tokens with replacement characters (partial multi-byte sequences) ---
    clean = decoded.replace(REPLACEMENT_CHAR, '')
    if not clean:
        return 'byte_token'

    t_stripped = clean.strip()

    if t_stripped == '':
        return 'whitespace'
    if t_stripped.isdigit():
        return 'number'
    if all(c in '.,;:!?-\u2014\u2013()[]{}"\'/\\@#$%^&*~`|<>+=_' for c in t_stripped):
        return 'punctuation'

    # --- Non-Latin script detection ---
    # Only classify by script if the majority of ALL decoded chars (including
    # replacement chars) belong to that script.
    all_chars = [c for c in decoded.strip() if c.strip()]
    script_chars = [c for c in t_stripped if c.strip()]
    if script_chars:
        script_counts = {}
        for c in script_chars:
            if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff':
                script_counts['japanese'] = script_counts.get('japanese', 0) + 1
            elif '\u4e00' <= c <= '\u9fff':
                script_counts['cjk'] = script_counts.get('cjk', 0) + 1
            elif '\u0400' <= c <= '\u04ff':
                script_counts['cyrillic'] = script_counts.get('cyrillic', 0) + 1
            elif '\u0600' <= c <= '\u06ff':
                script_counts['arabic'] = script_counts.get('arabic', 0) + 1
            elif '\u0590' <= c <= '\u05ff':
                script_counts['hebrew'] = script_counts.get('hebrew', 0) + 1
            elif '\u0370' <= c <= '\u03ff':
                script_counts['greek'] = script_counts.get('greek', 0) + 1
            elif '\uac00' <= c <= '\ud7af':
                script_counts['korean'] = script_counts.get('korean', 0) + 1

        if script_counts:
            best_script = max(script_counts, key=script_counts.get)
            if script_counts[best_script] > len(all_chars) / 2:
                return best_script

    # --- Latin-based categories ---
    # Only apply these if the token decoded cleanly (no replacement chars).
    if REPLACEMENT_CHAR not in decoded:
        if has_space and t_stripped.isalpha():
            return 'word'
        if t_stripped.isalpha() and t_stripped == t_stripped.upper() and len(t_stripped) > 1:
            return 'allcaps'
        if t_stripped.isalpha():
            return 'alpha_fragment'

    return 'other'
