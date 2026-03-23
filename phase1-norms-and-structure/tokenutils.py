"""
Shared token display and categorization for GPT-2 byte-level BPE tokens.

GPT-2 uses byte-level BPE, so tokens 0-255 represent raw bytes. Many of these
produce Unicode replacement characters (U+FFFD) when decoded, because they're
incomplete UTF-8 sequences. The same happens for ~224 higher-index tokens that
capture partial multi-byte sequences.

This module provides:
  - token_display(): readable label for any token (falls back to raw vocab form)
  - categorize(): unified 16-category classification
"""

REPLACEMENT_CHAR = '\ufffd'


def token_display(tok, idx):
    """Return a readable string for token `idx`.

    Uses the decoded form when it's clean. Falls back to the raw vocab
    representation (the internal byte-to-char mapping GPT-2 uses) when
    the decoded form contains replacement characters.
    """
    decoded = tok.decode([idx])
    if REPLACEMENT_CHAR not in decoded:
        return decoded
    raw = tok.convert_ids_to_tokens(idx)
    return f"{raw} [raw]"


def categorize(tok, idx):
    """Classify a GPT-2 token into one of 16 categories.

    Categories: control_char, byte_token, whitespace, number, punctuation,
    japanese, cjk, cyrillic, arabic, hebrew, greek, korean, word,
    allcaps, alpha_fragment, other.
    """
    decoded = tok.decode([idx])
    raw = tok.convert_ids_to_tokens(idx)
    has_space = raw.startswith('Ġ') if raw else False

    # --- Byte-level tokens (idx < 256) ---
    if idx < 256:
        if REPLACEMENT_CHAR in decoded:
            return 'byte_token'
        if len(decoded) == 1 and (ord(decoded) < 32 or ord(decoded) == 127):
            return 'control_char'

    # --- Tokens with replacement characters (partial multi-byte sequences) ---
    # Filter to non-replacement chars for classification
    clean = decoded.replace(REPLACEMENT_CHAR, '')
    if not clean:
        # Nothing left after removing replacement chars — entirely garbled
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
    # replacement chars) belong to that script. This prevents garbled tokens
    # like '��士' from being labeled 'cjk' based on one surviving character.
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
            # Require majority of ALL chars (including replacements) to be from that script
            if script_counts[best_script] > len(all_chars) / 2:
                return best_script

    # --- Latin-based categories ---
    # Only apply these if the token decoded cleanly (no replacement chars).
    # Garbled tokens with surviving non-Latin chars (e.g. '��士') shouldn't
    # become 'alpha_fragment' just because Python's isalpha() is Unicode-aware.
    if REPLACEMENT_CHAR not in decoded:
        if has_space and t_stripped.isalpha():
            return 'word'
        if t_stripped.isalpha() and t_stripped == t_stripped.upper() and len(t_stripped) > 1:
            return 'allcaps'
        if t_stripped.isalpha():
            return 'alpha_fragment'

    return 'other'
