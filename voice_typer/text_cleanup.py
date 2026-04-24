"""Lightweight cleanup for raw speech-to-text output."""

import re


_QUESTION_OPENERS = {
    "am", "are", "can", "could", "did", "do", "does", "has", "have",
    "how", "is", "may", "should", "was", "were", "what", "when",
    "where", "which", "who", "whom", "whose", "why", "will", "would",
}


def clean_transcribed_text(text: str) -> str:
    """Apply conservative cleanup without changing the user's meaning."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = _normalize_spacing(cleaned)
    cleaned = _remove_adjacent_duplicate_phrases(cleaned)
    cleaned = _capitalize_sentences(cleaned)
    cleaned = _capitalize_pronoun_i(cleaned)
    cleaned = _add_terminal_punctuation(cleaned)
    return cleaned


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=[^\s,.;:!?])", r"\1 ", text)
    return text.strip()


def _remove_adjacent_duplicate_phrases(text: str) -> str:
    tokens = text.split(" ")
    output = []
    i = 0
    while i < len(tokens):
        duplicate_len = _duplicate_phrase_length(tokens, i)
        if duplicate_len:
            output.extend(tokens[i:i + duplicate_len])
            i += duplicate_len * 2
        else:
            output.append(tokens[i])
            i += 1
    return " ".join(output)


def _duplicate_phrase_length(tokens: list[str], index: int) -> int:
    max_len = min(4, (len(tokens) - index) // 2)
    for size in range(max_len, 0, -1):
        left = [_token_key(token) for token in tokens[index:index + size]]
        right = [
            _token_key(token)
            for token in tokens[index + size:index + (size * 2)]
        ]
        if left == right and any(left):
            return size
    return 0


def _token_key(token: str) -> str:
    return re.sub(r"^\W+|\W+$", "", token).lower()


def _capitalize_sentences(text: str) -> str:
    chars = list(text)
    capitalize_next = True
    for index, char in enumerate(chars):
        if char.isalpha():
            if capitalize_next:
                chars[index] = char.upper()
            capitalize_next = False
        elif char in ".!?":
            capitalize_next = True
    return "".join(chars)


def _capitalize_pronoun_i(text: str) -> str:
    return re.sub(r"\bi\b", "I", text)


def _add_terminal_punctuation(text: str) -> str:
    if not text or text[-1] in ".!?":
        return text
    if _looks_like_question(text):
        return f"{text}?"
    return f"{text}."


def _looks_like_question(text: str) -> bool:
    sentence = re.split(r"[.!?]\s+", text.strip())[-1]
    words = re.findall(r"[A-Za-z']+", sentence.lower())
    if not words:
        return False
    if words[0] in _QUESTION_OPENERS:
        return True
    question_starters = {
        ("do", "you"),
        ("did", "you"),
        ("can", "you"),
        ("could", "you"),
        ("would", "you"),
        ("should", "we"),
    }
    return len(words) >= 2 and tuple(words[:2]) in question_starters
