"""Translate Vosk transcripts strictly into supported light actions with negation safety."""

import re

NEGATION_WORDS = {"not", "dont", "don't", "never", "no", "stop", "cant", "can't"}

ON_PHRASES = (
    "turn on the light",
    "turn on the lights",
    "turn on light",
    "turn on lights",
    "turn the light on",
    "turn the lights on",
    "switch on the light",
    "switch the light on",
    "lights on",
    "light on",
    "turn on",
    "switch on",
)

OFF_PHRASES = (
    "turn off the light",
    "turn off the lights",
    "turn off light",
    "turn off lights",
    "turn the light off",
    "turn the lights off",
    "switch off the light",
    "switch the light off",
    "lights off",
    "light off",
    "turn off",
    "switch off",
)


def _contains_phrase_safely(tokens: list[str], phrase: str) -> bool:
    """Check if exact phrase tokens appear in order without a preceding negation."""
    phrase_tokens = phrase.split()
    n = len(phrase_tokens)

    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == phrase_tokens:
            # Check up to 2 words before the matched phrase for negation (e.g., "do not turn off")
            preceding = tokens[max(0, i - 2) : i]
            if any(word in NEGATION_WORDS for word in preceding):
                return False
            return True
    return False


def check_light_command(text: str) -> str:
    """Return 'ON', 'OFF', or 'UNKNOWN' for a spoken light command."""
    # Strip punctuation except apostrophes and split into word tokens
    cleaned = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    tokens = cleaned.split()
    if not tokens:
        return "UNKNOWN"

    # Evaluate explicit OFF phrases first (safety priority)
    if any(_contains_phrase_safely(tokens, phrase) for phrase in OFF_PHRASES):
        return "OFF"

    # Evaluate explicit ON phrases
    if any(_contains_phrase_safely(tokens, phrase) for phrase in ON_PHRASES):
        return "ON"

    return "UNKNOWN"