"""Tweet-aware tokenization, negation marking, and stemming."""

from __future__ import annotations

from nltk.sentiment.util import mark_negation
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


_TOKENIZER = TweetTokenizer(
    preserve_case=False,
    strip_handles=True,
    reduce_len=True,
)
_STEMMER = PorterStemmer()


def baseline_tokenize(text: str) -> list[str]:
    """Tokenize with whitespace only, as used by the baseline model."""
    return text.split()


def _stem_negated_token(token: str) -> str:
    if token.endswith("_NEG"):
        word = token[:-4]
        return f"{_STEMMER.stem(word)}_NEG"
    return _STEMMER.stem(token)


def preprocess_tweet(text: str) -> list[str]:
    """Prepare a tweet for the final classifier.

    Handles Twitter punctuation and emoticons, marks words that fall under
    negation, and stems words while preserving the negation marker.
    """
    tokens = _TOKENIZER.tokenize(text)
    negation_marked = mark_negation(tokens)
    return [_stem_negated_token(token) for token in negation_marked]
