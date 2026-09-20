"""Feature extraction for the baseline and final sentiment models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from nltk import ngrams


# This list was one of the small hand-written style features tested in the final model.
PROFANITY_TERMS = {
    "fuck", "shit", "damn", "bitch", "crap", "piss", "dick", "darn",
    "cock", "pussy", "asshole", "fag", "bastard", "slut", "douche", "ass",
}


def load_opinion_lexicon() -> tuple[set[str], set[str]]:
    """Load NLTK's positive and negative opinion-word lists."""
    from nltk.corpus import opinion_lexicon

    try:
        positive = set(opinion_lexicon.positive())
        negative = set(opinion_lexicon.negative())
    except LookupError as exc:
        raise LookupError(
            "NLTK's opinion_lexicon is required. Install it with "
            "`python -m nltk.downloader opinion_lexicon`."
        ) from exc

    return positive, negative


def baseline_features(tokens: Iterable[str]) -> dict[str, float]:
    """Count unigram occurrences for the Bag-of-Words baseline."""
    return {token: float(count) for token, count in Counter(tokens).items()}


class FinalFeatureExtractor:
    """Create the unigram, bigram, lexicon, and style features used at the end."""

    def __init__(
        self,
        positive_words: set[str] | None = None,
        negative_words: set[str] | None = None,
    ) -> None:
        if positive_words is None or negative_words is None:
            positive_words, negative_words = load_opinion_lexicon()
        self.positive_words = positive_words
        self.negative_words = negative_words

    def __call__(self, tokens: list[str]) -> dict[str, float]:
        features: dict[str, float] = {}

        # The final notebook used binary unigram and bigram presence rather than counts.
        for token in tokens:
            features[token] = 1.0

        for left, right in ngrams(tokens, 2):
            features[f"{left} {right}"] = 1.0

        features["lexicon_pos_count"] = float(
            sum(token in self.positive_words for token in tokens)
        )
        features["lexicon_neg_count"] = float(
            sum(token in self.negative_words for token in tokens)
        )
        features["style_dot_count"] = float(
            sum("..." in token for token in tokens)
        )
        features["style_profanity_count"] = float(
            sum(
                any(term in token.lower() for term in PROFANITY_TERMS)
                for token in tokens
            )
        )

        return features
