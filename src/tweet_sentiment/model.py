"""Linear SVM classifiers for the baseline and final pipelines."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import FinalFeatureExtractor, baseline_features
from .preprocessing import baseline_tokenize, preprocess_tweet


FeatureFunction = Callable[[list[str]], dict[str, float]]
TokenizeFunction = Callable[[str], list[str]]


class TweetSentimentClassifier:
    """Wrap preprocessing, feature extraction, and a Linear SVM in one interface."""

    def __init__(
        self,
        variant: str = "final",
        *,
        feature_extractor: FeatureFunction | None = None,
    ) -> None:
        if variant not in {"baseline", "final"}:
            raise ValueError("variant must be 'baseline' or 'final'.")

        self.variant = variant
        if variant == "baseline":
            self.tokenize: TokenizeFunction = baseline_tokenize
            self.featurize: FeatureFunction = feature_extractor or baseline_features
            self.pipeline = Pipeline(
                [
                    ("vectorizer", DictVectorizer()),
                    ("classifier", LinearSVC()),
                ]
            )
        else:
            self.tokenize = preprocess_tweet
            self.featurize = feature_extractor or FinalFeatureExtractor()
            self.pipeline = Pipeline(
                [
                    ("vectorizer", DictVectorizer()),
                    ("tfidf", TfidfTransformer()),
                    ("classifier", LinearSVC(class_weight="balanced")),
                ]
            )

    def _features(self, texts: Iterable[str]) -> list[dict[str, float]]:
        return [self.featurize(self.tokenize(text)) for text in texts]

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "TweetSentimentClassifier":
        self.pipeline.fit(self._features(texts), list(labels))
        return self

    def predict(self, texts: Iterable[str]) -> list[str]:
        return list(self.pipeline.predict(self._features(texts)))

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]
