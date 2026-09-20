"""Evaluation helpers for the sentiment experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data import Sample
from .model import TweetSentimentClassifier


@dataclass(frozen=True)
class Scores:
    precision: float
    recall: float
    f1: float
    accuracy: float


def score_predictions(labels: list[str], predictions: list[str]) -> Scores:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return Scores(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(accuracy_score(labels, predictions)),
    )


def contiguous_cross_validate(
    samples: list[Sample],
    classifier_factory: Callable[[], TweetSentimentClassifier],
    folds: int = 10,
) -> Scores:
    """Run contiguous folds to match the split strategy in the original notebooks."""
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    if len(samples) < folds:
        raise ValueError("Need at least one sample per fold.")

    fold_size = int(len(samples) / folds) + 1
    fold_scores: list[Scores] = []

    for start in range(0, len(samples), fold_size):
        validation = samples[start : start + fold_size]
        training = samples[:start] + samples[start + fold_size :]
        if not validation or not training:
            continue

        classifier = classifier_factory()
        classifier.fit(
            [sample.text for sample in training],
            [sample.label for sample in training],
        )
        predictions = classifier.predict([sample.text for sample in validation])
        fold_scores.append(
            score_predictions([sample.label for sample in validation], predictions)
        )

    values = np.array(
        [[score.precision, score.recall, score.f1, score.accuracy] for score in fold_scores]
    )
    means = values.mean(axis=0)
    return Scores(*map(float, means))


def error_analysis(
    training: list[Sample],
    validation: list[Sample],
    classifier: TweetSentimentClassifier,
) -> tuple[np.ndarray, list[tuple[Sample, str]]]:
    """Return a confusion matrix and the misclassified examples."""
    classifier.fit(
        [sample.text for sample in training],
        [sample.label for sample in training],
    )
    predictions = classifier.predict([sample.text for sample in validation])
    labels = [sample.label for sample in validation]
    matrix = confusion_matrix(labels, predictions, labels=["negative", "positive"])
    errors = [
        (sample, prediction)
        for sample, prediction in zip(validation, predictions)
        if sample.label != prediction
    ]
    return matrix, errors
