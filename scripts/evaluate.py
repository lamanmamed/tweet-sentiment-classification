"""Run the baseline and final sentiment models on the expected TSV dataset."""

from __future__ import annotations

import argparse

from tweet_sentiment.data import load_tsv, train_test_split
from tweet_sentiment.evaluation import contiguous_cross_validate, score_predictions
from tweet_sentiment.model import TweetSentimentClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to sentiment-dataset.tsv")
    parser.add_argument(
        "--variant",
        choices=["baseline", "final"],
        default="final",
    )
    args = parser.parse_args()

    samples = load_tsv(args.dataset)
    training, test = train_test_split(samples)

    print(f"Loaded {len(samples)} tweets")
    print(f"Training split: {len(training)}")
    print(f"Held-out test split: {len(test)}")

    cv = contiguous_cross_validate(
        training,
        lambda: TweetSentimentClassifier(args.variant),
        folds=10,
    )
    print("\n10-fold cross-validation")
    print(f"precision: {cv.precision:.4f}")
    print(f"recall:    {cv.recall:.4f}")
    print(f"f1:        {cv.f1:.4f}")
    print(f"accuracy:  {cv.accuracy:.4f}")

    classifier = TweetSentimentClassifier(args.variant)
    classifier.fit(
        [sample.text for sample in training],
        [sample.label for sample in training],
    )
    predictions = classifier.predict([sample.text for sample in test])
    test_scores = score_predictions([sample.label for sample in test], predictions)

    print("\nHeld-out test set")
    print(f"precision: {test_scores.precision:.4f}")
    print(f"recall:    {test_scores.recall:.4f}")
    print(f"f1:        {test_scores.f1:.4f}")
    print(f"accuracy:  {test_scores.accuracy:.4f}")


if __name__ == "__main__":
    main()
