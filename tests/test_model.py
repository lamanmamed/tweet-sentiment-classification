import unittest

from tweet_sentiment.features import FinalFeatureExtractor
from tweet_sentiment.model import TweetSentimentClassifier


class ModelTests(unittest.TestCase):
    def test_baseline_can_fit_small_dataset(self):
        model = TweetSentimentClassifier("baseline")
        model.fit(
            ["love this", "great day", "hate this", "awful day"],
            ["positive", "positive", "negative", "negative"],
        )
        prediction = model.predict_one("great")
        self.assertIn(prediction, {"positive", "negative"})

    def test_final_model_can_use_injected_lexicon(self):
        extractor = FinalFeatureExtractor(
            positive_words={"love", "great"},
            negative_words={"hate", "aw"},
        )
        model = TweetSentimentClassifier("final", feature_extractor=extractor)
        model.fit(
            ["I love this", "great day", "I hate this", "awful day"],
            ["positive", "positive", "negative", "negative"],
        )
        predictions = model.predict(["love it", "hate it"])
        self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
