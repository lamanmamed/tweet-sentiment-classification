import unittest

from tweet_sentiment.features import FinalFeatureExtractor, baseline_features


class FeatureTests(unittest.TestCase):
    def test_baseline_counts_repeated_tokens(self):
        self.assertEqual(baseline_features(["good", "good", "day"]), {"good": 2.0, "day": 1.0})

    def test_final_features_include_unigrams_and_bigrams(self):
        extractor = FinalFeatureExtractor(
            positive_words={"good"},
            negative_words={"bad"},
        )
        features = extractor(["not", "good_NEG", "today"])
        self.assertEqual(features["not"], 1.0)
        self.assertEqual(features["not good_NEG"], 1.0)
        self.assertEqual(features["good_NEG today"], 1.0)

    def test_style_features_are_numeric(self):
        extractor = FinalFeatureExtractor(
            positive_words=set(),
            negative_words=set(),
        )
        features = extractor(["well", "...", "damn"])
        self.assertEqual(features["style_dot_count"], 1.0)
        self.assertEqual(features["style_profanity_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
