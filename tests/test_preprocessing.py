import unittest

from tweet_sentiment.preprocessing import baseline_tokenize, preprocess_tweet


class PreprocessingTests(unittest.TestCase):
    def test_baseline_keeps_whitespace_tokens(self):
        self.assertEqual(baseline_tokenize("not very good"), ["not", "very", "good"])

    def test_tweet_tokenizer_separates_emoticon(self):
        tokens = preprocess_tweet("I'm definitely going sunday!(: who's down?")
        self.assertIn("(:", tokens)
        self.assertIn("!", tokens)

    def test_negation_marker_is_preserved_after_stemming(self):
        tokens = preprocess_tweet("not enjoying this movie.")
        self.assertIn("enjoy_NEG", tokens)
        self.assertIn("thi_NEG", tokens)

    def test_repeated_characters_are_reduced(self):
        tokens = preprocess_tweet("sooooo much")
        self.assertEqual(tokens[0], "sooo")


if __name__ == "__main__":
    unittest.main()
