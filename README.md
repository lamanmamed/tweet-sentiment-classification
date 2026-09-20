# Tweet Sentiment Classification

This project classifies tweets as **positive** or **negative** with a Linear Support Vector Machine. The first version uses a simple Bag-of-Words representation. The later version changes how tweets are tokenized and represented after looking at the mistakes made by that baseline.

The main point of the project is not just the final score. It is the progression from a model that treats a tweet as a bag of separate words to one that keeps more of the information that matters in short, informal text. That includes punctuation, emoticons, repeated characters, adjacent word pairs, and the effect of negation.

## Results

The original experiment used **33,540 tweets**. The first 80% were used as the training portion and the remaining 20% were kept as a held-out test set. This gave 26,832 training tweets and 6,708 test tweets.

The baseline and final models were evaluated with weighted precision, recall, F1, and accuracy. Weighted scores account for the fact that the positive and negative classes were not equally common.

| Model | 10-fold CV weighted F1 |
| --- | ---: |
| Bag-of-Words baseline | 0.827 |
| Final pipeline | 0.881 |

The final model was then trained on the complete training portion and evaluated on the held-out test set:

| Metric | Held-out test score |
| --- | ---: |
| Precision | 0.879 |
| Recall | 0.880 |
| F1 | 0.879 |

These numbers come from the saved outputs in the original experiment. The dataset itself was not included in the submitted archive, so I have not presented them as newly reproduced results from this cleaned repository.

## Starting with a simple baseline

The baseline is deliberately basic. A tweet is split on whitespace, each token becomes a feature, and the value of that feature is the number of times the token appears in the tweet.

For example:

```text
I really love this
```

becomes roughly:

```text
I       -> 1
really  -> 1
love    -> 1
this    -> 1
```

A Linear SVM learns which of these features are associated with positive and negative labels.

This baseline reached a weighted F1 of about **0.827** in 10-fold cross-validation. That is already useful, but looking only at the overall score hides where the model is weaker.

## What the baseline was getting wrong

The error analysis showed that the baseline was noticeably better at recognising positive tweets than negative ones. In one validation fold, it correctly found 1,599 of 1,866 positive tweets, but only 608 of 818 negative tweets.

Looking at individual errors made the weaknesses easier to understand.

### Tokenization

The baseline simply splits text at spaces. That is a poor fit for tweets because punctuation and emoticons often carry meaning.

For example:

```text
I'm definitely going sunday!(: who's down?
```

A whitespace tokenizer keeps `sunday!(:` as one token. A tweet-aware tokenizer can separate the punctuation and emoticon so that they are available as their own features.

### Word order

A unigram model sees words separately. It cannot distinguish very well between phrases such as:

```text
good
not good
```

Both contain the word `good`. The second phrase has the opposite meaning, but the baseline has no feature that directly represents the pair `not good`.

### Negation

Negation can affect more than the word immediately after `not`.

The improved preprocessing marks words that fall inside a negated span. A phrase such as:

```text
not good at all
```

can produce features similar to:

```text
not
good_NEG
at_NEG
all_NEG
```

This gives the classifier a way to learn that `good` and `good_NEG` do not behave the same way.

### Informal language

Tweets contain slang, repeated characters, usernames, unusual punctuation, hashtags, and emoticons. A general whitespace tokenizer does not handle these consistently. The final preprocessing uses NLTK's `TweetTokenizer`, which was designed for this kind of text.

## The final text representation

The final pipeline still uses a Linear SVM. Most of the improvement comes from changing the representation of the tweet rather than replacing the classifier with a more complicated model.

### Tweet-aware tokenization

`TweetTokenizer` lowercases the text, removes Twitter handles, separates punctuation and emoticons, and shortens very long repeated-character sequences.

For example:

```text
IT IS RAINING sooooo much
```

is tokenized and stemmed to something close to:

```text
it
is
rain
sooo
much
```

### Stemming

Porter stemming reduces related word forms to a shared stem. This helps the model treat forms such as `running`, `runs`, and `run` as more closely related rather than completely separate features.

### Unigrams and bigrams

The final feature vector contains both individual tokens and adjacent token pairs.

For:

```text
not happy today
```

features include the individual tokens and pairs such as:

```text
not happy_NEG
happy_NEG today_NEG
```

Bigrams give the classifier some local word-order information without moving to a sequence model.

### TF-IDF weighting

The baseline works with raw token counts. The final model applies TF-IDF after the feature dictionaries have been converted into a feature matrix.

TF-IDF reduces the influence of features that appear in a large number of tweets and gives relatively more weight to features that are more distinctive. In the original experiments, introducing TF-IDF raised weighted F1 from about **0.827 to 0.843**.

### Class balancing

The training data contains more positive tweets than negative ones. The baseline therefore had more opportunity to learn the positive class and showed lower recall for negative tweets.

The final `LinearSVC` uses `class_weight="balanced"`. This gives more weight to examples from the less common class during training. In the experiment sequence, class balancing increased weighted F1 from roughly **0.873 to 0.875**.

### Small lexicon and style features

The final representation also includes counts derived from NLTK's opinion lexicon, which contains lists of positive and negative words.

Two simple style counts are included as well. One counts ellipsis-like tokens and the other counts a small hand-written set of profanity terms. These were added after looking through misclassified tweets rather than as a general claim that these patterns always indicate sentiment.

## What changed during the experiments

The improvements were added one at a time and evaluated rather than kept automatically.

| Change | Weighted F1 reported during development |
| --- | ---: |
| Bag-of-Words baseline | 0.827 |
| Add TF-IDF | about 0.843 |
| Use tweet-aware tokenization | about 0.862 |
| Add bigrams and stemming | about 0.873 |
| Balance class weights | about 0.875 |
| Add negation marking | about 0.881 |

Not every experiment improved the classifier. Those results were useful because they showed which changes were actually helping on this dataset.

| Experiment that was removed | Reported result |
| --- | ---: |
| Remove English stop words | about 0.873 F1 |
| Keep only features appearing at least 5 times | about 0.875 F1 |
| Replace Linear SVM with Logistic Regression | about 0.864 F1 |

Stop-word removal was a particularly useful result. Words that are often called uninformative in longer documents can still help in short tweets. Removing them did not improve this classifier.

## Error analysis after the changes

The same single validation split used for qualitative error analysis produced **467 errors** with the baseline and **290 errors** with the improved pipeline.

That does not mean the remaining mistakes all have a simple preprocessing fix. Sarcasm is a good example. A tweet can contain words that are normally positive while the intended meaning is negative. Bigrams and negation help with local context, but they do not give a linear classifier a full understanding of sarcasm or the situation being discussed.

The remaining errors are a reminder that better tokenization and feature engineering improve what the classifier can see, but they do not turn a Bag-of-Words-style model into a general language understanding system.

## How the code is organised

The notebooks have been split into normal Python modules so that the baseline, preprocessing, features, model, and evaluation code can be read separately.

```text
tweet-sentiment-classification/
├── src/tweet_sentiment/
│   ├── data.py              # load the TSV file and create the 80/20 split
│   ├── preprocessing.py     # whitespace baseline and tweet-aware preprocessing
│   ├── features.py          # unigram, bigram, lexicon, and style features
│   ├── model.py             # baseline and final Linear SVM pipelines
│   └── evaluation.py        # cross-validation, metrics, and error analysis
├── scripts/
│   └── evaluate.py          # run the experiment on a local dataset
├── examples/
│   └── preprocess_tweet.py  # inspect the final tokenization on one tweet
├── data/
│   └── README.md            # expected dataset format
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_model.py
├── pyproject.toml
└── README.md
```

The original notebooks and assignment instructions are not part of this cleaned version. They mixed starter code, written answers, experiments, and final code in the same files. The repository keeps the implemented pipeline and the parts needed to understand how it was evaluated.

## Running the code

Python 3.10 or newer is recommended.

Install the package from the repository root:

```bash
pip install -e .
```

The final feature extractor uses NLTK's opinion lexicon. Download it once with:

```bash
python -m nltk.downloader opinion_lexicon
```

You can inspect how a tweet is tokenized without the dataset:

```bash
python examples/preprocess_tweet.py
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

The cleaned project currently has **9 tests** covering the baseline tokenizer, tweet-aware tokenization, repeated-character handling, negation marking, stemming, unigram and bigram features, style features, and both classifier variants.

## Dataset

The tweet dataset is not included in this repository because it was not present in the submitted project archive and its redistribution terms are not available here.

The evaluation script expects a tab-separated file with at least three columns:

```text
<id>    <label>    <tweet text>
```

The first column is ignored. The second column contains the label and the third contains the tweet text.

If you have the original dataset locally, run the final model with:

```bash
python scripts/evaluate.py path/to/sentiment-dataset.tsv --variant final
```

To run the original Bag-of-Words baseline instead:

```bash
python scripts/evaluate.py path/to/sentiment-dataset.tsv --variant baseline
```

The script keeps the original 80/20 ordering of the data and uses contiguous folds for cross-validation so that the evaluation setup stays close to the submitted experiment.

## What I took from this project

The most useful part was seeing how much information can be lost before a classifier even starts learning. The baseline score was already above 0.82, so the overall number did not immediately show what needed to change. The confusion matrix and individual mistakes made the problems more concrete. Negative tweets were being missed more often, emoticons could disappear inside badly split tokens, and phrases such as `not good` were being reduced to unrelated single-word features.

That changed how I approached the later experiments. Instead of adding preprocessing steps because they are common in NLP, I could connect several changes to specific errors. Tweet-aware tokenization addressed Twitter punctuation and emoticons. Bigrams kept short phrases together. Class balancing responded to the difference between positive and negative recall. Negation marking gave the model a separate representation for words used inside a negative phrase.

It was also useful to see that standard preprocessing advice is not universal. Removing stop words sounded reasonable, but it reduced performance here. Restricting the vocabulary to more frequent features also lost useful information. Those experiments made the evaluation step as important as the feature itself. A transformation is only useful if it helps on the data being modelled.

The project also gave me a clearer sense of what a linear text classifier can and cannot do. Better features helped it capture local context, but sarcasm and wider meaning remained difficult. The model can learn patterns in the representation it receives. It cannot recover context that the representation never gives it.
