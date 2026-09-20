# Dataset format

The dataset used in the original experiment is not included in this repository.

The code expects a tab-separated file with at least three columns:

```text
<id>    <label>    <tweet text>
```

The first column is ignored. The second column must contain the sentiment label, such as `positive` or `negative`. The third column contains the tweet text.

Place the file anywhere on your machine and pass its path to `scripts/evaluate.py`.
