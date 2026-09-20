"""Loading and splitting the tweet sentiment dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    text: str
    label: str


def load_tsv(path: str | Path) -> list[Sample]:
    """Load the three-column TSV format used by the original experiment.

    Column 0 is ignored, column 1 contains the sentiment label, and column 2
    contains the tweet text.
    """
    path = Path(path)
    samples: list[Sample] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 3:
                raise ValueError(
                    f"Row {row_number} has {len(row)} columns. Expected at least 3."
                )
            samples.append(Sample(text=row[2], label=row[1]))

    return samples


def train_test_split(
    samples: list[Sample],
    train_fraction: float = 0.8,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples in their original order, matching the submitted experiment."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    split_index = int(train_fraction * len(samples))
    return samples[:split_index], samples[split_index:]
