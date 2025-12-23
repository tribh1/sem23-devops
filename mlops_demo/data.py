"""Data loading utilities for the demo MLOps pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    """Container for split data."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class IrisDataLoader:
    """Loads and splits the Iris dataset for classification tasks."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> DatasetSplit:
        iris = datasets.load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name="target")

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return DatasetSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
