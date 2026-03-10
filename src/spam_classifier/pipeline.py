from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def words_in_texts(words: Iterable[str], texts: pd.Series) -> np.ndarray:
    """
    Binary indicator matrix:
    X[i, j] = 1 if words[j] appears in texts[i], else 0
    """
    texts = texts.fillna("").astype(str)
    words = list(words)
    X = np.zeros((len(texts), len(words)), dtype=int)

    for j, w in enumerate(words):
        X[:, j] = texts.str.contains(w, case=False, regex=False).to_numpy(dtype=int)

    return X


# You can replace this list with your Data 100 keyword list
WORD_FEATURES: List[str] = [
    "click", "free", "remove", "reply", "receive", "offer", "win",
    "cash", "credit", "loan", "sale", "money", "guarantee", "risk", "html"
]


def make_engineered_features(texts: pd.Series) -> np.ndarray:
    """
    Extra numeric features.
    """
    texts = texts.fillna("").astype(str)

    num_chars = texts.str.len()
    num_urls = texts.str.count(r"http|www")
    num_words = texts.str.split().str.len()
    num_exclaim = texts.str.count("!")

    is_reply = texts.str.contains(r"^re:", regex=True, case=False).astype(int)
    is_forward = texts.str.contains(r"^fwd:", regex=True, case=False).astype(int)

    return np.column_stack([
        num_chars.to_numpy(),
        num_urls.to_numpy(),
        num_words.to_numpy(),
        num_exclaim.to_numpy(),
        is_reply.to_numpy(),
        is_forward.to_numpy(),
    ])


def process_data(df: pd.DataFrame, words: Optional[Iterable[str]] = None) -> np.ndarray:
    """
    Build feature matrix X from df with an 'email' column.
    """
    if words is None:
        words = WORD_FEATURES

    emails = df["email"].fillna("").astype(str)
    word_ind = words_in_texts(words, emails)
    engineered = make_engineered_features(emails)

    return np.hstack([word_ind, engineered])


@dataclass
class TrainResult:
    model: LogisticRegression
    best_params: dict
    feature_dim: int


def train_model(train_df: pd.DataFrame, label_col: str = "spam") -> TrainResult:
    """
    Train LogisticRegression with GridSearchCV.
    """
    X = process_data(train_df)
    y = train_df[label_col].to_numpy()

    param_grid = {
        "C": [0.1, 1, 3, 10],
        "penalty": ["l2"],
        "solver": ["liblinear"],
        "max_iter": [2000],
    }

    grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X, y)

    best_model: LogisticRegression = grid.best_estimator_
    return TrainResult(model=best_model, best_params=grid.best_params_, feature_dim=X.shape[1])