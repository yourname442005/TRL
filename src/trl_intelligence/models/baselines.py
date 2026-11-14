"""Baseline model builders: B1 LogReg, B2 cumulative ordinal LR, B3 TF-IDF+SVD+GBC."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def default_tfidf(**overrides) -> TfidfVectorizer:
    params: dict = {
        "min_df": 2,
        "max_df": 0.95,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "max_features": 50_000,
    }
    params.update(overrides)
    return TfidfVectorizer(**params)


def _tfidf_from(tfidf_params: dict | None) -> TfidfVectorizer:
    return default_tfidf(**(tfidf_params or {}))


def build_b1(
    random_state: int = 42,
    c: float = 1.0,
    max_iter: int = 1000,
    tfidf_params: dict | None = None,
) -> Pipeline:
    """B1: TF-IDF + balanced multinomial logistic regression."""
    return Pipeline(
        [
            ("tfidf", _tfidf_from(tfidf_params)),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


class _ConstantProbEstimator:
    """Stores a fixed P(y>t) when training labels for a threshold are constant."""

    def __init__(self, prob: float):
        self.prob = float(prob)

    def predict_proba(self, X) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        p0 = 1.0 - self.prob
        out = np.empty((n, 2), dtype=float)
        out[:, 0] = p0
        out[:, 1] = self.prob
        return out


class CumulativeOrdinalLogReg(BaseEstimator, ClassifierMixin):
    """B2: eight binary threshold logistic regressions for ordinal TRL.

    Fits TF-IDF once, then for t in 1..8 fits P(y > t | x).
    Prediction is ``clip(round(1 + sum_t P(y > t)), 1, 9)``.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
        tfidf: TfidfVectorizer | None = None,
    ):
        self.random_state = random_state
        self.max_iter = max_iter
        self.tfidf = tfidf

    def fit(self, X: Sequence[str], y: Sequence[int]) -> CumulativeOrdinalLogReg:
        y_arr = np.asarray(y, dtype=int)
        if y_arr.ndim != 1 or len(y_arr) == 0:
            raise ValueError("y must be a non-empty 1d array")
        self.classes_ = np.arange(1, 10)
        self.thresholds_ = list(range(1, 9))
        vectorizer = self.tfidf if self.tfidf is not None else default_tfidf()
        # min_df=2 can empty tiny test vocabs; fall back once
        try:
            Xv = vectorizer.fit_transform(X)
            if Xv.shape[1] == 0:
                raise ValueError("empty vocabulary")
        except ValueError:
            vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
            Xv = vectorizer.fit_transform(X)
        self.tfidf_ = vectorizer
        self.threshold_clfs_: list = []
        for t in self.thresholds_:
            binary = (y_arr > t).astype(int)
            if len(np.unique(binary)) < 2:
                est: object = _ConstantProbEstimator(prob=float(binary[0]))
            else:
                est = LogisticRegression(
                    max_iter=self.max_iter,
                    class_weight="balanced",
                    random_state=self.random_state,
                )
                est.fit(Xv, binary)
            self.threshold_clfs_.append(est)
        return self

    def _positive_probs(self, X: Sequence[str]) -> np.ndarray:
        Xv = self.tfidf_.transform(X)
        total = np.zeros(Xv.shape[0], dtype=float)
        for est in self.threshold_clfs_:
            proba = est.predict_proba(Xv)
            # column for class 1
            classes = list(getattr(est, "classes_", [0, 1]))
            idx = classes.index(1)
            total += proba[:, idx]
        return total

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        expected = self._expected_value(X)
        rounded = np.clip(np.rint(expected).astype(int), 1, 9)
        proba = np.zeros((len(rounded), 9), dtype=float)
        proba[np.arange(len(rounded)), rounded - 1] = 1.0
        return proba

    def _expected_value(self, X: Sequence[str]) -> np.ndarray:
        # E[y] = 1 + sum_{t=1}^{8} P(y > t)
        return 1.0 + self._positive_probs(X)

    def predict(self, X: Sequence[str]) -> np.ndarray:
        expected = self._expected_value(X)
        return np.clip(np.rint(expected).astype(int), 1, 9)


def build_b2(
    random_state: int = 42,
    max_iter: int = 1000,
    tfidf_params: dict | None = None,
) -> CumulativeOrdinalLogReg:
    return CumulativeOrdinalLogReg(
        random_state=random_state,
        max_iter=max_iter,
        tfidf=_tfidf_from(tfidf_params),
    )


class _DynamicTruncatedSVD(TruncatedSVD):
    """TruncatedSVD that shrinks n_components to fit the matrix width."""

    def fit(self, X, y=None):
        n_features = int(X.shape[1])
        if self.n_components >= n_features:
            object.__setattr__(self, "n_components", max(1, n_features - 1))
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        n_features = int(X.shape[1])
        if self.n_components >= n_features:
            object.__setattr__(self, "n_components", max(1, n_features - 1))
        return super().fit_transform(X, y)


def build_b3(
    random_state: int = 42,
    n_components: int = 100,
    n_estimators: int = 100,
    tfidf_params: dict | None = None,
) -> Pipeline:
    """B3: TF-IDF -> TruncatedSVD (dense) -> GradientBoostingClassifier."""
    return Pipeline(
        [
            ("tfidf", _tfidf_from(tfidf_params)),
            ("svd", _DynamicTruncatedSVD(n_components=n_components, random_state=random_state)),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=n_estimators, random_state=random_state
                ),
            ),
        ]
    )


def build_from_config(name: str, cfg: dict):
    """Build a baseline by name from the baseline_v1 JSON config."""
    seed = int(cfg.get("seed", 42))
    tfidf_params = dict(cfg.get("tfidf") or {})
    ngram = tfidf_params.get("ngram_range")
    if isinstance(ngram, list):
        tfidf_params["ngram_range"] = tuple(ngram)
    if name == "b1":
        b1 = dict(cfg.get("b1") or {})
        return build_b1(
            random_state=seed,
            c=float(b1.get("C", 1.0)),
            max_iter=int(b1.get("max_iter", 1000)),
            tfidf_params=tfidf_params,
        )
    if name == "b2":
        b2 = dict(cfg.get("b2") or {})
        return build_b2(
            random_state=seed,
            max_iter=int(b2.get("max_iter", 1000)),
            tfidf_params=tfidf_params,
        )
    if name == "b3":
        b3 = dict(cfg.get("b3") or {})
        return build_b3(
            random_state=seed,
            n_components=int(b3.get("n_components", 100)),
            n_estimators=int(b3.get("n_estimators", 100)),
            tfidf_params=tfidf_params,
        )
    raise ValueError(f"unknown model {name}")


def fit_predict(model, texts_train, y_train, texts_test) -> np.ndarray:
    model.fit(texts_train, y_train)
    return np.asarray(model.predict(texts_test))
