"""RED tests for the three baseline models."""

import numpy as np
import pytest

from trl_intelligence.models.baselines import (
    build_b1,
    build_b2,
    build_b3,
    fit_predict,
)

# Tiny separable synthetic set: higher TRL when "mature" word count grows
TRAIN_TEXTS = [
    "early laboratory concept discovery only " * 3,
    "early laboratory concept discovery only " * 3,
    "component validation in lab " * 4,
    "component validation in lab " * 4,
    "prototype demonstrated in relevant environment " * 4,
    "prototype demonstrated in relevant environment " * 4,
    "system qualified in flight " * 4,
    "system qualified in flight " * 4,
    "proven operational at scale mature mission " * 4,
    "proven operational at scale mature mission " * 4,
    "early laboratory concept discovery only " * 3,
    "prototype demonstrated in relevant environment " * 4,
    "component validation in lab " * 4,
    "system qualified in flight " * 4,
    "proven operational at scale mature mission " * 4,
]
TRAIN_LABELS = [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 2, 5, 4, 7, 8]
TEST_TEXTS = [
    "early laboratory concept discovery only " * 3,
    "prototype demonstrated in relevant environment " * 4,
    "proven operational at scale mature mission " * 4,
]
TEST_LABELS = [1, 5, 9]


@pytest.fixture(params=["b1", "b2", "b3"])
def model_name(request):
    return request.param


def _build(name: str):
    if name == "b1":
        return build_b1(random_state=42)
    if name == "b2":
        return build_b2(random_state=42)
    if name == "b3":
        return build_b3(random_state=42)
    raise AssertionError(name)


class TestBuilders:
    def test_b1_is_logreg_pipeline(self, model_name):
        model = _build(model_name)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_fit_predict_labels_in_range(self, model_name):
        model = _build(model_name)
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        preds = model.predict(TEST_TEXTS)
        assert len(preds) == len(TEST_TEXTS)
        assert all(1 <= int(p) <= 9 for p in preds)

    def test_deterministic_fit_same_seed(self, model_name):
        m1 = _build(model_name)
        m2 = _build(model_name)
        m1.fit(TRAIN_TEXTS, TRAIN_LABELS)
        m2.fit(TRAIN_TEXTS, TRAIN_LABELS)
        assert list(m1.predict(TEST_TEXTS)) == list(m2.predict(TEST_TEXTS))


class TestB1Structure:
    """Pin change 1: TF-IDF + balanced multinomial LogisticRegression."""

    def test_steps_are_tfidf_then_logreg(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        model = build_b1(random_state=42)
        names = [name for name, _ in model.steps]
        assert names == ["tfidf", "clf"]
        assert isinstance(dict(model.steps)["tfidf"], TfidfVectorizer)
        assert isinstance(dict(model.steps)["clf"], LogisticRegression)

    def test_logreg_is_balanced(self):
        model = build_b1(random_state=42)
        assert dict(model.steps)["clf"].class_weight == "balanced"

    def test_predict_proba_is_nine_class_multinomial(self):
        model = build_b1(random_state=42)
        # include every TRL class 1..9 so the multinomial head spans all nine
        texts = list(TRAIN_TEXTS) + ["component validation in lab " * 4]
        labels = list(TRAIN_LABELS) + [6]
        model.fit(texts, labels)
        proba = model.predict_proba(TEST_TEXTS)
        # sklearn >= 1.5: lbfgs + >2 classes => multinomial; multi_class attr removed
        assert proba.shape == (len(TEST_TEXTS), 9)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestB2Ordinal:
    def test_uses_eight_thresholds(self):
        model = build_b2(random_state=42)
        # after fit, should expose 8 binary estimators for thresholds 1..8
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        assert len(model.threshold_clfs_) == 8
        assert list(model.thresholds_) == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_predict_is_int_1_to_9(self):
        model = build_b2(random_state=42)
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        preds = model.predict(TEST_TEXTS)
        assert all(isinstance(int(p), int) for p in preds)
        assert all(1 <= int(p) <= 9 for p in preds)

    def test_expected_value_is_one_plus_sum_of_threshold_probs(self):
        model = build_b2(random_state=42)
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        expected = model._expected_value(TEST_TEXTS)
        positive = model._positive_probs(TEST_TEXTS)
        assert np.allclose(expected, 1.0 + positive)
        assert np.all(expected >= 1.0)
        assert np.all(expected <= 9.0)


class TestB3SvdGbc:
    def test_pipeline_includes_svd_then_gbc(self):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.pipeline import Pipeline

        model = build_b3(random_state=42)
        assert isinstance(model, Pipeline)
        names = [name for name, _ in model.steps]
        assert "svd" in names
        assert "clf" in names
        svd = dict(model.steps)["svd"]
        clf = dict(model.steps)["clf"]
        assert isinstance(svd, TruncatedSVD)
        assert isinstance(clf, GradientBoostingClassifier)

    def test_fit_produces_dense_transform(self):
        model = build_b3(random_state=42, n_components=4)
        model.fit(TRAIN_TEXTS, TRAIN_LABELS)
        preds = model.predict(TEST_TEXTS)
        assert len(preds) == 3

    def test_step_order_is_tfidf_svd_clf(self):
        model = build_b3(random_state=42)
        assert [name for name, _ in model.steps] == ["tfidf", "svd", "clf"]

    def test_never_feeds_sparse_into_gradient_boosting(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = build_b3(random_state=42)
        assert not any(
            isinstance(est, HistGradientBoostingClassifier) for _, est in model.steps
        )
        # SVD sits between TF-IDF and the classifier, so the classifier gets dense input
        names = [name for name, _ in model.steps]
        assert names.index("svd") < names.index("clf")


class TestFitPredictHelper:
    def test_fit_predict_returns_array(self):
        model = build_b1(random_state=0)
        preds = fit_predict(model, TRAIN_TEXTS, TRAIN_LABELS, TEST_TEXTS)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 3


class TestBuildFromConfig:
    def test_b1_honors_config_overrides(self):
        from trl_intelligence.models.baselines import build_from_config

        cfg = {
            "seed": 7,
            "tfidf": {"max_features": 13},
            "b1": {"C": 0.25, "max_iter": 55},
        }
        model = build_from_config("b1", cfg)
        assert dict(model.steps)["tfidf"].max_features == 13
        clf = dict(model.steps)["clf"]
        assert clf.C == 0.25
        assert clf.max_iter == 55
        assert clf.random_state == 7
        assert clf.class_weight == "balanced"

    def test_b2_honors_config_overrides(self):
        from trl_intelligence.models.baselines import build_from_config

        cfg = {"seed": 3, "tfidf": {"max_features": 21}, "b2": {"max_iter": 44}}
        model = build_from_config("b2", cfg)
        assert model.max_iter == 44
        assert model.random_state == 3
        assert model.tfidf is not None
        assert model.tfidf.max_features == 21

    def test_b3_honors_config_overrides(self):
        from trl_intelligence.models.baselines import build_from_config

        cfg = {
            "seed": 1,
            "tfidf": {"min_df": 1},
            "b3": {"n_components": 5, "n_estimators": 3},
        }
        model = build_from_config("b3", cfg)
        assert dict(model.steps)["svd"].n_components == 5
        assert dict(model.steps)["clf"].n_estimators == 3
        assert dict(model.steps)["tfidf"].min_df == 1

    def test_ngram_range_list_from_json_becomes_tuple(self):
        from trl_intelligence.models.baselines import build_from_config

        cfg = {"seed": 1, "tfidf": {"ngram_range": [1, 2]}, "b1": {}}
        model = build_from_config("b1", cfg)
        assert tuple(dict(model.steps)["tfidf"].ngram_range) == (1, 2)

    def test_unknown_model_raises(self):
        from trl_intelligence.models.baselines import build_from_config

        with pytest.raises(ValueError, match="unknown model"):
            build_from_config("b9", {"seed": 1})

    def test_defaults_match_approved_config_values(self):
        import json
        from pathlib import Path

        from trl_intelligence.models.baselines import build_from_config

        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "baseline_v1.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        b1 = build_from_config("b1", cfg)
        assert dict(b1.steps)["clf"].C == 1.0
        assert dict(b1.steps)["tfidf"].max_features == 50000
        b3 = build_from_config("b3", cfg)
        assert dict(b3.steps)["svd"].n_components == 100
        assert dict(b3.steps)["clf"].n_estimators == 100
