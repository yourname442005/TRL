"""RED tests for model serialization and offline inference."""

from pathlib import Path

import pytest

from trl_intelligence.models.baselines import build_b1
from trl_intelligence.prediction.artifacts import load_model, predict_texts, save_model

TEXTS = [
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
]
LABELS = [1, 1, 3, 3, 5, 5, 7, 7, 9, 9]


def test_save_and_load_roundtrip(tmp_path: Path):
    model = build_b1(random_state=42)
    model.fit(TEXTS, LABELS)
    path = tmp_path / "b1.joblib"
    save_model(model, path)
    assert path.exists()
    loaded = load_model(path)
    queries = ["early laboratory concept only " * 3, "proven operational mature mission " * 4]
    assert list(model.predict(queries)) == list(loaded.predict(queries))


def test_predict_texts_returns_ints(tmp_path: Path):
    model = build_b1(random_state=42)
    model.fit(TEXTS, LABELS)
    path = tmp_path / "b1.joblib"
    save_model(model, path)
    preds = predict_texts(path, ["component validation in lab " * 4])
    assert len(preds) == 1
    assert 1 <= int(preds[0]) <= 9


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")
