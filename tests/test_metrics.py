"""RED tests for evaluation metrics."""

import pytest

from trl_intelligence.evaluation.metrics import (
    classification_metrics,
    confusion_matrix_9x9,
    mae,
    relative_mae_delta,
    within_1_accuracy,
)


class TestMae:
    def test_exact(self):
        y_true = [1, 2, 3, 9]
        y_pred = [1, 3, 3, 5]
        # |0|+|1|+|0|+|4| = 5/4 = 1.25
        assert mae(y_true, y_pred) == pytest.approx(1.25)

    def test_perfect(self):
        assert mae([1, 5, 9], [1, 5, 9]) == 0.0


class TestWithin1:
    def test_counts(self):
        y_true = [1, 2, 3, 9]
        y_pred = [1, 4, 3, 5]  # ok, no, ok, no
        assert within_1_accuracy(y_true, y_pred) == pytest.approx(0.5)


class TestClassificationMetrics:
    def test_returns_required_keys(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        m = classification_metrics(y_true, y_pred)
        for key in ("mae", "within_1", "macro_f1", "qwk", "accuracy", "per_class", "confusion"):
            assert key in m

    def test_perfect_predictions(self):
        y_true = list(range(1, 10)) * 2
        y_pred = list(y_true)
        m = classification_metrics(y_true, y_pred)
        assert m["mae"] == 0.0
        assert m["within_1"] == 1.0
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == pytest.approx(1.0)
        assert m["qwk"] == pytest.approx(1.0)

    def test_per_class_covers_1_to_9(self):
        m = classification_metrics([1, 2], [1, 2])
        assert set(m["per_class"].keys()) == {str(i) for i in range(1, 10)}


class TestConfusion:
    def test_shape_9x9(self):
        cm = confusion_matrix_9x9([1, 9, 5], [1, 8, 5])
        assert len(cm) == 9
        assert all(len(row) == 9 for row in cm)
        assert cm[0][0] == 1  # true 1 -> pred 1
        assert cm[8][7] == 1  # true 9 -> pred 8


class TestRelativeMaeDelta:
    def test_absolute_and_relative(self):
        # abs delta and relative change from baseline to new
        result = relative_mae_delta(mae_full=1.2, mae_masked=1.0)
        assert result["abs_delta"] == pytest.approx(0.2)
        # relative = (full - masked) / full  (positive => masked improved)
        assert result["rel_delta"] == pytest.approx(0.2 / 1.2)

    def test_zero_baseline_raises(self):
        with pytest.raises(ValueError):
            relative_mae_delta(mae_full=0.0, mae_masked=1.0)
