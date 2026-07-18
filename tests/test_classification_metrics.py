import numpy as np
import pytest

from exomanifold.ml.metrics.classification import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def test_accuracy_perfect():
    y = np.array([0, 1, 1, 0])

    assert accuracy_score(y, y) == pytest.approx(1.0)


def test_accuracy_half():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])

    assert accuracy_score(y_true, y_pred) == pytest.approx(0.5)


def test_accuracy_zero():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])

    assert accuracy_score(y_true, y_pred) == pytest.approx(0.0)


def test_accuracy_length_mismatch():
    with pytest.raises(ValueError):
        accuracy_score(
            np.array([0, 1]),
            np.array([0]),
        )


def test_accuracy_empty():
    with pytest.raises(ValueError):
        accuracy_score(
            np.array([]),
            np.array([]),
        )


def test_confusion_matrix():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    expected = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )

    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred),
        expected,
    )


def test_confusion_matrix_perfect():
    y = np.array([0, 0, 1, 1])

    expected = np.array(
        [
            [2, 0],
            [0, 2],
        ]
    )

    np.testing.assert_array_equal(
        confusion_matrix(y, y),
        expected,
    )


def test_confusion_matrix_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])

    expected = np.array(
        [
            [0, 2],
            [2, 0],
        ]
    )

    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred),
        expected,
    )


def test_confusion_matrix_invalid_lengths():
    with pytest.raises(ValueError):
        confusion_matrix(
            np.array([0, 1]),
            np.array([0]),
        )


def test_confusion_matrix_multiclass():
    with pytest.raises(ValueError):
        confusion_matrix(
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
        )

def test_precision_score():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    assert precision_score(
        y_true,
        y_pred,
    ) == pytest.approx(0.5)

def test_recall_score():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    assert recall_score(
        y_true,
        y_pred,
    ) == pytest.approx(0.5)

def test_f1_score():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    assert f1_score(
        y_true,
        y_pred,
    ) == pytest.approx(0.5)

def test_metrics_perfect():
    y = np.array([0, 1, 1, 0])

    assert precision_score(y, y) == pytest.approx(1.0)
    assert recall_score(y, y) == pytest.approx(1.0)
    assert f1_score(y, y) == pytest.approx(1.0)

def test_metrics_zero():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])

    assert precision_score(y_true, y_pred) == 0.0
    assert recall_score(y_true, y_pred) == 0.0
    assert f1_score(y_true, y_pred) == 0.0

def test_classification_report_keys():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])

    expected = {
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "support",
    }
    
    assert set(classification_report(y_true, y_pred).keys()) == expected

def test_classification_report_values():

    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])

    report = classification_report(y_true, y_pred)

    assert report["accuracy"] == accuracy_score(y_true, y_pred)
    assert report["precision"] == precision_score(y_true, y_pred)
    assert report["recall"] == recall_score(y_true, y_pred)
    assert report["f1_score"] == f1_score(y_true, y_pred)
    assert report["support"] == len(y_true)

def test_classification_report_perfect():
    
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])

    report = classification_report(y_true, y_pred)

    assert report["accuracy"] == 1.0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["f1_score"] == 1.0
    assert report["support"] == 3

def test_classification_report_zero_predictions():

    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])

    report = classification_report(y_true, y_pred)

    assert report["accuracy"] == 0.0
    assert report["precision"] == 0.0
    assert report["recall"] == 0.0
    assert report["f1_score"] == 0.0
    assert report["support"] == 3

def test_classification_report_mixed():

    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    report = classification_report(y_true, y_pred)

    assert report["accuracy"] == accuracy_score(y_true, y_pred)
    assert report["precision"] == precision_score(y_true, y_pred)
    assert report["recall"] == recall_score(y_true, y_pred)
    assert report["f1_score"] == f1_score(y_true, y_pred)
