"""Segmentation and classification metrics."""

from __future__ import annotations

import numpy as np
import pytest

from medmlops.metrics import (
    best_threshold,
    classification_metrics,
    expected_calibration_error,
    segmentation_metrics,
)


def test_segmentation_perfect_prediction():
    target = np.zeros((2, 16, 16), dtype=np.uint8)
    target[0, 4:12, 4:12] = 1
    target[1, 0:8, 0:8] = 1
    metrics = segmentation_metrics(target.copy(), target)
    assert metrics["dice"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["iou"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["global_dice"] == pytest.approx(1.0, abs=1e-6)


def test_segmentation_empty_prediction():
    target = np.ones((1, 8, 8), dtype=np.uint8)
    pred = np.zeros((1, 8, 8), dtype=np.uint8)
    metrics = segmentation_metrics(pred, target)
    assert metrics["dice"] < 1e-6
    assert metrics["recall"] < 1e-6
    assert metrics["specificity"] == pytest.approx(1.0)


def test_segmentation_accepts_channel_dim_and_tensor():
    import torch

    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1
    metrics = segmentation_metrics(target, target)
    assert metrics["dice"] == pytest.approx(1.0, abs=1e-6)


def test_best_threshold_recovers_separation():
    prob = np.zeros((1, 10, 10), dtype=np.float32)
    target = np.zeros((1, 10, 10), dtype=np.uint8)
    prob[0, :5, :5] = 0.9
    target[0, :5, :5] = 1
    threshold = best_threshold(prob, target)
    assert 0.3 <= threshold <= 0.75
    assert segmentation_metrics(prob >= threshold, target)["dice"] > 0.9


def test_expected_calibration_error_bounds():
    labels = np.array([0, 1, 0, 1])
    confident_and_correct = np.eye(2)[labels]
    assert expected_calibration_error(confident_and_correct, labels) == pytest.approx(0.0, abs=1e-6)

    confident_and_wrong = np.eye(2)[1 - labels]
    assert expected_calibration_error(confident_and_wrong, labels) == pytest.approx(1.0, abs=1e-6)

    uniform = np.full((4, 2), 0.5)
    assert expected_calibration_error(uniform, np.ones(4, dtype=int)) == pytest.approx(0.5, abs=1e-6)


def test_classification_metrics_binary():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
        ]
    )
    metrics = classification_metrics(y_true, y_prob, class_names=["normal", "pneumonia"])
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert len(metrics["confusion_matrix"]) == 2
    assert set(metrics["per_class_recall"]) == {"normal", "pneumonia"}


def test_classification_metrics_handles_single_class_auc():
    y_true = np.zeros(5, dtype=int)
    y_prob = np.tile([0.6, 0.4], (5, 1))
    metrics = classification_metrics(y_true, y_prob)
    assert np.isnan(metrics["auc"])
    assert metrics["accuracy"] == pytest.approx(1.0)
