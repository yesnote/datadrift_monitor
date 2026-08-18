import pytest
import torch

from methods.ada_fnp.training.false_negative_targets import count_false_negatives


def _count(pred_boxes, scores, pred_labels, gt_boxes, gt_labels):
    return count_false_negatives(
        torch.tensor(pred_boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(scores, dtype=torch.float32),
        torch.tensor(pred_labels),
        torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(gt_labels),
    )


@pytest.mark.parametrize('prediction_count, expected', [(0, 2), (1, 1)])
def test_no_or_partial_predictions(prediction_count, expected):
    boxes = [[0, 0, 10, 10]][:prediction_count]
    assert _count(boxes, [.9][:prediction_count], [1][:prediction_count],
                  [[0, 0, 10, 10], [20, 20, 30, 30]], [1, 1]) == expected


def test_wrong_class_and_duplicate_do_not_hide_false_negative():
    assert _count(
        [[0, 0, 10, 10], [0, 0, 10, 10]], [.9, .8], [2, 1],
        [[0, 0, 10, 10], [20, 20, 30, 30]], [1, 1],
    ) == 1


def test_iou_threshold_is_inclusive():
    assert _count([[0, 0, 10, 10]], [.5], [1], [[0, 0, 20, 10]], [1]) == 0


def test_no_ground_truth_has_zero_false_negatives():
    assert _count([[0, 0, 10, 10]], [.9], [1], [], []) == 0
