import pytest
import torch

from methods.ada_fnp.training.pseudo_labels import (
    classification_only_losses, project_boxes, project_pseudo_labels,
    select_pseudo_labels, weak_to_strong_homography,
)


def test_variance_threshold_is_inclusive_and_has_no_score_cutoff():
    boxes = torch.tensor([[0., 0., 10., 10.], [2., 2., 4., 4.]])
    probabilities = torch.tensor([[.01, .02, .97], [.2, .1, .7]])
    variances = torch.tensor([[.1, .1, .1, .1], [.11, .1, .1, .1]])
    result = select_pseudo_labels(boxes, probabilities, variances)
    assert result['boxes'].shape == (1, 4)
    assert result['scores'].item() == pytest.approx(.02)


def test_box_projection_translation():
    boxes = torch.tensor([[0., 0., 10., 20.]])
    homography = torch.tensor([[1., 0., 3.], [0., 1., 4.], [0., 0., 1.]])
    expected = torch.tensor([[3., 4., 13., 24.]])
    assert torch.equal(project_boxes(boxes, homography), expected)


def test_weak_to_strong_homography_uses_strong_times_inverse_weak():
    weak = torch.tensor([[2., 0., 0.], [0., 2., 0.], [0., 0., 1.]])
    strong = torch.tensor([[1., 0., 3.], [0., 1., 4.], [0., 0., 1.]])

    transform = weak_to_strong_homography(weak, strong)

    assert torch.allclose(transform, strong @ torch.linalg.inv(weak))


def test_project_pseudo_labels_filters_only_variance_before_projection():
    boxes = torch.tensor([
        [2., 4., 6., 8.],
        [20., 40., 60., 80.],
    ])
    probabilities = torch.tensor([
        [.001, .002, .997],
        [.8, .1, .1],
    ])
    variances = torch.tensor([
        [.1, .1, .1, .1],
        [.2, .2, .2, .2],
    ])
    weak = torch.tensor([[2., 0., 0.], [0., 2., 0.], [0., 0., 1.]])
    strong = torch.tensor([[1., 0., 3.], [0., 1., 4.], [0., 0., 1.]])

    result = project_pseudo_labels(
        boxes, probabilities, variances, weak, strong)

    assert torch.allclose(result['boxes'], torch.tensor([[4., 6., 6., 8.]]))
    assert result['scores'].item() == pytest.approx(.002)
    assert result['labels'].tolist() == [1]


def test_only_classification_losses_are_retained():
    losses = {
        'loss_rpn_cls': torch.tensor(1.),
        'loss_rpn_bbox': torch.tensor(2.),
        'loss_cls': torch.tensor(3.),
        'loss_bbox': torch.tensor(4.),
    }
    assert set(classification_only_losses(losses)) == {'loss_rpn_cls', 'loss_cls'}
