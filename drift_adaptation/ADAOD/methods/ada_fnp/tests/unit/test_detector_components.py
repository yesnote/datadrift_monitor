'''Unit tests for the detector-facing ADA-FNP tensor contracts.'''

import pytest
import torch

from methods.ada_fnp.models.detector import (
    SOURCE_BRANCH, TARGET_LABELED_BRANCH, TARGET_UNLABELED_STRONG_BRANCH,
    TARGET_UNLABELED_WEAK_BRANCH, multi_target_domain_loss,
    route_detection_losses, select_domain_target_branches,
    validate_loss_branches)
from methods.ada_fnp.models.roi_head import (
    map_multiclass_nms_to_proposals, normalize_xyxy_box_samples,
    run_rpn_once_for_fixed_roi, summarize_mc_predictions)


def test_validate_loss_branches_enforces_explicit_contract():
    inputs = {
        SOURCE_BRANCH: torch.zeros(2, 3, 8, 8),
        TARGET_UNLABELED_WEAK_BRANCH: torch.zeros(1, 3, 8, 8),
        TARGET_UNLABELED_STRONG_BRANCH: torch.zeros(1, 3, 8, 8),
    }
    samples = {
        SOURCE_BRANCH: [object(), object()],
        TARGET_UNLABELED_WEAK_BRANCH: [object()],
        TARGET_UNLABELED_STRONG_BRANCH: [object()],
    }
    validate_loss_branches(inputs, samples)

    with pytest.raises(ValueError, match='must match'):
        validate_loss_branches(inputs, {SOURCE_BRANCH: samples[SOURCE_BRANCH]})
    with pytest.raises(ValueError, match='unknown ADA-FNP branches'):
        validate_loss_branches(
            {**inputs, 'unexpected': torch.zeros(1, 3, 8, 8)},
            {**samples, 'unexpected': [object()]})
    with pytest.raises(ValueError, match='batch sizes differ'):
        validate_loss_branches(inputs, {
            SOURCE_BRANCH: [object()],
            TARGET_UNLABELED_WEAK_BRANCH: [object()],
            TARGET_UNLABELED_STRONG_BRANCH: [object()],
        })
    mismatched_views = {
        **inputs,
        TARGET_UNLABELED_STRONG_BRANCH: torch.zeros(2, 3, 8, 8),
    }
    mismatched_samples = {
        **samples,
        TARGET_UNLABELED_STRONG_BRANCH: [object(), object()],
    }
    with pytest.raises(ValueError, match='weak and strong batch sizes'):
        validate_loss_branches(mismatched_views, mismatched_samples)


def test_route_detection_losses_keeps_unsupervised_classification_only():
    source = {
        'loss_rpn_cls': torch.tensor(1.0),
        'loss_rpn_bbox': torch.tensor(2.0),
        'loss_cls': torch.tensor(3.0),
        'loss_bbox': torch.tensor(4.0),
    }
    target_labeled = {'loss_cls': torch.tensor(5.0)}
    target_unlabeled = {
        'loss_rpn_cls': torch.tensor(6.0),
        'loss_rpn_bbox': torch.tensor(7.0),
        'loss_cls': torch.tensor(8.0),
        'loss_bbox': torch.tensor(9.0),
    }

    routed = route_detection_losses(
        source, target_labeled, target_unlabeled)

    assert set(routed) == {
        'source.loss_rpn_cls',
        'source.loss_rpn_bbox',
        'source.loss_cls',
        'source.loss_bbox',
        'target_labeled.loss_cls',
        'target_unlabeled_strong.loss_rpn_cls',
        'target_unlabeled_strong.loss_cls',
    }
    assert torch.equal(
        routed['target_unlabeled_strong.loss_cls'],
        target_unlabeled['loss_cls'])


def test_routing_supports_mmdet_list_losses_without_copying_logic():
    rpn_terms = [torch.tensor(1.0)]

    routed = route_detection_losses(
        {'loss_rpn_cls': rpn_terms},
        target_unlabeled_strong_losses={
            'loss_rpn_cls': rpn_terms,
            'loss_cls': torch.tensor(2.0),
        })

    assert routed['source.loss_rpn_cls'][0] is rpn_terms[0]
    assert routed['target_unlabeled_strong.loss_rpn_cls'][0] is rpn_terms[0]


def test_domain_routing_uses_only_the_strong_unlabeled_view():
    branches = {
        TARGET_LABELED_BRANCH: object(),
        TARGET_UNLABELED_WEAK_BRANCH: object(),
        TARGET_UNLABELED_STRONG_BRANCH: object(),
    }

    assert select_domain_target_branches(branches) == (
        TARGET_LABELED_BRANCH, TARGET_UNLABELED_STRONG_BRANCH)


def test_domain_loss_uses_source_one_and_target_zero_labels():
    source_logits = torch.zeros(2, requires_grad=True)
    target_a = torch.zeros(2, requires_grad=True)
    target_b = torch.zeros(2, requires_grad=True)

    loss = multi_target_domain_loss(source_logits, [target_a, target_b])
    loss.backward()

    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert torch.all(source_logits.grad < 0)
    assert torch.all(target_a.grad > 0)
    assert torch.all(target_b.grad > 0)
    assert torch.allclose(target_a.grad, target_b.grad)


def test_summarize_mc_predictions_is_mean_before_nms():
    probabilities = torch.tensor([
        [[0.2, 0.7, 0.1], [0.5, 0.2, 0.3]],
        [[0.4, 0.5, 0.1], [0.3, 0.4, 0.3]],
    ])
    boxes = torch.tensor([
        [[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 4.0, 4.0]],
        [[2.0, 0.0, 4.0, 2.0], [4.0, 2.0, 6.0, 4.0]],
    ])

    mean_probabilities, mean_boxes, variances = summarize_mc_predictions(
        probabilities, boxes)

    assert torch.allclose(mean_probabilities, probabilities.mean(dim=0))
    assert torch.allclose(mean_boxes, boxes.mean(dim=0))
    assert torch.allclose(variances, boxes.var(dim=0, unbiased=True))
    with pytest.raises(ValueError, match='at least two passes'):
        summarize_mc_predictions(probabilities[:1], boxes[:1])


def test_map_multiclass_nms_indices_back_to_proposals():
    kept_indices = torch.tensor([0, 2, 3, 8], dtype=torch.long)

    local = map_multiclass_nms_to_proposals(kept_indices, 3)
    mapped = map_multiclass_nms_to_proposals(
        kept_indices, 3, torch.tensor([10, 20, 30]))

    assert local.tolist() == [0, 0, 1, 2]
    assert mapped.tolist() == [10, 10, 20, 30]


def test_normalize_box_samples_uses_cxcywh_and_image_dimensions():
    samples = torch.tensor([[[10.0, 20.0, 30.0, 60.0]]])

    normalized = normalize_xyxy_box_samples(samples, (100, 200))

    assert torch.allclose(
        normalized, torch.tensor([[[0.1, 0.4, 0.1, 0.4]]]))


def test_fixed_roi_wrapper_materializes_rpn_proposals_once():
    calls = {'rpn': 0, 'roi': 0}
    proposals = object()

    def rpn_predict():
        calls['rpn'] += 1
        return proposals

    def roi_predict(received):
        calls['roi'] += 1
        assert received is proposals
        return 'detections'

    result = run_rpn_once_for_fixed_roi(rpn_predict, roi_predict)

    assert result == 'detections'
    assert calls == {'rpn': 1, 'roi': 1}
