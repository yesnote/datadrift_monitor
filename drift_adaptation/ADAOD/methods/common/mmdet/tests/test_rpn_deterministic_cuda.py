'''CUDA regression for deterministic RPN target assignment.'''

from copy import deepcopy

import pytest
import torch

pytest.importorskip('mmcv.ops')

from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from configs._base_.models.faster_rcnn_vgg16_factory import (
    build_faster_rcnn_vgg16,
)
from mmdet.registry import MODELS


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
def test_rpn_bbox_weights_support_deterministic_cuda_indexing():
    '''Exercise the MMDetection target path that failed on torch 2.0.1.'''
    init_default_scope('mmdet')
    model_config = build_faster_rcnn_vgg16()
    head_config = deepcopy(model_config['rpn_head'])
    head_config['train_cfg'] = deepcopy(model_config['train_cfg']['rpn'])
    head_config['test_cfg'] = deepcopy(model_config['test_cfg']['rpn'])
    head = MODELS.build(head_config)

    device = torch.device('cuda')
    flat_anchors = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0]], device=device
    ).repeat(100, 1)
    valid_flags = torch.ones(100, dtype=torch.bool, device=device)
    gt_instances = InstanceData(
        bboxes=torch.tensor(
            [[0.0, 0.0, 10.0, 10.0]], device=device
        ),
        labels=torch.zeros(1, dtype=torch.long, device=device),
    )

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        targets = head._get_targets_single(
            flat_anchors,
            valid_flags,
            gt_instances,
            {'img_shape': (20, 20)},
            unmap_outputs=False,
        )
        bbox_weights = targets[3]
        positive_indices = targets[4]
        torch.cuda.synchronize(device)
    finally:
        torch.use_deterministic_algorithms(
            deterministic, warn_only=warn_only
        )
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, device)

    assert positive_indices.numel() == 64
    assert torch.equal(
        bbox_weights[positive_indices],
        torch.ones_like(bbox_weights[positive_indices]),
    )
