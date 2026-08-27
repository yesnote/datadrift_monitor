'''AADA defaults under the ADA-FNP Cityscapes-to-Foggy protocol.'''

from copy import deepcopy

from methods.common.protocols.ada_fnp_detection import (
    ACQUISITION_MILESTONES,
    MAXIMUM_DETECTOR_ITERATION,
)


_CONFIG = {
    'method': 'aada',
    'scenario': 'cityscapes-to-foggy',
    'seed': 0,
    'detector': {
        'name': 'faster-rcnn-vgg16',
        'class_agnostic_bbox_regression': False,
    },
    'training': {
        'max_iterations': MAXIMUM_DETECTOR_ITERATION,
        'acquisition_milestones': ACQUISITION_MILESTONES,
        'lr': 0.02,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'gradient_clip_max_norm': 10.0,
        'gradient_clip_norm_type': 2.0,
        'warmup_iterations': 400,
        'warmup_start_factor': 0.001,
        'lr_milestones': (30000, 35000),
        'lr_decay_factor': 0.1,
        'source_batch_size': 4,
        'target_labeled_batch_size': 4,
        'target_unlabeled_batch_size': 4,
    },
    'domain_adaptation': {
        'loss_weight': 0.01,
        'gradient_reversal_scale': 1.0,
    },
    'acquisition': {
        'budget_percent': 1.0,
        'domain_probability_epsilon': 0.000001,
    },
    'inference': {
        'acquisition_batch_size': 4,
        'evaluation_batch_size': 4,
    },
}


def get_config() -> dict:
    return deepcopy(_CONFIG)
