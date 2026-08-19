'''ADA-FNP defaults for Cityscapes to Foggy Cityscapes.'''

from copy import deepcopy

from methods.ada_fnp.phases import (
    FNPM_ITERATIONS_PER_ROUND,
    FNPM_MILESTONES,
)


_CONFIG = {
    'method': 'ada-fnp',
    'scenario': 'cityscapes-to-foggy',
    'seed': 0,
    'detector': {
        'name': 'faster-rcnn-vgg16',
        'dropout_probability': 0.1,
        'class_agnostic_bbox_regression': False,
    },
    'training': {
        'max_iterations': 40000,
        'acquisition_milestones': FNPM_MILESTONES,
        'lr': 0.02,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'warmup_iterations': 400,
        'warmup_start_factor': 0.001,
        'lr_milestones': (30000, 35000),
        'source_batch_size': 4,
        'target_labeled_batch_size': 4,
        'target_unlabeled_batch_size': 4,
        'ema_decay': 0.9996,
        'mmdet_ema_momentum': 0.0004,
        'adversarial_weight': 0.01,
    },
    'fnpm': {
        'iterations_per_round': FNPM_ITERATIONS_PER_ROUND,
        'lr': 0.0001,
        'matcher_iou_threshold': 0.5,
        'max_detections': 100,
    },
    'acquisition': {
        'budget_percent': 1.0,
        'mc_passes': 10,
        'dropout_probability': 0.1,
        'domain_probability_epsilon': 0.000001,
        'constant_score_normalized_value': 0.5,
        'empty_detection_final_score': 0.0,
    },
    'pseudo_label': {
        'localization_variance_threshold': 0.1,
        'hard_confidence_threshold': None,
    },
}


def get_config() -> dict:
    '''Return an independent mutable copy of the default configuration.'''

    return deepcopy(_CONFIG)
