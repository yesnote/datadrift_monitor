'''ADA-FNP defaults for Cityscapes to Foggy Cityscapes.'''

from copy import deepcopy

from methods.ada_fnp.schedule import (
    ACQUISITION_MILESTONES,
    FALSE_NEGATIVE_TRAINING_ITERATIONS_PER_ROUND,
    MAXIMUM_DETECTOR_ITERATION,
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
        'teacher_ema_decay': 0.9996,
    },
    'domain_adaptation': {
        'loss_weight': 0.01,
        'gradient_reversal_scale': 1.0,
    },
    'false_negative_predictor': {
        'iterations_per_round': (
            FALSE_NEGATIVE_TRAINING_ITERATIONS_PER_ROUND
        ),
        'lr': 0.0001,
        'matcher_iou_threshold': 0.5,
        'max_detections': 100,
    },
    'acquisition': {
        'budget_percent': 1.0,
        'mc_passes': 10,
        'domain_probability_epsilon': 0.000001,
        'constant_score_normalized_value': 0.5,
        'empty_detection_final_score': 0.0,
    },
    'inference': {
        'acquisition_batch_size': 4,
        'evaluation_batch_size': 4,
    },
    'pseudo_label': {
        'localization_variance_threshold': 0.1,
    },
}


def get_config() -> dict:
    '''Return an independent mutable copy of the default configuration.'''

    return deepcopy(_CONFIG)
