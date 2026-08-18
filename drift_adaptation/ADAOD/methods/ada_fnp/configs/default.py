'''ADA-FNP defaults for Cityscapes to Foggy Cityscapes.'''

from copy import deepcopy


_CONFIG = {
    'method': 'ada-fnp',
    'scenario': 'cityscapes-to-foggy',
    'seed': 0,
    'dataset': {
        'source': {
            'name': 'cityscapes',
            'image_root': 'data/leftImg8bit',
            'annotation_root': 'data/gtFine',
            'split': 'train',
            'expected_images': 2975,
        },
        'target': {
            'name': 'foggy_cityscapes',
            'image_root': 'data/leftImg8bit_foggy',
            'annotation_root': 'data/gtFine',
            'train_split': 'train',
            'eval_split': 'val',
            'beta': 0.02,
            'expected_train_images': 2975,
            'expected_eval_images': 500,
        },
        'classes': (
            'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle',
        ),
    },
    'detector': {
        'name': 'faster-rcnn-vgg16',
        'num_classes': 8,
        'dropout_probability': 0.1,
        'class_agnostic_bbox_regression': True,
    },
    'training': {
        'max_iterations': 40000,
        'acquisition_milestones': (5000, 10000, 15000, 20000, 25000),
        'lr': 0.02,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'lr_milestones': (30000, 35000),
        'source_batch_size': 4,
        'target_labeled_batch_size': 4,
        'target_unlabeled_batch_size': 4,
        'ema_decay': 0.9996,
        'mmdet_ema_momentum': 0.0004,
        'adversarial_weight': 0.01,
    },
    'fnpm': {
        'iterations_per_round': 2000,
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
