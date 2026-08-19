'''Serial five-round ADA-FNP experiment plan.'''

from typing import List, Mapping

from methods.ada_fnp.schedule import (
    ACQUISITION_MILESTONES,
    INITIAL_DETECTOR_SEGMENT,
    resolve_total_budget,
)
from methods.common.contracts import ExperimentPlan, StageSpec
from methods.common.data.pool import split_budget


def build_plan(config: Mapping) -> ExperimentPlan:
    milestones = tuple(config['training']['acquisition_milestones'])
    if milestones != ACQUISITION_MILESTONES:
        raise ValueError('ADA-FNP requires acquisition at 5k through 25k')
    budgets = split_budget(resolve_total_budget(config), len(milestones))
    initial_start, initial_end = INITIAL_DETECTOR_SEGMENT
    stages: List[StageSpec] = [
        StageSpec(
            'prepare_vgg16_caffe_weights',
            'ada_fnp.prepare_vgg16_caffe_weights',
        ),
        StageSpec(
            'prepare_cityscapes_to_foggy',
            'ada_fnp.prepare_cityscapes_to_foggy',
        ),
        StageSpec(
            'train_detector_{:05d}_{:05d}'.format(
                initial_start,
                initial_end,
            ),
            'ada_fnp.train_detector',
            {
                'start_iteration': initial_start,
                'end_iteration': initial_end,
            },
        ),
    ]
    for index, (milestone, budget) in enumerate(zip(milestones, budgets), 1):
        token = f'{index:02d}'
        stages.extend((
            StageSpec(
                f'train_false_negative_predictor_round_{token}',
                'ada_fnp.train_false_negative_predictor',
                {
                    'round': index,
                    'iterations': config['false_negative_predictor'][
                        'iterations_per_round'
                    ],
                },
            ),
            StageSpec(
                f'score_unlabeled_pool_round_{token}',
                'ada_fnp.score_unlabeled_pool',
                {'round': index},
            ),
            StageSpec(
                f'select_samples_round_{token}', 'ada_fnp.select_samples',
                {'round': index, 'budget': budget},
            ),
            StageSpec(
                f'reveal_selected_annotations_round_{token}',
                'ada_fnp.reveal_selected_annotations',
                {'round': index},
            ),
        ))
        end_iteration = (
            milestones[index]
            if index < len(milestones)
            else config['training']['max_iterations']
        )
        stages.append(StageSpec(
            f'train_detector_{milestone:05d}_{end_iteration:05d}',
            'ada_fnp.train_detector',
            {
                'start_iteration': milestone,
                'end_iteration': end_iteration,
            },
        ))
    stages.append(StageSpec(
        'evaluate_final_teacher', 'ada_fnp.evaluate_final_teacher',
        {'metric': 'AP50'},
    ))
    return ExperimentPlan(tuple(stages))
