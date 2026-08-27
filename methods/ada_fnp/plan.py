'''Serial ADA-FNP experiment plans for active and zero-budget UDA.'''

from typing import List, Mapping

from methods.ada_fnp.schedule import (
    ACQUISITION_MILESTONES,
    ADAPTATION_DETECTOR_SEGMENTS,
    INITIAL_DETECTOR_SEGMENT,
    resolve_total_budget,
)
from methods.common.contracts import ExperimentPlan, StageSpec
from methods.common.data.pool import split_budget


def build_plan(config: Mapping) -> ExperimentPlan:
    milestones = tuple(config['training']['acquisition_milestones'])
    if milestones != ACQUISITION_MILESTONES:
        raise ValueError('ADA-FNP requires acquisition at 5k through 25k')
    total_budget = resolve_total_budget(config)
    budgets = split_budget(total_budget, len(milestones))
    stages: List[StageSpec] = [
        StageSpec(
            'prepare_vgg16_caffe_weights',
            'ada_fnp.prepare_vgg16_caffe_weights',
        ),
        StageSpec(
            'prepare_cityscapes_to_foggy',
            'ada_fnp.prepare_cityscapes_to_foggy',
        ),
    ]

    def append_detector_and_evaluation(
        start_iteration: int,
        end_iteration: int,
    ) -> None:
        stages.append(StageSpec(
            'train_detector_{:05d}_{:05d}'.format(
                start_iteration,
                end_iteration,
            ),
            'ada_fnp.train_detector',
            {
                'start_iteration': start_iteration,
                'end_iteration': end_iteration,
            },
        ))
        stages.append(StageSpec(
            'evaluate_teacher_{:05d}'.format(end_iteration),
            'ada_fnp.evaluate_teacher_checkpoint',
            {
                'iteration': end_iteration,
                'metric': 'AP50',
            },
        ))

    append_detector_and_evaluation(*INITIAL_DETECTOR_SEGMENT)
    if total_budget == 0:
        for segment in ADAPTATION_DETECTOR_SEGMENTS:
            append_detector_and_evaluation(*segment)
        return ExperimentPlan(tuple(stages))

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
                f'select_samples_round_{token}',
                'ada_fnp.select_samples',
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
        append_detector_and_evaluation(milestone, end_iteration)
    return ExperimentPlan(tuple(stages))
