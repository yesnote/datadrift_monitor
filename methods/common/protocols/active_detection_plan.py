'''Plan builder for methods evaluated with the ADA-FNP five-round protocol.'''

from typing import Callable, List, Mapping, Sequence

from methods.common.acquisition.budget import resolve_percentage_budget
from methods.common.contracts import ExperimentPlan, StageSpec
from methods.common.data.pool import split_budget
from methods.common.protocols.ada_fnp_detection import (
    ACQUISITION_MILESTONES,
    ADAPTATION_DETECTOR_SEGMENTS,
    INITIAL_DETECTOR_SEGMENT,
)


RoundStageFactory = Callable[
    [int, int, Mapping],
    Sequence[StageSpec],
]


def build_active_detection_plan(
    config: Mapping,
    *,
    executor_prefix: str,
    round_stage_factory: RoundStageFactory,
    evaluation_executor: str,
) -> ExperimentPlan:
    milestones = tuple(config['training']['acquisition_milestones'])
    if milestones != ACQUISITION_MILESTONES:
        raise ValueError('the ADA-FNP protocol requires 5k through 25k rounds')
    total_budget = resolve_percentage_budget(config)
    budgets = split_budget(total_budget, len(milestones))
    detector_executor = '{}.train_detector'.format(executor_prefix)
    stages: List[StageSpec] = [
        StageSpec(
            'prepare_vgg16_caffe_weights',
            '{}.prepare_vgg16_caffe_weights'.format(executor_prefix),
        ),
        StageSpec(
            'prepare_cityscapes_to_foggy',
            '{}.prepare_cityscapes_to_foggy'.format(executor_prefix),
        ),
    ]

    def append_detector_and_evaluation(start: int, end: int) -> None:
        stages.append(StageSpec(
            'train_detector_{:05d}_{:05d}'.format(start, end),
            detector_executor,
            {'start_iteration': start, 'end_iteration': end},
        ))
        stages.append(StageSpec(
            'evaluate_detector_{:05d}'.format(end),
            evaluation_executor,
            {
                'iteration': end,
                'metric': 'AP50',
                'detector_executor_key': detector_executor,
            },
        ))

    append_detector_and_evaluation(*INITIAL_DETECTOR_SEGMENT)
    if total_budget == 0:
        for segment in ADAPTATION_DETECTOR_SEGMENTS:
            append_detector_and_evaluation(*segment)
        return ExperimentPlan(tuple(stages))
    for index, (milestone, budget) in enumerate(zip(milestones, budgets), 1):
        stages.extend(round_stage_factory(index, budget, config))
        end_iteration = (
            milestones[index]
            if index < len(milestones)
            else config['training']['max_iterations']
        )
        append_detector_and_evaluation(milestone, end_iteration)
    return ExperimentPlan(tuple(stages))
