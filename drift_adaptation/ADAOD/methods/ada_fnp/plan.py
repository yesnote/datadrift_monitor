'''Serial five-round ADA-FNP experiment plan.'''

from typing import List, Mapping

from methods.ada_fnp.phases import FNPM_MILESTONES
from methods.common.contracts import ExperimentPlan, StageSpec
from methods.common.data.pool import split_budget


def resolve_total_budget(config: Mapping) -> int:
    acquisition = config['acquisition']
    if 'total_budget' in acquisition:
        return int(acquisition['total_budget'])
    target_size = int(config['dataset']['target']['expected_train_images'])
    percentage = float(acquisition['budget_percent'])
    if not 0.0 <= percentage <= 100.0:
        raise ValueError('budget_percent must be between 0 and 100')
    return int(target_size * percentage / 100.0 + 0.5)


def build_plan(config: Mapping) -> ExperimentPlan:
    milestones = tuple(config['training']['acquisition_milestones'])
    if milestones != FNPM_MILESTONES:
        raise ValueError('ADA-FNP requires acquisition at 5k through 25k')
    budgets = split_budget(resolve_total_budget(config), len(milestones))
    stages: List[StageSpec] = [
        StageSpec(
            'prepare_pretrained',
            'common.prepare_pretrained',
            {'detector': config['detector']['name']},
        ),
        StageSpec(
            'prepare_datasets',
            'common.prepare_datasets',
            {'scenario': config['scenario']},
        ),
        StageSpec(
            'train_detector_00000_05000',
            'ada_fnp.train_detector',
            {
                'start_iteration': 0,
                'end_iteration': 5000,
            },
        ),
    ]
    for index, (milestone, budget) in enumerate(zip(milestones, budgets), 1):
        token = f'{index:02d}'
        stages.extend((
            StageSpec(
                f'train_fnpm_round_{token}',
                'ada_fnp.train_fnpm',
                {'round': index, 'detector_iteration': milestone,
                 'iterations': config['fnpm']['iterations_per_round']},
            ),
            StageSpec(
                f'score_pool_round_{token}',
                'ada_fnp.score_pool',
                {'round': index, 'detector_iteration': milestone},
            ),
            StageSpec(
                f'select_round_{token}', 'common.select',
                {'round': index, 'budget': budget},
            ),
            StageSpec(
                f'reveal_round_{token}', 'common.reveal_annotations',
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
        'evaluate_final_teacher', 'common.evaluate',
        {'model': 'teacher', 'metric': 'AP50'},
    ))
    return ExperimentPlan(tuple(stages))
