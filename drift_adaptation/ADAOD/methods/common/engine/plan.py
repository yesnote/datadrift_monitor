'''Plan validation independent of concrete methods.'''

from methods.common.contracts import ExperimentPlan


def validate_plan(plan: ExperimentPlan) -> None:
    if not plan.stages:
        raise ValueError('experiment plan must contain at least one stage')
    evaluate_positions = [
        index for index, stage in enumerate(plan.stages)
        if stage.executor_key == 'common.evaluate'
    ]
    if evaluate_positions and evaluate_positions[-1] != len(plan.stages) - 1:
        raise ValueError('final evaluation must be the last stage')
