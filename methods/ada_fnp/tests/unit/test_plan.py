from methods.ada_fnp.configs.default import get_config
from methods.ada_fnp.plan import build_plan
from methods.common.data.pool import split_budget


def test_round_budget_uses_earliest_remainder():
    assert split_budget(149) == (30, 30, 30, 30, 29)


def test_default_plan_is_complete():
    plan = build_plan(get_config())
    assert len(plan.stages) == 29
    assert plan.stages[0].stage_id == 'prepare_pretrained'
    assert plan.stages[0].executor_key == 'common.prepare_pretrained'
    assert plan.stages[0].payload == {'detector': 'faster-rcnn-vgg16'}
    assert plan.stages[1].stage_id == 'prepare_datasets'
    assert plan.stages[-1].stage_id == 'evaluate_final_teacher'
    budgets = [
        stage.payload['budget'] for stage in plan.stages
        if stage.executor_key == 'common.select'
    ]
    assert budgets == [6, 6, 6, 6, 6]


def test_five_percent_budget_rounds_to_149():
    config = get_config()
    config['acquisition']['budget_percent'] = 5.0
    plan = build_plan(config)
    budgets = [
        stage.payload['budget'] for stage in plan.stages
        if stage.executor_key == 'common.select'
    ]
    assert budgets == [30, 30, 30, 30, 29]
