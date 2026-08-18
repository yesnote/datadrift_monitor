import json

import pytest

from methods.common.contracts import ExperimentPlan, StageSpec
from methods.common.engine.artifacts import ArtifactStore
from methods.common.engine.runner import StageExecutorRegistry, StageRunner
from methods.common.engine.state import RunStateStore


def test_artifact_store_detects_tampering(tmp_path):
    store = ArtifactStore(tmp_path)
    artifact = store.write_json('rounds/one.json', {'value': 1}, 'test', 'stage')
    store.verify(artifact)
    (tmp_path / artifact.relative_path).write_text('{}', encoding='utf-8')
    with pytest.raises(RuntimeError):
        store.verify(artifact)


def test_runner_resumes_after_completed_stage(tmp_path):
    calls = []
    fail_second = {'value': True}

    def first(stage):
        calls.append(stage.stage_id)
        return {'ok': True}

    def second(stage):
        calls.append(stage.stage_id)
        if fail_second['value']:
            fail_second['value'] = False
            raise RuntimeError('injected failure')
        return {'ok': True}

    registry = StageExecutorRegistry()
    registry.register('first', first)
    registry.register('second', second)
    state_store = RunStateStore(tmp_path / 'state.json')
    runner = StageRunner(registry, state_store)
    plan = ExperimentPlan((StageSpec('one', 'first'), StageSpec('two', 'second')))
    with pytest.raises(RuntimeError):
        runner.run(plan)
    runner.run(plan)
    assert calls == ['one', 'two', 'two']
    state = json.loads((tmp_path / 'state.json').read_text(encoding='utf-8'))
    assert state['status'] == 'complete'
    assert [item['stage_id'] for item in state['completed_stages']] == ['one', 'two']


def test_registry_rejects_duplicate_executor():
    registry = StageExecutorRegistry()
    registry.register('same', lambda stage: {})
    with pytest.raises(ValueError):
        registry.register('same', lambda stage: {})
