'''Method-independent serial stage runner.'''

from typing import Callable, Dict, Mapping

from methods.common.contracts import ExperimentPlan, StageSpec

from .state import RunStateStore


StageExecutor = Callable[[StageSpec], Mapping]


class StageExecutorRegistry:
    def __init__(self) -> None:
        self._executors: Dict[str, StageExecutor] = {}

    def register(self, key: str, executor: StageExecutor) -> None:
        if key in self._executors:
            raise ValueError(f'duplicate stage executor: {key}')
        self._executors[key] = executor

    def resolve(self, key: str) -> StageExecutor:
        if key not in self._executors:
            raise KeyError(f'unregistered stage executor: {key}')
        return self._executors[key]


class StageRunner:
    def __init__(self, registry: StageExecutorRegistry, state_store: RunStateStore):
        self.registry = registry
        self.state_store = state_store

    def run(self, plan: ExperimentPlan) -> None:
        state = self.state_store.load()
        completed = {item['stage_id'] for item in state.completed_stages}
        for stage in plan.stages:
            if stage.stage_id in completed:
                continue
            state.status = 'running'
            state.active_stage_id = stage.stage_id
            state.failed_stage_id = None
            self.state_store.save(state)
            try:
                result = dict(self.registry.resolve(stage.executor_key)(stage))
            except Exception:
                state.status = 'failed'
                state.failed_stage_id = stage.stage_id
                self.state_store.save(state)
                raise
            state.completed_stages.append({
                'stage_id': stage.stage_id,
                'executor_key': stage.executor_key,
                'result': result,
            })
            state.active_stage_id = None
            self.state_store.save(state)
        state.status = 'complete'
        self.state_store.save(state)
