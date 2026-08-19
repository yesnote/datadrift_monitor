'''Method-independent serial stage runner.'''

from typing import Callable, Dict, Mapping

from methods.common.contracts import ArtifactRef, ExperimentPlan, StageSpec

from .state import RunStateStore
from .context import ExecutionContext


StageExecutor = Callable[[StageSpec, ExecutionContext], Mapping]


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
    def __init__(
        self,
        registry: StageExecutorRegistry,
        state_store: RunStateStore,
        context: ExecutionContext,
    ):
        self.registry = registry
        self.state_store = state_store
        self.context = context

    def _execute(self, stage: StageSpec) -> Mapping:
        executor = self.registry.resolve(stage.executor_key)
        return executor(stage, self.context)

    def _verify_result_artifacts(self, value) -> None:
        if isinstance(value, Mapping):
            artifact_fields = {
                'artifact_id', 'artifact_type', 'schema_version',
                'producer_stage_id', 'relative_path', 'sha256',
            }
            if set(value) == artifact_fields:
                self.context.artifact_store.verify(ArtifactRef(**value))
                return
            for nested in value.values():
                self._verify_result_artifacts(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                self._verify_result_artifacts(nested)

    def run(self, plan: ExperimentPlan) -> None:
        state = self.state_store.load()
        completed_entries = {
            item['stage_id']: item for item in state.completed_stages
        }
        completed = set(completed_entries)
        for stage in plan.stages:
            if stage.stage_id in completed:
                entry = completed_entries[stage.stage_id]
                if entry['executor_key'] != stage.executor_key:
                    raise RuntimeError(
                        'completed stage executor differs from the current plan'
                    )
                self._verify_result_artifacts(entry['result'])
                continue
            state.status = 'running'
            state.active_stage_id = stage.stage_id
            state.failed_stage_id = None
            self.state_store.save(state)
            try:
                result = dict(self._execute(stage))
            except Exception:
                state = self.state_store.load()
                state.status = 'failed'
                state.failed_stage_id = stage.stage_id
                self.state_store.save(state)
                raise
            state = self.state_store.load()
            state.completed_stages.append({
                'stage_id': stage.stage_id,
                'executor_key': stage.executor_key,
                'result': result,
            })
            state.active_stage_id = None
            self.state_store.save(state)
        state.status = 'complete'
        state.active_stage_id = None
        state.failed_stage_id = None
        self.state_store.save(state)
