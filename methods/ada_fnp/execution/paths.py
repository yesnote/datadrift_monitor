'''Repository-contained paths shared by ADA-FNP executors.'''

from pathlib import Path, PurePosixPath

from methods.common.engine.context import ExecutionContext


def _round_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('round index must be a non-negative integer')
    return value


def dataset_cache_directory(context: ExecutionContext) -> Path:
    configured = PurePosixPath(
        context.config['runtime']['dataset_cache_root']
    )
    if configured.is_absolute() or '..' in configured.parts:
        raise ValueError('dataset cache root must be repository-relative')
    path = context.repository_root.joinpath(
        *configured.parts, context.config['scenario']
    ).resolve()
    try:
        path.relative_to(context.repository_root)
    except ValueError as error:
        raise ValueError('dataset cache directory escapes the repository') from error
    return path


def pool_state_path(context: ExecutionContext, round_index: int) -> Path:
    return (
        context.run_directory / 'artifacts' / 'pool' /
        'round_{:02d}.json'.format(_round_index(round_index))
    )


def target_labeled_manifest_path(
    context: ExecutionContext, round_index: int
) -> Path:
    return (
        context.run_directory / 'datasets' /
        'target_train_labeled_round_{:02d}.json'.format(
            _round_index(round_index)
        )
    )


def target_unlabeled_manifest_path(
    context: ExecutionContext, round_index: int
) -> Path:
    return (
        context.run_directory / 'datasets' /
        'target_train_unlabeled_pool_{:02d}.json'.format(
            _round_index(round_index)
        )
    )
