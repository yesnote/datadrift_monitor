'''Repository-contained paths shared by ADA-FNP executors.'''

from pathlib import Path, PurePosixPath

from methods.common.engine.context import ExecutionContext


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
