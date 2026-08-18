'''ADA-FNP checkpoint lookup with content verification.'''

from pathlib import Path
from typing import Optional

from methods.common.artifacts import sha256_file
from methods.common.engine.context import ExecutionContext


def completed_checkpoint(
    context: ExecutionContext, artifact_type: str
) -> Optional[Path]:
    for completed in reversed(context.state_store.load().completed_stages):
        artifact = completed.get('result', {}).get('checkpoint_artifact')
        if artifact and artifact.get('artifact_type') == artifact_type:
            path = context.run_directory / artifact['relative_path']
            if not path.is_file():
                raise FileNotFoundError(
                    'completed checkpoint is missing: {!s}'.format(path)
                )
            if sha256_file(path) != artifact['sha256']:
                raise RuntimeError('completed checkpoint failed SHA256 verification')
            return path
    return None
