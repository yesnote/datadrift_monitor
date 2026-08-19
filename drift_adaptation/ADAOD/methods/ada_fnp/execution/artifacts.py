'''ADA-FNP checkpoint lookup with content verification.'''

from pathlib import Path
from typing import Optional

from methods.common.contracts import ArtifactRef
from methods.common.engine.context import ExecutionContext


def completed_checkpoint(
    context: ExecutionContext, artifact_type: str
) -> Optional[Path]:
    for completed in reversed(context.state_store.load().completed_stages):
        artifact = completed.get('result', {}).get('checkpoint_artifact')
        if artifact and artifact.get('artifact_type') == artifact_type:
            reference = ArtifactRef(**artifact)
            context.artifact_store.verify(reference)
            return context.run_directory / reference.relative_path
    return None
