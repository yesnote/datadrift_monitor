'''AADA stages under the ADA-FNP five-round comparison protocol.'''

from methods.common.contracts import StageSpec
from methods.common.protocols.active_detection_plan import (
    build_active_detection_plan,
)


def _round_stages(index, budget, config):
    del config
    token = '{:02d}'.format(index)
    return (
        StageSpec(
            'score_unlabeled_pool_round_{}'.format(token),
            'aada.score_unlabeled_pool',
            {'round': index},
        ),
        StageSpec(
            'select_samples_round_{}'.format(token),
            'aada.select_samples',
            {'round': index, 'budget': budget},
        ),
        StageSpec(
            'reveal_selected_annotations_round_{}'.format(token),
            'aada.reveal_selected_annotations',
            {'round': index},
        ),
    )


def build_plan(config):
    return build_active_detection_plan(
        config,
        executor_prefix='aada',
        round_stage_factory=_round_stages,
        evaluation_executor='aada.evaluate_detector_checkpoint',
    )
