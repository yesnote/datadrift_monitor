from mmengine.runner.loops import IterBasedTrainLoop

from methods.common.mmdet.loops.segmented_iter_loop import (
    ADAODSegmentedIterBasedTrainLoop,
)


def test_segmented_loop_skips_logical_iterations_without_consuming_batches(
    monkeypatch,
):
    actual_iterator = iter(['first-stage-batch'])
    loop = object.__new__(ADAODSegmentedIterBasedTrainLoop)
    loop._iter = 3
    loop.dataloader_iterator = actual_iterator
    observed = []

    def fake_run(current_loop):
        for _ in range(current_loop._iter):
            observed.append(next(current_loop.dataloader_iterator))
        observed.append(next(current_loop.dataloader_iterator))
        return 'complete'

    monkeypatch.setattr(IterBasedTrainLoop, 'run', fake_run)

    assert loop.run() == 'complete'
    assert observed == [None, None, None, 'first-stage-batch']
    assert loop.dataloader_iterator is actual_iterator
