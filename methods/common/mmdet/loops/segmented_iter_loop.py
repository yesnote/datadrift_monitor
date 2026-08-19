'''Iteration loop for globally numbered, independently sampled segments.'''

from typing import Any, Iterator

from mmengine.runner.loops import IterBasedTrainLoop


class _LogicalContinuationIterator:
    '''Consume MMEngine's logical offset without advancing a new dataloader.'''

    def __init__(self, iterator: Iterator[Any], logical_steps: int) -> None:
        self.iterator = iterator
        self.logical_steps = int(logical_steps)

    def __iter__(self):
        return self

    def __next__(self):
        if self.logical_steps > 0:
            self.logical_steps -= 1
            return None
        return next(self.iterator)


class ADAODSegmentedIterBasedTrainLoop(IterBasedTrainLoop):
    '''Continue global iteration state without skipping a new stage dataset.'''

    def run(self):
        if self._iter <= 0:
            return super().run()
        iterator = self.dataloader_iterator
        self.dataloader_iterator = _LogicalContinuationIterator(
            iterator,
            self._iter,
        )
        try:
            return super().run()
        finally:
            self.dataloader_iterator = iterator
