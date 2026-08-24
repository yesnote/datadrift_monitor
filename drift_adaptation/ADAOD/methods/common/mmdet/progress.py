'''MMEngine adapters for compact ADAOD terminal progress.'''

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence

from mmengine.hooks import Hook
from mmengine.runner import Runner

from methods.common.progress import ProgressReporter


class AdaodConsoleQuietRunner(Runner):
    '''Keep MMEngine detail in files while showing only errors in the console.'''

    def build_logger(
        self,
        log_level: str = 'INFO',
        log_file: Optional[str] = None,
        **kwargs,
    ):
        logger = super().build_logger(
            log_level=log_level,
            log_file=log_file,
            **kwargs,
        )
        for handler in logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
            ):
                handler.setLevel(logging.ERROR)
        return logger


class TqdmProgressHook(Hook):
    '''Update the process-wide tqdm line after completed train/test work.'''

    priority = 'LOWEST'

    def __init__(
        self,
        reporter: ProgressReporter,
        *,
        task_total: Optional[int] = None,
        task_unit: str = 'iter',
        required_keys: Sequence[str] = (),
    ) -> None:
        self.reporter = reporter
        self.task_total = task_total
        self.task_unit = task_unit
        self.required_keys = tuple(required_keys)
        self._validated_train_outputs = False

    def before_train(self, runner: Runner) -> None:
        if self.task_total is None:
            raise ValueError('training progress requires an explicit task total')
        self.reporter.start_task(self.task_total, self.task_unit)

    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch=None,
        outputs: Optional[Mapping] = None,
    ) -> None:
        del runner, batch_idx, data_batch
        if outputs is None:
            raise RuntimeError('MMEngine train iteration returned no log outputs')
        if not self._validated_train_outputs:
            missing = sorted(set(self.required_keys) - set(outputs))
            if missing:
                raise KeyError(
                    'training log outputs are missing required metrics: {}'.format(
                        ', '.join(missing)
                    )
                )
            if 'loss' not in outputs:
                raise KeyError('training log outputs are missing total loss')
            self._validated_train_outputs = True
        self.reporter.advance(1, loss=outputs['loss'])

    def before_test(self, runner: Runner) -> None:
        total = self.task_total
        if total is None:
            total = len(runner.test_dataloader)
        self.reporter.start_task(total, self.task_unit)

    def after_test_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        del runner, batch_idx, data_batch, outputs
        self.reporter.advance(1)
