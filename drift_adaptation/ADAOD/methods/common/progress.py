'''One reusable terminal progress line for an ADAOD process.'''

from __future__ import annotations

import math
import sys
from typing import Optional, TextIO

from tqdm import tqdm


class ProgressReporter:
    '''Render serial ADAOD work through one reusable tqdm instance.'''

    def __init__(
        self,
        enabled: bool = True,
        *,
        stream: Optional[TextIO] = None,
    ) -> None:
        self._stream = stream or sys.stderr
        self._enabled = bool(enabled and self._stream.isatty())
        self._bar = None
        self._stage_name = ''

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _ensure_bar(self):
        if not self._enabled:
            return None
        if self._bar is None:
            self._bar = tqdm(
                total=None,
                file=self._stream,
                ascii=True,
                dynamic_ncols=True,
                leave=False,
                mininterval=0.2,
                unit='item',
            )
        return self._bar

    def start_stage(
        self,
        index: int,
        total: int,
        name: str,
    ) -> None:
        if not 1 <= index <= total:
            raise ValueError('stage index must be within the plan')
        name = str(name).strip()
        if not name:
            raise ValueError('stage progress name must not be empty')
        self._stage_name = '[{:02d}/{:02d}] {}'.format(index, total, name)
        bar = self._ensure_bar()
        if bar is None:
            return
        bar.reset()
        bar.total = None
        bar.unit = 'stage'
        bar.set_description_str(self._stage_name, refresh=True)
        bar.set_postfix_str('', refresh=False)

    def start_task(
        self,
        total: Optional[int],
        unit: str,
        *,
        initial: int = 0,
    ) -> None:
        if total is not None:
            if isinstance(total, bool) or not isinstance(total, int):
                raise TypeError('progress total must be an integer or None')
            if total < 0:
                raise ValueError('progress total must not be negative')
        if isinstance(initial, bool) or not isinstance(initial, int):
            raise TypeError('initial progress must be an integer')
        if initial < 0 or (total is not None and initial > total):
            raise ValueError('initial progress is outside the task range')
        unit = str(unit).strip()
        if not unit:
            raise ValueError('progress unit must not be empty')
        bar = self._ensure_bar()
        if bar is None:
            return
        bar.reset()
        bar.total = total
        bar.n = initial
        bar.unit = unit
        if self._stage_name:
            bar.set_description_str(self._stage_name, refresh=False)
        bar.set_postfix_str('', refresh=False)
        bar.refresh()

    @staticmethod
    def _loss_text(loss: object) -> str:
        value = float(loss)
        if not math.isfinite(value):
            return str(value)
        return '{:.4f}'.format(value)

    def set_completed(
        self,
        value: int,
        *,
        loss: Optional[object] = None,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError('completed progress must be an integer')
        bar = self._ensure_bar()
        if bar is None:
            return
        if value < bar.n:
            raise ValueError('progress must not move backwards')
        if bar.total is not None and value > bar.total:
            raise ValueError('progress exceeds the task total')
        if loss is not None:
            bar.set_postfix_str(
                'loss={}'.format(self._loss_text(loss)),
                refresh=False,
            )
        bar.update(value - bar.n)

    def advance(
        self,
        count: int = 1,
        *,
        loss: Optional[object] = None,
    ) -> None:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError('progress increment must be an integer')
        if count < 0:
            raise ValueError('progress increment must not be negative')
        bar = self._ensure_bar()
        if bar is None:
            return
        self.set_completed(bar.n + count, loss=loss)

    def finish_stage(self) -> None:
        if self._bar is not None:
            self._bar.set_postfix_str('', refresh=False)
            self._bar.clear()

    def fail_stage(self) -> None:
        if self._bar is not None:
            self._bar.clear()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class NullProgressReporter(ProgressReporter):
    '''No-output reporter used outside an interactive rank-zero process.'''

    def __init__(self) -> None:
        super().__init__(enabled=False)
