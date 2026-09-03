"""Convert ALOD experiment outputs into TensorBoard event files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common.tensorboard_export import export_work_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert ALOD work_dirs metrics to TensorBoard events.',
    )
    parser.add_argument(
        '--work-dir',
        type=Path,
        default=Path('work_dirs'),
        help='Experiment output root. Defaults to work_dirs.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = export_work_dir(args.work_dir)
    if not summaries:
        raise SystemExit('No ALOD runs found under %s.' % args.work_dir)

    total_seeds = sum(summary.seed_count for summary in summaries)
    total_rounds = sum(summary.round_count for summary in summaries)
    total_points = sum(
        summary.training_points
        + summary.active_learning_points
        + summary.aggregate_points
        for summary in summaries
    )
    print(
        'TensorBoard export complete: runs=%d seeds=%d rounds=%d scalar_points=%d'
        % (len(summaries), total_seeds, total_rounds, total_points)
    )
    print('Launch with: tensorboard --logdir %s' % args.work_dir)


if __name__ == '__main__':
    main()
