'''Fail-fast validation for the candidate MMDetection 3.3 environment.'''

import argparse
import importlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

EXPECTED_VERSIONS = {
    'torch': '2.0.1',
    'torchvision': '0.15.2',
    'mmcv': '2.1.0',
    'mmengine': '0.10.5',
    'mmdet': '3.3.0',
}
EXPECTED_CUDA = '11.8'


class EnvironmentCheckError(RuntimeError):
    pass


def _base_version(raw_version: str) -> str:
    return raw_version.split('+', 1)[0]


def _import_exact(name: str):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise EnvironmentCheckError(
            'cannot import {}: {}. See requirements/README.md.'.format(
                name, exc)) from exc
    version = _base_version(module.__version__)
    expected = EXPECTED_VERSIONS[name]
    if version != expected:
        raise EnvironmentCheckError(
            '{} {} is installed; expected {}'.format(name, version, expected))
    print('[ok] {} {}'.format(name, module.__version__))
    return module


def _check_local_mmdet(mmdet_module) -> None:
    expected_root = (REPOSITORY_ROOT / 'mmdet').resolve()
    module_path = Path(mmdet_module.__file__).resolve()
    try:
        module_path.relative_to(expected_root)
    except ValueError as exc:
        raise EnvironmentCheckError(
            'mmdet resolves to {!s}, not the repository-local {!s}'.format(
                module_path, expected_root)) from exc
    print('[ok] repository-local mmdet: {}'.format(module_path))


def _check_numpy() -> None:
    try:
        numpy = importlib.import_module('numpy')
    except Exception as exc:
        raise EnvironmentCheckError(
            'cannot import numpy: {}'.format(exc)
        ) from exc
    major_version = int(numpy.__version__.split('.', 1)[0])
    if major_version >= 2:
        raise EnvironmentCheckError(
            'NumPy {} is installed; PyTorch 2.0.1 requires NumPy <2 for '
            'the compiled NumPy ABI'.format(numpy.__version__)
        )
    print('[ok] numpy {}'.format(numpy.__version__))


def _check_mmcv_ops(torch, allow_cpu: bool) -> None:
    try:
        from mmcv.ops import RoIAlign, nms
    except Exception as exc:
        raise EnvironmentCheckError(
            'cannot import compiled mmcv.ops (nms, roi_align): {}'.format(
                exc)) from exc

    if torch.cuda.is_available():
        device = torch.device('cuda')
        if torch.version.cuda != EXPECTED_CUDA:
            raise EnvironmentCheckError(
                'PyTorch CUDA runtime is {}; expected {}'.format(
                    torch.version.cuda, EXPECTED_CUDA))
        print('[ok] CUDA {} on {}'.format(torch.version.cuda,
                                          torch.cuda.get_device_name(0)))
    elif allow_cpu:
        device = torch.device('cpu')
        print('[warning] CUDA is unavailable; CPU checks do not satisfy the '
              'GPU acceptance gate')
    else:
        raise EnvironmentCheckError(
            'CUDA is unavailable. Use the CUDA 11.8 PyTorch build on the '
            'target GPU, or pass --allow-cpu for structural checks only.')

    boxes = torch.tensor(
        [[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 5.0, 5.0]], device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    _, keep = nms(boxes, scores, iou_threshold=0.5)
    if keep.numel() == 0:
        raise EnvironmentCheckError('MMCV NMS returned no boxes')

    features = torch.randn(
        1, 1, 8, 8, device=device, requires_grad=True)
    rois = torch.tensor([[0.0, 1.0, 1.0, 6.0, 6.0]], device=device)
    roi_align = RoIAlign(
        output_size=(2, 2),
        spatial_scale=1.0,
        sampling_ratio=0,
        pool_mode='avg',
        aligned=True)
    pooled = roi_align(features, rois)
    pooled.sum().backward()
    if features.grad is None or not torch.isfinite(features.grad).all():
        raise EnvironmentCheckError('MMCV RoIAlign backward is invalid')
    print('[ok] MMCV NMS and RoIAlign forward/backward on {}'.format(device))


def check_environment(allow_cpu: bool = False) -> None:
    if sys.version_info[:2] != (3, 9):
        raise EnvironmentCheckError(
            'Python {}.{} is active; expected Python 3.9'.format(
                sys.version_info.major, sys.version_info.minor))
    print('[ok] Python {}'.format(sys.version.split()[0]))

    torch = _import_exact('torch')
    _check_numpy()
    _import_exact('torchvision')
    _import_exact('mmengine')
    _import_exact('mmcv')
    mmdet = _import_exact('mmdet')
    _check_local_mmdet(mmdet)
    _check_mmcv_ops(torch, allow_cpu=allow_cpu)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--allow-cpu',
        action='store_true',
        help='run structural CPU checks without satisfying the GPU gate')
    args = parser.parse_args(argv)
    try:
        check_environment(allow_cpu=args.allow_cpu)
    except EnvironmentCheckError as exc:
        print('[failed] {}'.format(exc), file=sys.stderr)
        return 1
    print('[passed] candidate MMDetection environment')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
