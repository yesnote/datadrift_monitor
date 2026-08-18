'''BatchNorm-free VGG16 backbone and torchvision checkpoint mapping.'''

from collections import OrderedDict
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

try:
    from mmengine.model import BaseModule
except ModuleNotFoundError as exc:
    if exc.name != 'mmengine':
        raise

    class BaseModule(nn.Module):  # type: ignore[no-redef]
        '''Small fallback used only by PyTorch-only unit tests.'''

        def __init__(self, init_cfg=None) -> None:
            super().__init__()
            self.init_cfg = init_cfg


_CONV_SPECS = (
    ('conv1_1', 3, 64),
    ('conv1_2', 64, 64),
    ('conv2_1', 64, 128),
    ('conv2_2', 128, 128),
    ('conv3_1', 128, 256),
    ('conv3_2', 256, 256),
    ('conv3_3', 256, 256),
    ('conv4_1', 256, 512),
    ('conv4_2', 512, 512),
    ('conv4_3', 512, 512),
    ('conv5_1', 512, 512),
    ('conv5_2', 512, 512),
    ('conv5_3', 512, 512),
)

_TORCHVISION_CONV_INDICES = (
    0,
    2,
    5,
    7,
    10,
    12,
    14,
    17,
    19,
    21,
    24,
    26,
    28,
)

_STAGE_CONVS = (2, 2, 3, 3, 3)


class VGG16Backbone(BaseModule):
    '''VGG16 convolutional trunk used by Faster R-CNN.

    The network contains ``conv1_1`` through ``conv5_3`` and only four max
    pooling layers.  Consequently, its single output has 512 channels and an
    output stride of 16.  Batch normalization and ``pool5`` are intentionally
    absent.

    Args:
        frozen_stages: Number of leading convolutional stages to freeze.
            ``-1`` and ``0`` both leave every stage trainable.  Values 1--5
            freeze that many stages.
        init_cfg: Optional MMEngine initialization configuration.  Torchvision
            VGG16 checkpoints require :func:`map_torchvision_vgg16_state_dict`
            because their state-dict keys do not match this module directly.
    '''

    out_channels = 512
    output_stride = 16

    def __init__(self,
                 frozen_stages: int = -1,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if frozen_stages < -1 or frozen_stages > 5:
            raise ValueError('frozen_stages must be between -1 and 5')
        self.frozen_stages = frozen_stages
        self.features, self.stage_names = self._build_features()
        self._freeze_stages()

    @staticmethod
    def _build_features() -> Tuple[nn.Sequential, Tuple[Tuple[str, ...], ...]]:
        layers = OrderedDict()
        stages = []
        spec_offset = 0
        for stage_index, num_convs in enumerate(_STAGE_CONVS, start=1):
            stage_names = []
            for conv_offset in range(num_convs):
                name, in_channels, out_channels = _CONV_SPECS[
                    spec_offset + conv_offset]
                relu_name = name.replace('conv', 'relu', 1)
                layers[name] = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=1)
                layers[relu_name] = nn.ReLU(inplace=True)
                stage_names.extend((name, relu_name))
            spec_offset += num_convs
            if stage_index < 5:
                pool_name = 'pool{}'.format(stage_index)
                layers[pool_name] = nn.MaxPool2d(kernel_size=2, stride=2)
                stage_names.append(pool_name)
            stages.append(tuple(stage_names))
        return nn.Sequential(layers), tuple(stages)

    def _freeze_stages(self) -> None:
        for stage_names in self.stage_names[:max(self.frozen_stages, 0)]:
            for name in stage_names:
                module = self.features._modules[name]
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def train(self, mode: bool = True) -> 'VGG16Backbone':
        super().train(mode)
        self._freeze_stages()
        return self

    def forward(self, inputs: Tensor) -> Tuple[Tensor, ...]:
        return (self.features(inputs), )


def _required_torchvision_shapes() -> Dict[str, Tuple[int, ...]]:
    shapes = {}
    for index, (_, in_channels, out_channels) in zip(
            _TORCHVISION_CONV_INDICES, _CONV_SPECS):
        shapes['features.{}.weight'.format(index)] = (
            out_channels, in_channels, 3, 3)
        shapes['features.{}.bias'.format(index)] = (out_channels, )
    shapes.update({
        'classifier.0.weight': (4096, 512 * 7 * 7),
        'classifier.0.bias': (4096, ),
        'classifier.3.weight': (4096, 4096),
        'classifier.3.bias': (4096, ),
    })
    return shapes


def _validate_source_tensor(key: str, tensor: Tensor,
                            expected_shape: Sequence[int]) -> None:
    if tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(
            'unexpected shape for {!r}: expected {}, got {}'.format(
                key, tuple(expected_shape), tuple(tensor.shape)))


def map_torchvision_vgg16_state_dict(
        source_state_dict: Mapping[str, Tensor],
        backbone_prefix: str = '',
        bbox_head_prefix: str = '') -> Dict[str, Tensor]:
    '''Map torchvision VGG16 ImageNet weights to the ADAOD detector modules.

    The ImageNet classifier's final 1000-class layer is deliberately ignored.
    Returned tensors are references to the source tensors; this helper does not
    duplicate the approximately 500 MB VGG16 state dictionary.

    Args:
        source_state_dict: A torchvision VGG16 state dictionary.
        backbone_prefix: Prefix for backbone target keys, for example
            ``'backbone.'``.
        bbox_head_prefix: Prefix for bbox-head target keys, for example
            ``'roi_head.bbox_head.'``.
    '''
    expected_shapes = _required_torchvision_shapes()
    missing = sorted(set(expected_shapes).difference(source_state_dict))
    if missing:
        raise KeyError('torchvision VGG16 state dict is missing: {}'.format(
            ', '.join(missing)))

    mapped = {}
    for index, (target_name, _, _) in zip(_TORCHVISION_CONV_INDICES,
                                          _CONV_SPECS):
        for suffix in ('weight', 'bias'):
            source_key = 'features.{}.{}'.format(index, suffix)
            _validate_source_tensor(source_key, source_state_dict[source_key],
                                    expected_shapes[source_key])
            target_key = '{}features.{}.{}'.format(
                backbone_prefix, target_name, suffix)
            mapped[target_key] = source_state_dict[source_key]

    classifier_mapping = {
        'classifier.0.weight': 'shared_fcs.0.weight',
        'classifier.0.bias': 'shared_fcs.0.bias',
        'classifier.3.weight': 'shared_fcs.1.weight',
        'classifier.3.bias': 'shared_fcs.1.bias',
    }
    for source_key, target_name in classifier_mapping.items():
        _validate_source_tensor(source_key, source_state_dict[source_key],
                                expected_shapes[source_key])
        mapped['{}{}'.format(bbox_head_prefix,
                             target_name)] = source_state_dict[source_key]
    return mapped
