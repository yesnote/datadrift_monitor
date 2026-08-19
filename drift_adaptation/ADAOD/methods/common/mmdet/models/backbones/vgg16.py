'''BatchNorm-free VGG16 backbone and PT Caffe checkpoint loading.'''

from collections import OrderedDict
from pathlib import Path
import re
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from mmengine.model import BaseModule

from methods.common.artifacts import sha256_file
from methods.common.external_assets import AssetVerificationError


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

_CAFFE_FEATURE_INDICES = (
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
_SHA256_PATTERN = re.compile(r'^[0-9a-fA-F]{64}$')


class VGG16Backbone(BaseModule):
    '''VGG16 convolutional trunk used by Faster R-CNN.

    The network contains ``conv1_1`` through ``conv5_3`` and only four max
    pooling layers.  Consequently, its single output has 512 channels and an
    output stride of 16.  Batch normalization and ``pool5`` are intentionally
    absent.

    Args:
        frozen_stages: Number of leading convolutional stages to freeze. PT
            freezes the first two stages for its C-to-F detector.
        pretrained_checkpoint: Local PT ``vgg16_caffe.pth`` path. The asset
            preparation stage must download this file before model creation.
        pretrained_sha256: Pinned checksum for ``pretrained_checkpoint``. The
            backbone verifies it again immediately before deserialization.
        init_cfg: Optional MMEngine initialization configuration. It cannot be
            combined with the explicit PT checkpoint path.
    '''

    out_channels = 512
    output_stride = 16

    def __init__(self,
                 frozen_stages: int = 2,
                 pretrained_checkpoint: Optional[str] = None,
                 pretrained_sha256: Optional[str] = None,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if frozen_stages < -1 or frozen_stages > 5:
            raise ValueError('frozen_stages must be between -1 and 5')
        if (pretrained_checkpoint is None) != (pretrained_sha256 is None):
            raise ValueError(
                'pretrained_checkpoint and pretrained_sha256 must be set together')
        if pretrained_checkpoint is not None and init_cfg is not None:
            raise ValueError(
                'explicit PT checkpoint loading cannot be combined with init_cfg')
        self.frozen_stages = frozen_stages
        self.pretrained_checkpoint = pretrained_checkpoint
        self.pretrained_sha256 = pretrained_sha256
        self.features, self.stage_names = self._build_features()
        if pretrained_checkpoint is not None:
            self._load_pt_caffe_checkpoint(
                pretrained_checkpoint, str(pretrained_sha256))
        self._freeze_stages()

    def _load_pt_caffe_checkpoint(self, checkpoint: str, sha256: str) -> None:
        if not _SHA256_PATTERN.fullmatch(sha256):
            raise ValueError(
                'pretrained_sha256 must contain exactly 64 hexadecimal characters')
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                'PT VGG16 Caffe checkpoint does not exist: {}'.format(
                    checkpoint_path))
        expected_digest = sha256.lower()
        actual_digest = sha256_file(checkpoint_path)
        if actual_digest != expected_digest:
            raise AssetVerificationError(
                'PT VGG16 Caffe checkpoint SHA-256 mismatch: expected {}, got {}'.format(
                    expected_digest, actual_digest))
        source_state_dict = torch.load(
            checkpoint_path, map_location='cpu', weights_only=True)
        if not isinstance(source_state_dict, Mapping):
            raise TypeError('PT VGG16 Caffe checkpoint must contain a state dict')
        mapped = map_caffe_vgg16_state_dict(source_state_dict)
        self.load_state_dict(mapped, strict=True)

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


def _required_caffe_conv_shapes() -> Dict[str, Tuple[int, ...]]:
    shapes = {}
    for index, (_, in_channels, out_channels) in zip(
            _CAFFE_FEATURE_INDICES, _CONV_SPECS):
        shapes['features.{}.weight'.format(index)] = (
            out_channels, in_channels, 3, 3)
        shapes['features.{}.bias'.format(index)] = (out_channels, )
    return shapes


def _validate_source_tensor(key: str, tensor: Tensor,
                            expected_shape: Sequence[int]) -> None:
    if tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(
            'unexpected shape for {!r}: expected {}, got {}'.format(
                key, tuple(expected_shape), tuple(tensor.shape)))


def map_caffe_vgg16_state_dict(
        source_state_dict: Mapping[str, Tensor],
        backbone_prefix: str = '') -> Dict[str, Tensor]:
    '''Map PT's Caffe VGG16 checkpoint to the ADAOD convolutional trunk.

    PT reads the 26 torchvision-style ``features.*`` convolution tensors from
    ``vgg16_caffe.pth`` and intentionally leaves all detector FC layers newly
    initialized. Returned tensors reference the source mapping without copies.
    '''

    expected_shapes = _required_caffe_conv_shapes()
    missing = sorted(set(expected_shapes).difference(source_state_dict))
    if missing:
        raise KeyError('PT Caffe VGG16 state dict is missing: {}'.format(
            ', '.join(missing)))

    mapped = {}
    for index, (target_name, _, _) in zip(_CAFFE_FEATURE_INDICES,
                                          _CONV_SPECS):
        for suffix in ('weight', 'bias'):
            source_key = 'features.{}.{}'.format(index, suffix)
            _validate_source_tensor(source_key, source_state_dict[source_key],
                                    expected_shapes[source_key])
            target_key = '{}features.{}.{}'.format(
                backbone_prefix, target_name, suffix)
            mapped[target_key] = source_state_dict[source_key]
    return mapped
