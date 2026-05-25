# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .batch_norm import FrozenBatchNorm2d
from .misc import Conv2d
from .misc import ConvTranspose2d
from .misc import BatchNorm2d
from .misc import interpolate
from .nms import nms, ml_nms
from .smooth_l1_loss import smooth_l1_loss
from .sigmoid_focal_loss import SigmoidFocalLoss
from .iou_loss import IOULoss
from .scale import Scale


class _UnavailableDeformableLayer(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Deformable FCOS layers require the original FCOS C++/CUDA extension. "
            "Use a non-DCN FCOS config or build the extension."
        )


class DFConv2d(_UnavailableDeformableLayer):
    pass


DeformConv = _UnavailableDeformableLayer
ModulatedDeformConv = _UnavailableDeformableLayer
ModulatedDeformConvPack = _UnavailableDeformableLayer
DeformRoIPooling = _UnavailableDeformableLayer
DeformRoIPoolingPack = _UnavailableDeformableLayer
ModulatedDeformRoIPoolingPack = _UnavailableDeformableLayer


def deform_conv(*args, **kwargs):
    raise NotImplementedError("deform_conv requires the original FCOS C++/CUDA extension.")


def modulated_deform_conv(*args, **kwargs):
    raise NotImplementedError("modulated_deform_conv requires the original FCOS C++/CUDA extension.")


def deform_roi_pooling(*args, **kwargs):
    raise NotImplementedError("deform_roi_pooling requires the original FCOS C++/CUDA extension.")


def roi_align(*args, **kwargs):
    raise NotImplementedError("roi_align is not copied for FCOS-only usage.")


class ROIAlign(_UnavailableDeformableLayer):
    pass


def roi_pool(*args, **kwargs):
    raise NotImplementedError("roi_pool is not copied for FCOS-only usage.")


class ROIPool(_UnavailableDeformableLayer):
    pass


__all__ = [
    "nms",
    "ml_nms",
    "roi_align",
    "ROIAlign",
    "roi_pool",
    "ROIPool",
    "smooth_l1_loss",
    "Conv2d",
    "DFConv2d",
    "ConvTranspose2d",
    "interpolate",
    "BatchNorm2d",
    "FrozenBatchNorm2d",
    "SigmoidFocalLoss",
    'deform_conv',
    'modulated_deform_conv',
    'DeformConv',
    'ModulatedDeformConv',
    'ModulatedDeformConvPack',
    'deform_roi_pooling',
    'DeformRoIPooling',
    'DeformRoIPoolingPack',
    'ModulatedDeformRoIPoolingPack',
    "IOULoss",
    "Scale"
]
