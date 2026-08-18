'''BN-free VGG16 Faster R-CNN model for MMDetection 3.3.'''

from configs._base_.models.faster_rcnn_vgg16_factory import (
    build_faster_rcnn_vgg16 as _build_faster_rcnn_vgg16,
)


custom_imports = dict(
    imports=['methods.common.mmdet.registration'],
    allow_failed_imports=False,
)

# This shared detector is intentionally branch-agnostic.  ADA methods wrap a
# fresh factory result and own any multi-branch preprocessing themselves.
model = _build_faster_rcnn_vgg16()
