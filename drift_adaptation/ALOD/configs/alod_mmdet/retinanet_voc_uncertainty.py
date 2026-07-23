_base_ = './retinanet_voc_base.py'

model = dict(
    type='ALRetinaNet',
    bbox_head=dict(
        type='RetinaHeadUncertainty',
    ),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=200),
)

data = dict(
    test=dict(
        type='ALCocoDataset',
        ann_file=None,
        img_prefix='data/VOC0712/images/',
        classes=CLASSES),
)
unlabeled_data = ''
