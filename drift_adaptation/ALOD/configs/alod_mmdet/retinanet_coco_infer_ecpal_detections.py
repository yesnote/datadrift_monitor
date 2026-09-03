_base_ = './retinanet_coco_base.py'

model = dict(
    type='ALRetinaNet',
    bbox_head=dict(
        type='RetinaHeadECPAL',
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.3,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=200,
        support_iou_thr=0.5,
        support_box_equal_tol=1e-6,
        miss_eps=1e-12),
)

data = dict(
    test=dict(
        type='ECPALFeatureCocoDataset',
        ann_file=None,
        img_prefix='data/coco/train2017/'),
)
