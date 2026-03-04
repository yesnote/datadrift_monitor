_base_ = '/sise/home/omerhof/fujitsu-explainability-wp2-internal2/OD_models/mmdetection/configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20)))

dataset_type = 'COCODataset'

# Modify dataset related settings
classes = ('Agnesi Polenta', 'Almond Milk','Snyders','Calvin Klein','Dr Pepper','Flour','Groats','Jack Daniels',
           'Nespresso','Oil','Paco Rabanne','Pixel4','Samsung_s20','Greek Olives','Curry Spice','Chablis Wine',
           'Lindor','Piling Sabon','Tea','Versace',)
data = dict(
    train=dict(
        img_prefix='/dt/shabtaia/dt-fujitsu-explainability/super_store/coco format/train/',
        classes=classes,
        ann_file='/dt/shabtaia/dt-fujitsu-explainability/super_store/coco format/train_ms_coco_format.json'),
    val=dict(
        img_prefix='/dt/shabtaia/dt-fujitsu-explainability/super_store/coco format/test/',
        classes=classes,
        ann_file='/dt/shabtaia/dt-fujitsu-explainability/super_store/coco format/test_ms_coco_format.json'),
    test=dict(
        img_prefix='dt/shabtaia/dt-fujitsu-explainability/super_store/coco format/test/',
        classes=classes,
        ann_file='/dt/shabtaia/dt-fujitsu-explainability/super_store/coco format/test_ms_coco_format.json'))


# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
