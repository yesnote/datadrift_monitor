# The new config inherits a base config to highlight the necessary modification
_base_ = '/sise/home/omerhof/fujitsu-explainability-wp2-internal2/OD_models/mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20)))
dataset_type = 'COCODataset'
data_root = '/dt/shabtaia/dt-fujitsu-explainability/super_store/coco format/'
# Modify dataset related settings
metainfo = {
'classes': ('Agnesi Polenta', 'Almond Milk','Snyders','Calvin Klein','Dr Pepper','Flour','Groats','Jack Daniels',
           'Nespresso','Oil','Paco Rabanne','Pixel4','Samsung_s20','Greek Olives','Curry Spice','Chablis Wine',
           'Lindor','Piling Sabon','Tea','Versace')
}
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_ms_coco_format.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test_ms_coco_format.json',
        data_prefix=dict(img='test/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test_ms_coco_format.json',
        data_prefix=dict(img='test/')))


# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'test_coco_format.json')
test_evaluator = dict(ann_file=data_root + 'test_coco_format.json')

runner = dict(type='EpochBasedRunner', max_epochs=20)

