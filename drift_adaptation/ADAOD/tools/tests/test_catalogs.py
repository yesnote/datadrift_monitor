from configs.catalog import (
    get_dataset,
    get_detector,
    get_runtime,
    list_datasets,
    list_detectors,
    list_runtimes,
)


EXPECTED_CLASSES = (
    'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle',
)


def test_c2f_dataset_catalog_contract():
    assert list_datasets() == ('cityscapes-to-foggy',)
    dataset = get_dataset('cityscapes-to-foggy')
    assert dataset['source']['split'] == 'train'
    assert dataset['source']['expected_images'] == 2975
    assert dataset['target']['train_split'] == 'train'
    assert dataset['target']['eval_split'] == 'val'
    assert dataset['target']['beta'] == 0.02
    assert dataset['target']['expected_train_images'] == 2975
    assert dataset['target']['expected_eval_images'] == 500
    assert dataset['target']['train_annotation_access'] == 'oracle_only'
    assert dataset['target']['eval_annotation_access'] == 'evaluator_only'
    assert tuple(dataset['classes']) == EXPECTED_CLASSES


def test_catalog_accessors_return_independent_values():
    dataset = get_dataset('cityscapes-to-foggy')
    dataset['source']['expected_images'] = 1
    detector = get_detector('faster-rcnn-vgg16')
    detector['capabilities'] = ()
    runtime = get_runtime('default')
    runtime['work_root'] = 'changed'
    assert get_dataset('cityscapes-to-foggy')['source']['expected_images'] == 2975
    assert get_detector('faster-rcnn-vgg16')['capabilities']
    assert get_runtime('default')['work_root'] == 'work_dirs'


def test_detector_and_runtime_catalogs_are_c2f_foundation_only():
    assert list_detectors() == ('faster-rcnn-vgg16',)
    assert list_runtimes() == ('default',)
    detector = get_detector('faster-rcnn-vgg16')
    assert detector['architecture'] == 'FasterRCNN'
    assert detector['backbone'] == 'VGG16'
    assert detector['batch_normalization'] is False
    assert detector['num_classes'] == 8
    runtime = get_runtime('default')
    assert runtime['deterministic'] is True
    assert runtime['cudnn_benchmark'] is False

