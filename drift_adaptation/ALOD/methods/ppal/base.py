from pathlib import Path

import numpy as np

from methods.common.coco_pool import image_ids, read_coco_json, write_coco_pool_split

COCO_CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
)

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
)


eps = 1e-10


class BaseALSampler(object):

    def __init__(
        self,
        n_sample_images,
        oracle_annotation_path,
        is_random,
        dataset_type='coco',
        **kwargs
    ):

        if dataset_type == 'coco':
            self.CLASSES = COCO_CLASSES
        elif dataset_type == 'voc':
            self.CLASSES = VOC_CLASSES
        else:
            raise NotImplementedError
        self.dataset_type = dataset_type

        self.n_images = n_sample_images
        self.is_random = is_random

        # read and store oracle annotations
        data = read_coco_json(Path(oracle_annotation_path))

        self.image_pool_size = len(data['images'])
        self.oracle_json = dict(data)
        self.oracle_data = dict()
        self.categories = data['categories']

        self.categories_dict = dict()
        self.class_id2name = dict()
        self.class_name2id = dict()

        self.valid_categories = []
        for c in self.categories:
            self.categories_dict[c['id']] = c['name']
            if c['name'] in self.CLASSES:
                self.class_name2id[c['name']] = c['id']
                self.class_id2name[c['id']] = c['name']
                self.valid_categories.append(c['id'])

        for img in data['images']:
            self.oracle_data[img['id']] = dict()
            self.oracle_data[img['id']]['image'] = img
            self.oracle_data[img['id']]['annotations'] = []

        for ann in data['annotations']:
            img_id = ann['image_id']
            if self.categories_dict[ann['category_id']] in self.CLASSES:
                self.oracle_data[img_id]['annotations'].append(ann)
        self.oracle_json['annotations'] = [
            ann for records in self.oracle_data.values()
            for ann in records['annotations']
        ]

        self.oracle_cate_prob = self.cate_prob_stat(input_json=None)

        self.round = 1  # the init round is the first round

        self.size_thr = 16
        self.ratio_thr = 5.

        # for logging
        self.oracle_path = oracle_annotation_path
        self.requires_result = True
        self.latest_labeled = None

    def cate_prob_stat(self, input_json=None):
        cate_freqs = dict()
        for cid in self.valid_categories:
            cate_freqs[cid] = 0.
        if input_json is None:
            for img_id in self.oracle_data.keys():
                for ann in self.oracle_data[img_id]['annotations']:
                    cate_freqs[ann['category_id']] += 1.
        else:
            data = read_coco_json(Path(input_json))
            for ann in data['annotations']:
                if ann['category_id'] in self.valid_categories:
                    cate_freqs[ann['category_id']] += 1.

        total = sum(cate_freqs.values())
        cate_probs = dict()
        for k, v in cate_freqs.items():
            cate_probs[k] = v / total
        return cate_probs

    def is_box_valid(self, box, img_size):
        # clip box and filter out outliers
        img_w, img_h = img_size
        x1, y1, w, h = box
        if (x1 > img_w) or (y1 > img_h):
            return False
        x2 = min(img_w, x1 + w)
        y2 = min(img_h, y1 + h)
        w = x2 - x1
        h = y2 - y1
        return (
            (np.sqrt(w * h) > self.size_thr)
            and (w / (h + eps) < self.ratio_thr)
            and (h / (w + eps) < self.ratio_thr)
        )

    def set_round(self, new_round):
        self.round = new_round

    def al_acquisition(self, result_json):
        pass

    def create_jsons(
        self,
        sampled_img_ids,
        unsampled_img_ids,
        last_labeled_json,
        out_label_path,
        out_unlabeled_path
    ):
        last_labeled_data = read_coco_json(Path(last_labeled_json))

        last_labeled_img_ids = image_ids(last_labeled_data)
        all_labeled_img_ids = last_labeled_img_ids + sampled_img_ids
        assert len(set(all_labeled_img_ids)) == len(last_labeled_img_ids) + len(sampled_img_ids)
        assert len(all_labeled_img_ids) + len(unsampled_img_ids) == self.image_pool_size

        write_coco_pool_split(
            self.oracle_json,
            all_labeled_img_ids,
            unsampled_img_ids,
            Path(out_label_path),
            Path(out_unlabeled_path),
            labeled_include_annotations=True,
        )

        self.latest_labeled = out_label_path
        return {
            'last_labeled_count': len(last_labeled_img_ids),
            'new_labeled_count': len(all_labeled_img_ids),
            'new_unlabeled_count': len(unsampled_img_ids),
        }

    def al_round(self, result_path, last_label_path, out_label_path, out_unlabeled_path):
        self.round += 1
        self.latest_labeled = last_label_path

        sampled_img_ids, rest_img_ids = self.al_acquisition(result_path)
        counts = self.create_jsons(
            sampled_img_ids,
            rest_img_ids,
            last_label_path,
            out_label_path,
            out_unlabeled_path,
        )
        return {
            'selected_image_ids': sampled_img_ids,
            'selected_count': len(sampled_img_ids),
            'metrics': counts,
        }
