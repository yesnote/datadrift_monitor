import json
import os
import random
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage.io as io
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset

from OD_models.yolov5.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector

"""
This module is used for upload, explore and evaluate datasets.
YOLOv5 + COCO/PascalVOC.
"""

id = 1600
coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

pascal_voc_names = [
    '__background__',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def get_class_names_for_dataset(model_dataset):
    if model_dataset == 'COCO':
        return coco_names
    if model_dataset == 'PascalVOC':
        return pascal_voc_names
    raise ValueError(f"Unsupported model_dataset: {model_dataset}")


def load_dataset(dataset_path, model_algorithm='YOLOv5', batch_size=1):
    if model_algorithm != 'YOLOv5':
        raise ValueError("Only 'YOLOv5' is supported.")

    images_paths = [os.path.join(dataset_path, img_file) for img_file in os.listdir(dataset_path)]
    image_dataset = ImageDataset(images_paths=images_paths, backbone_name=model_algorithm)
    return DataLoader(image_dataset, batch_size=batch_size, num_workers=6, pin_memory=True)


def load_dataset_attack_format(dataset_path, algorithm_name='YOLOv5'):
    if algorithm_name != 'YOLOv5':
        raise ValueError("Only 'YOLOv5' is supported.")

    images_paths = [os.path.join(dataset_path, img_file) for img_file in os.listdir(dataset_path)]
    images_paths = sorted(images_paths)
    images = [load_image(image, algorithm_name) for image in images_paths]
    file_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path in images_paths]
    return images, file_names


def load_images(dataset_path):
    images = []
    for file in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, file)
        image = load_image(image_path, 'YOLOv5')
        images.append(image)
    return images


def load_images_yolo(dataset_path):
    images_paths = [os.path.join(dataset_path, img_file) for img_file in os.listdir(dataset_path)]
    images_paths = sorted(images_paths)
    images = [cv2.imread(image) for image in images_paths]
    processed_images = [yolo_preprocessing(image) for image in images]
    return processed_images


def image_preprocess(image, algorithm_name='YOLOv5'):
    if algorithm_name != 'YOLOv5':
        raise ValueError("Only 'YOLOv5' is supported.")
    return yolo_preprocessing(image)


def yolo_preprocessing(img):
    if len(img.shape) != 4:
        img = np.expand_dims(img, axis=0)
    img = img.astype(np.uint8)
    img = np.array(YOLOV5TorchObjectDetector.yolo_resize(img.squeeze(0), new_shape=(640, 640))[0])
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img / 255.0
    img = img.unsqueeze(0)
    return img


def load_image(image, algorithm_name='YOLOv5'):
    if isinstance(image, str):
        image = np.array(Image.open(image))
    return image_preprocess(image, algorithm_name)


def process_image_for_real_time_pipeline(image):
    image_float_np = np.float32(image) / 255
    image_transpose = np.transpose(image_float_np, (0, 3, 1, 2))
    image = torch.from_numpy(image_transpose)
    return image


def transfomations():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform


def get_image_paths(dataset_path, image_width=640, image_height=640):
    images = []
    for image in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image)
        images.append(image_path)
    images = np.array(sorted(images))
    return images


def load_image_in_art_format(image_path, image_width=640, image_height=640):
    image = io.imread(image_path)
    image = image.astype(np.float32)
    return image


def plot_image_with_boxes(img, boxes, pred_cls, confidence, output_path, image_id, save=True):
    text_size = 0.6
    text_th = 2
    rect_th = 6
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
        (255, 255, 0), (255, 165, 0), (128, 0, 128), (255, 105, 180), (0, 255, 0)
    ]

    for i in range(len(boxes)):
        boxes_as_int = [int(coordinate) for coordinate in boxes[i]]
        start_point = (boxes_as_int[0], boxes_as_int[1])
        end_point = (boxes_as_int[2], boxes_as_int[3])
        color_idx = random.randint(0, len(colors) - 1)
        cv2.rectangle(img, start_point, end_point, color=colors[color_idx], thickness=rect_th)

        curr_confidence = "%.2f" % confidence[i]
        prediction_text = f'{pred_cls[i]} {curr_confidence}'
        start_point = (boxes_as_int[0] + 5, boxes_as_int[1] + 20)
        cv2.putText(img, prediction_text, start_point, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 150, 255), thickness=text_th)

    if save:
        image_id = os.path.splitext(os.path.basename(str(image_id)))[0]
        fig = plt.figure(figsize=(10, 7))
        plt.axis("off")
        plt.imshow(img, interpolation="nearest")
        plt.savefig(f'{output_path}/{image_id}.jpg', dpi=fig.dpi)
    else:
        return img


def process_prediction(images, prediction_dicts, original_images):
    process_prediction_dicts = []
    for image, prediction_dict, original_image in zip(images, prediction_dicts, original_images):
        image_height, image_width = image.size(2), image.size(3)
        original_image_height, original_image_width = (original_image['height'], original_image['width'])
        width_diff = original_image_width / image_width
        height_diff = original_image_height / image_height
        process_prediction_dict = process_bbox(prediction_dict, width_diff, height_diff)
        process_prediction_dicts.append(process_prediction_dict)
    return prediction_dicts


def process_bbox(prediction_dict, width_diff, height_diff):
    process_boxes = []
    for box in prediction_dict['boxes']:
        x_1 = int(np.round(box[0] * width_diff))
        x_2 = int(np.round(box[2] * width_diff))
        y_1 = int(np.round(box[1] * height_diff))
        y_2 = int(np.round(box[3] * height_diff))
        process_boxes.append([x_1, y_1, x_2, y_2])
    prediction_dict['boxes'] = process_boxes
    return prediction_dict


class ImageDataset(Dataset):
    def __init__(self, images_paths, backbone_name='YOLOv5'):
        if backbone_name != 'YOLOv5':
            raise ValueError("Only 'YOLOv5' is supported.")
        self.images_paths = images_paths
        self.backbone_name = backbone_name

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx])
        image = image_preprocess(image, self.backbone_name)
        return image

    def transfomations(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        return transform


class MS_COCO_util:
    """
    Class representing MSCOCO util object.
    """

    def __init__(self, dataset_path, annotations_file_path, images_path):
        super().__init__()
        self.dataset_path = dataset_path
        self.annotations_file_path = annotations_file_path
        self.coco = self.create_coco_object()
        self.ids_list = self.set_ids_list(images_path)
        self.image_dict = self.load_coco_dataset_from_ids_list()

    def create_coco_object(self):
        return COCO(self.annotations_file_path)

    def set_ids_list(self, images_path):
        return [int(image_file.split('/')[-1].split('.')[0]) for image_file in images_path]

    def get_imgs_dict_by_id_list(self):
        imgIds = self.coco.getImgIds(imgIds=self.ids_list)
        imgIds = sorted(imgIds)
        imgIds = self.coco.loadImgs(imgIds)
        return imgIds

    def get_img_by_img_dict(self, img_dict):
        return io.imread(img_dict['coco_url'])

    def get_img_annotations_by_img_dict(self, img_dict):
        annIds = self.coco.getAnnIds(imgIds=img_dict['id'], catIds=[], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        return anns

    def plot_image_bbox(self, img_dict, catIds):
        img = self.get_img_by_img_dict(img_dict)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)
        plt.axis('off')
        annIds = self.coco.getAnnIds(imgIds=img_dict['id'], catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        for i, ann in enumerate(anns):
            plt.gca().add_patch(
                Rectangle(
                    (anns[i]['bbox'][0], anns[i]['bbox'][1]),
                    anns[i]['bbox'][2],
                    anns[i]['bbox'][3],
                    edgecolor='green',
                    facecolor='none',
                )
            )
            ax.text(
                anns[i]['bbox'][0],
                anns[i]['bbox'][1],
                coco_names[anns[i]['category_id']],
                style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5},
            )
        plt.show()

    def label_list_to_indexes(self, label_list):
        return [self.label_to_index(label) for label in label_list]

    def label_to_index(self, label):
        return coco_names.index(label)

    def index_to_label(self, index):
        return coco_names[index]

    def display_coco_categories(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

    def load_coco_dataset_from_ids_list(self):
        images_dict = self.get_imgs_dict_by_id_list()
        for img_dict in images_dict:
            img_dict['annotations'] = self.get_img_annotations_by_img_dict(img_dict)
            img_dict['image'] = load_image_in_art_format(img_dict['coco_url'])
        return images_dict

    def eval(self, cocoDt, annType, cat_ids):
        coco_eval = COCOeval(self.coco, cocoDt, annType)
        imgIds = sorted(self.coco.getImgIds(imgIds=self.ids_list))
        coco_eval.params.imgIds = imgIds
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def image_dict_to_json(self, images_dict, output_path):
        images = []
        categories = []
        annotations = []
        for image_dict in images_dict:
            images.append(self.process_image(image_dict, output_path))
            for annotation in image_dict['annotations']:
                annotations.append(self.process_annotations(image_dict, annotation))

        data_coco = {
            "images": images,
            "categories": categories,
            "annotations": annotations,
        }
        return data_coco

    def process_image(self, image_dict, output_path):
        image = {
            "height": image_dict['height'],
            "width": image_dict['width'],
            "id": image_dict['id'],
            "file_name": image_dict['file_name'],
        }
        output_image = cv2.cvtColor(image_dict['image'], cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{output_path}/{image_dict['id']}.jpeg", output_image)
        return image

    def process_annotations(self, image_dict, curr_annotation):
        area = image_dict['width'] * image_dict['height']
        annotation = {
            "segmentation": [],
            "iscrowd": curr_annotation['iscrowd'],
            "area": float(area),
            "image_id": curr_annotation['image_id'],
            "bbox": curr_annotation['bbox'],
            "category_id": curr_annotation['category_id'],
            "id": curr_annotation['id'],
        }
        return annotation

    def get_ids_from_specific_class(self, classes):
        ids_list = []
        for class_name in classes:
            catIds = self.coco.getCatIds(catNms=class_name)
            imgIds = self.coco.getImgIds(catIds=catIds)
            imgIds = imgIds[:300]
            ids_list.append(imgIds)
        self.ids_list = [item for sublist in ids_list for item in sublist]

    def transform_ground_truth_to_dict(self):
        ground_truth_dicts = []
        for image_dict in self.image_dict:
            boxes = np.array([self.xywh2x1y1x2y2(annotation['bbox']) for annotation in image_dict['annotations']])
            labels = np.array([annotation['category_id'] for annotation in image_dict['annotations']])
            scores = np.ones(len(labels))
            ground_truth_dict = {
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
            }
            ground_truth_dicts.append(ground_truth_dict)
        return ground_truth_dicts

    @staticmethod
    def xywh2x1y1x2y2(bbox):
        x_center = bbox[0]
        y_center = bbox[1]
        width = bbox[2]
        height = bbox[3]
        x1 = x_center - width / 2
        x2 = x_center + width / 2
        y1 = y_center - height / 2
        y2 = y_center + height / 2
        return np.array([x1, y1, x2, y2])


class PascalVOC_util:
    """
    Utility for Pascal VOC style datasets.
    Expected structure:
    - dataset_path/JPEGImages/*.jpg|png
    - dataset_path/Annotations/*.xml
    - dataset_path/ImageSets/Main/{train,val,trainval,test}.txt (optional)
    """

    def __init__(self, dataset_path, image_set='train', ids_list_path=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.annotations_dir = os.path.join(dataset_path, 'Annotations')
        self.images_dir = os.path.join(dataset_path, 'JPEGImages')
        self.ids_list = self.set_ids_list(image_set=image_set, ids_list_path=ids_list_path)
        self.image_dict = self.load_voc_dataset_from_ids_list()

    def set_ids_list(self, image_set='train', ids_list_path=None):
        if ids_list_path is not None:
            with open(ids_list_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]

        split_file = os.path.join(self.dataset_path, 'ImageSets', 'Main', f'{image_set}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]

        return [
            os.path.splitext(file_name)[0]
            for file_name in os.listdir(self.annotations_dir)
            if file_name.lower().endswith('.xml')
        ]

    def _resolve_image_path(self, image_id):
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            candidate = os.path.join(self.images_dir, f'{image_id}{ext}')
            if os.path.exists(candidate):
                return candidate
        return os.path.join(self.images_dir, f'{image_id}.jpg')

    def parse_voc_annotation(self, image_id):
        annotation_path = os.path.join(self.annotations_dir, f'{image_id}.xml')
        root = ET.parse(annotation_path).getroot()

        file_name = root.findtext('filename')
        if not file_name:
            file_name = os.path.basename(self._resolve_image_path(image_id))

        size_node = root.find('size')
        width = int(size_node.findtext('width'))
        height = int(size_node.findtext('height'))

        annotations = []
        ann_id = 0
        for obj in root.findall('object'):
            class_name = obj.findtext('name')
            if class_name not in pascal_voc_names:
                continue

            difficult = int(obj.findtext('difficult', default='0'))
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.findtext('xmin'))
            ymin = float(bndbox.findtext('ymin'))
            xmax = float(bndbox.findtext('xmax'))
            ymax = float(bndbox.findtext('ymax'))
            width_box = max(0.0, xmax - xmin)
            height_box = max(0.0, ymax - ymin)

            annotations.append({
                'segmentation': [],
                'iscrowd': difficult,
                'area': float(width_box * height_box),
                'image_id': image_id,
                'bbox': [xmin, ymin, width_box, height_box],
                'category_id': pascal_voc_names.index(class_name),
                'id': ann_id,
            })
            ann_id += 1

        image_path = self._resolve_image_path(image_id)
        image = load_image_in_art_format(image_path)
        return {
            'id': image_id,
            'file_name': file_name,
            'width': width,
            'height': height,
            'image': image,
            'annotations': annotations,
        }

    def load_voc_dataset_from_ids_list(self):
        return [self.parse_voc_annotation(image_id) for image_id in self.ids_list]

    def label_to_index(self, label):
        return pascal_voc_names.index(label)

    def index_to_label(self, index):
        return pascal_voc_names[index]

    def transform_ground_truth_to_dict(self):
        ground_truth_dicts = []
        for image_dict in self.image_dict:
            boxes = np.array([self.xywh2x1y1x2y2(annotation['bbox']) for annotation in image_dict['annotations']])
            labels = np.array([annotation['category_id'] for annotation in image_dict['annotations']])
            scores = np.ones(len(labels))
            ground_truth_dicts.append({
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
            })
        return ground_truth_dicts

    @staticmethod
    def xywh2x1y1x2y2(bbox):
        x1 = bbox[0]
        y1 = bbox[1]
        width = bbox[2]
        height = bbox[3]
        return np.array([x1, y1, x1 + width, y1 + height])


def transform_to_yolo_format(bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    image_width = 640
    image_height = 360

    x_center = (x1 + x2) / (2 * image_width)
    y_center = (y1 + y2) / (2 * image_height)
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return [round(x_center, 3), round(y_center, 3), round(width, 3), round(height, 3)]


def edit_dataset(input_path, output_path):
    for label_file_path in os.listdir(input_path):
        with open(os.path.join(input_path, label_file_path), "r") as f:
            data = json.loads(f.read())

        boxes = transform_to_yolo_format(data['boxes'][0])
        label_file_path = label_file_path.split('.')[0]
        with open(f"{os.path.join(output_path, label_file_path)}.txt", "w") as text_file:
            text_file.write(f'{13} {boxes[0]} {boxes[1]} {boxes[2]} {boxes[3]}')


def plot_image(input_image, input_labels):
    cv2.imread(input_image)


def check_duplicates(dir1, dir2):
    dir1_files = set(os.listdir(dir1))
    dir2_files = set(os.listdir(dir2))

    unique_files = dir1_files - dir2_files
    for file in unique_files:
        print(file)


def print_files_name(path):
    for file_path in os.listdir(path):
        print(file_path.split(".")[0])


def create_dict(idx, file_name):
    return {
        "height": 360,
        "width": 640,
        "id": idx + 1600,
        "file_name": file_name,
    }


def xcycwh2x1y1wh(anno):
    xc, yc, w, h = anno
    h = float(h.replace("\n", ""))
    xc, yc, w = float(xc), float(yc), float(w)
    x1 = (xc - (w / 2)) * 640
    y1 = (yc - (h / 2)) * 360
    w = w * 640
    h = h * 360
    return x1, y1, w, h


def data_disturbution(array, mode):
    np_array = np.array(array)
    thresholds = np.linspace(10, 100, 10)
    percentiles = [np.percentile(np_array, thr, axis=0) for thr in thresholds]
    counts = []
    for threshold in percentiles:
        if mode == 'TN':
            count = np.sum(array <= threshold)
        else:
            count = np.sum(array > threshold)
        counts.append(count)

    total = 800
    percentages = [count / total * 100 for count in counts]
    percentiles = [round(threshold, 5) for threshold in percentiles]

    sns.set_style("darkgrid", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    plt.figure(figsize=(10, 6))
    if len(percentiles) > 1:
        percentiles[1] = 0.0001
    sns.barplot(x=percentiles, y=percentages, color='skyblue', edgecolor='black')
    plt.title(f'Clean {mode} per DiL value', fontsize=15)
    plt.xlabel('DiL value', fontsize=12)
    plt.ylim((0, 100))
    plt.ylabel('CLean Data Percentage', fontsize=12)
    plt.show()


def upload_annotation_file(path):
    with open(path) as json_file:
        data = json.load(json_file)

    original_image_ids = [image['file_name'].split('.')[0].split('-')[1] for image in data['images']]
    for idx in range(len(data['annotations'])):
        image_id = data['annotations'][idx]['image_id']
        x1, y1, width, height = data['annotations'][idx]['bbox']
        x2, y2 = x1 + width, y1 + height
        data['annotations'][idx]['bbox'] = [x1, y1, x2, y2]
        anno_category = data['categories'][data['annotations'][idx]['category_id']]['name']
        data['annotations'][idx]['category_id'] = coco_names.index(anno_category)
        data['annotations'][idx]['original_image_id'] = original_image_ids[image_id]

    return data
