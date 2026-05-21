import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


coco_names = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

pascal_voc_names = [
    "__background__",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

kitti_names = [
    "car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc",
]

bdd100k_names = [
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "traffic light", "traffic sign",
]

cityscapes_names = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

foggy_cityscapes_names = list(cityscapes_names)

DATASET_CLASS_NAMES = {
    "coco": [name for name in coco_names[1:] if name != "N/A"],
    "voc": pascal_voc_names[1:],
    "pascal_voc": pascal_voc_names[1:],
    "kitti": kitti_names,
    "bdd100k": bdd100k_names,
    "bdd": bdd100k_names,
    "cityscapes": cityscapes_names,
    "foggy_cityscapes": foggy_cityscapes_names,
    "foggy_city": foggy_cityscapes_names,
}


def list_image_files(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        return []
    files = []
    for name in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
            files.append(os.path.join(image_dir, name))
    return files


def plot_image_with_boxes(img, boxes, pred_cls, confidence, output_path, image_id, save=True):
    text_size = 0.6
    text_th = 2
    rect_th = 3
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
        (255, 255, 0), (255, 165, 0), (128, 0, 128), (255, 105, 180), (0, 255, 0),
    ]

    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=rect_th)
        score = float(confidence[i]) if len(confidence) > i else 0.0
        label = pred_cls[i] if len(pred_cls) > i else "obj"
        cv2.putText(
            img,
            f"{label} {score:.2f}",
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (0, 150, 255),
            thickness=text_th,
        )

    if not save:
        return img

    os.makedirs(output_path, exist_ok=True)
    file_stem = os.path.splitext(os.path.basename(str(image_id)))[0]
    fig = plt.figure(figsize=(10, 7))
    plt.axis("off")
    plt.imshow(img)
    plt.savefig(os.path.join(output_path, f"{file_stem}.jpg"), dpi=fig.dpi)
    plt.close(fig)


def read_image_as_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
