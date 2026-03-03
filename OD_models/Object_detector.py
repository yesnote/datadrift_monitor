import cv2
import numpy as np
import random
import torch
from torch.cuda.amp import autocast
from torchvision.ops import box_iou
from tqdm import tqdm

from Data.Data_util import coco_names, pascal_voc_names, plot_image_with_boxes
from OD_models.yolov5.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector

"""
YOLOv5 object detection wrapper for COCO/PascalVOC.
"""


class Object_detection_model:

    def __init__(self, model_params, decision_threshold, device, img_size, target_model_path=None):
        super().__init__()
        self.device = device
        self.algorithm = model_params.get('model_algorithm', 'YOLOv5')
        self.model_dataset = model_params.get('model_dataset', 'COCO')
        self.img_size = (img_size, img_size)
        self.model_params = model_params

        if self.algorithm != 'YOLOv5':
            raise ValueError("Only 'YOLOv5' is supported in this wrapper.")
        if self.model_dataset not in ('COCO', 'PascalVOC'):
            raise ValueError("Only 'COCO' and 'PascalVOC' dataset flows are supported in this wrapper.")

        self.model = self.upload_pre_train_model(decision_threshold, target_model_path)
        self.dataset_class_names = coco_names if self.model_dataset == 'COCO' else pascal_voc_names
        self.COLORS = np.random.uniform(0, 255, size=(len(self.dataset_class_names), 3))

    def upload_pre_train_model(self, decision_threshold, target_model_path):
        yolo_weights = target_model_path[0] if target_model_path else 'yolov5x.pt'
        default_names = None if self.model_dataset == 'COCO' else pascal_voc_names[1:]
        class_names = self.model_params.get('class_names', default_names)
        model = YOLOV5TorchObjectDetector(
            yolo_weights,
            self.device,
            img_size=self.img_size,
            names=class_names,
            confidence=decision_threshold,
        )
        model.eval().to(self.device)
        return model

    def predict_wrapper(self, image_dataloader, detection_threshold=0.9, use_grad=True, saliency_maps=None, DiL_scores=None):
        prediction_dicts = []
        if saliency_maps is None:
            saliency_maps = [None] * len(image_dataloader)
            DiL_scores = [None] * len(image_dataloader)
        if isinstance(detection_threshold, float):
            detection_threshold = [detection_threshold] * len(image_dataloader)

        for index, (input_tensor, _saliency_map, _dil_score) in enumerate(
            tqdm(zip(image_dataloader, saliency_maps, DiL_scores), desc="Detecting objects in images")
        ):
            outputs = self.predict(input_tensor, use_grad)
            prediction_dicts.append(self.process_preds(outputs, detection_threshold[index]))

        del outputs
        torch.cuda.empty_cache()
        return np.array(prediction_dicts)

    def predict(self, input_tensor, use_grad):
        if not use_grad:
            with autocast(enabled=True):
                with torch.no_grad():
                    input_tensor = input_tensor.to(self.device)
                    return self.model(input_tensor)

        input_tensor = input_tensor.to(self.device)
        return self.model(input_tensor)

    @staticmethod
    def extract_predictions_YOLO(results, _detection_threshold):
        boxes = results[0][0]
        class_ids = results[1][0]
        labels = results[2][0]
        classes = [int(class_id) + 1 for class_id in class_ids]
        scores = results[3][0]
        return boxes, classes, labels, None, scores

    def evaluate_predictions(self, dataset, predictions, decision_threshold):
        if self.model_dataset != 'COCO':
            raise NotImplementedError("evaluate_predictions currently supports COCO only.")

        image_ids = [image_dict['id'] for image_dict in dataset.image_dict]
        preds_list = self.process_preds_for_eval(predictions, image_ids, decision_threshold)
        coco_detections = dataset.coco.loadRes(preds_list)
        dataset.eval(coco_detections, 'bbox', list(range(0, len(coco_names))))

    def process_preds_for_eval(self, predictions, image_ids, decision_threshold):
        preds_list = []
        if isinstance(decision_threshold, float):
            decision_threshold = [decision_threshold] * len(image_ids)

        for index, (image_id, preds) in enumerate(
            tqdm(zip(image_ids, predictions), desc="Processing predictions for evaluation")
        ):
            for i in range(len(preds['boxes'])):
                if float(preds['scores'][i]) > decision_threshold[index]:
                    pred_dict = {
                        'image_id': image_id,
                        'category_id': int(preds['classes'][i]),
                        'bbox': self.transform_bbox_to_yolo_format(preds['boxes'][i]),
                        'score': preds['scores'][i],
                    }
                    preds_list.append(pred_dict)

        return preds_list

    def plot_predictions(self, patched_images, image_dicts, output_path, mode='benign'):
        images = [patched_image.detach().cpu().numpy() * 255 for patched_image in patched_images]
        images_id = [image_dict['id'] for image_dict in image_dicts]
        if mode == 'benign':
            pred_dicts = [image_dict['benign_prediction'] for image_dict in image_dicts]
        else:
            pred_dicts = [image_dict['adversarial_prediction'] for image_dict in image_dicts]

        for image_id, patch_image, pred_dict in zip(images_id, images, pred_dicts):
            patch_image = np.squeeze(patch_image)
            patch_image = np.transpose(patch_image, (1, 2, 0)).astype(np.uint8).copy()
            plot_image_with_boxes(
                patch_image,
                boxes=pred_dict['boxes'],
                pred_cls=pred_dict['labels'],
                confidence=pred_dict['scores'],
                output_path=output_path,
                image_id=image_id,
            )

    @staticmethod
    def transform_bbox_to_yolo_format(bbox_list):
        x1 = bbox_list[0]
        y1 = bbox_list[1]
        w = bbox_list[2] - x1
        h = bbox_list[3] - y1
        return [x1, y1, w, h]

    @staticmethod
    def add_pred_dict(boxes, classes, labels, indices, scores):
        return {
            'boxes': boxes,
            'classes': classes,
            'labels': labels,
            'indices': indices,
            'scores': scores,
        }

    def draw_boxes(self, boxes, labels, classes, image):
        for i, box in enumerate(boxes):
            color_idx = random.randint(0, len(classes) - 1)
            color = self.COLORS[color_idx]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color,
                6,
            )
        return image

    @staticmethod
    def init_seeds(seed=42):
        import torch.backends.cudnn as cudnn
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

    @staticmethod
    def activate_image_manipulation_by_dil_score(detection_threshold, saliency_map, dil):
        if dil and dil < 0.2:
            return 0.8, None
        return detection_threshold, saliency_map

    def process_preds(self, outputs, dynamic_detection_threshold=0.6):
        output = [i for i in outputs[0]]
        boxes, classes, labels, indices, scores = self.extract_predictions_YOLO(output, dynamic_detection_threshold)
        return self.add_pred_dict(boxes, classes, labels, indices, scores)


class YOLOBoxScoreTarget:
    """Compute target score for YOLO predictions from original boxes/labels."""

    def __init__(self, labels, bounding_boxes, detection_threshold, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.detection_threshold = detection_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        boxes, _colors, categories, _names, confidences = Object_detection_model.extract_predictions_YOLO(
            model_outputs,
            self.detection_threshold,
        )
        boxes = torch.Tensor(boxes)
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()
            boxes = boxes.cuda()

        if len(boxes) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = box_iou(box, boxes)
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and categories[index] == label:
                score = ious[0, index] + confidences[index]
                output = output + score
        return output
