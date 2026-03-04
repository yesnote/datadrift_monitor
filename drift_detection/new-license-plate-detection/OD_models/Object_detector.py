import cv2
import numpy as np
from Data.Data_util import coco_names,plot_image_with_boxes,mmdet_coco_classes,SuperStore_classes,SuperStore_classes_YOLO
import random
from OD_models.Faster_RCNN.faster_rcnn import fasterrcnn_resnet50_fpn_local,get_faster_rcnn_resnet50_fpn
from OD_models.yolov5.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
from torchvision.ops import box_iou
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import GPUtil
import torch
try:
    from mmdet.apis.inference import init_detector, inference_detector
except ModuleNotFoundError:
    init_detector = None
    inference_detector = None

"""
This module is used to handle object detection models.
Includes: load a model, generate a prediction, plot a prediction, etc...
"""


class Object_detection_model():

    def __init__(self, model_params, decision_threshold, device, img_size, target_model_path=None):
        super().__init__()
        self.device = device
        self.algorithm = model_params['model_algorithm']
        self.img_size = (img_size,img_size)
        self.model_dataset = model_params['model_dataset']
        self.model_params = model_params
        self.model = self.upload_pre_train_model(model_params, decision_threshold, target_model_path)
        self.COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

    def upload_pre_train_model(self, model_params, decision_threshold, target_model_path):
        """
        Function that uploads pytorch pre-trained detectors (such as faster rcnn)
        :param model_algorithm: Optional. the backbone that the model is based on, default is 'Faster_RCNN'.
        :return: A pre-train model in pytorch format.
        """
        if model_params['model_dataset']== 'COCO':
            model = self.load_COCO_model(decision_threshold,target_model_path)
        if model_params['model_dataset']== 'SuperStore':
            model = self.load_SuperStore_model(model_params,decision_threshold,target_model_path)
        model.eval().to(self.device)
        return model

    def load_COCO_model(self,decision_threshold,target_model_path):
        if self.algorithm == 'Faster_RCNN':
            model = fasterrcnn_resnet50_fpn_local(pretrained=True)
        elif self.algorithm == 'YOLOv5':
            yolo_weights = target_model_path[0] if target_model_path else 'yolov5x.pt'
            class_names = self.model_params.get('class_names', None)
            # Guard against Faster R-CNN style COCO labels ("__background__", "N/A", etc.)
            # which are misaligned with YOLOv5 0..79 class indices.
            if class_names is not None:
                if len(class_names) != 80 or class_names[0] == '__background__':
                    class_names = None
            model = YOLOV5TorchObjectDetector(yolo_weights, self.device, img_size=self.img_size, names=class_names,
                                              confidence=decision_threshold)
        elif self.algorithm == 'MMDetection':
            if init_detector is None:
                raise ModuleNotFoundError("MMDetection is selected but 'mmdet' is not installed.")
            model = init_detector(target_model_path[1], target_model_path[0], device=self.device)
        return model

    def load_SuperStore_model(self,model_params,decision_threshold,target_model_path):
        if self.algorithm == 'Faster_RCNN':
            model_state_dict = torch.load(target_model_path[0])
            model = get_faster_rcnn_resnet50_fpn(num_classes=model_params['num_of_classes']+1)
            # load weights
            model.load_state_dict(model_state_dict)
        elif self.algorithm == 'MMDetection':
            if init_detector is None:
                raise ModuleNotFoundError("MMDetection is selected but 'mmdet' is not installed.")
            model = init_detector(target_model_path[1], target_model_path[0], device=self.device)
        elif self.algorithm == 'YOLOv5':
            class_names = self.model_params.get('class_names', SuperStore_classes_YOLO)
            model = YOLOV5TorchObjectDetector(target_model_path[0], self.device, img_size=self.img_size, names=class_names,
                                              confidence=decision_threshold,fuse=False)
        return model


    def predict_wrapper(self, image_dataloader, detection_threshold = 0.9, use_grad=True, saliency_maps = None, DiL_scores=None):
        """
        Object detection predict function. Detects object in the given scene by a given threshold.
        :param image_dataloader: required. numpy array of images (in numpy array format).
        :param detection_threshold: optional, int. The confidence threshold upon the detections is set.
        :return: numpy array of dictionaries containing detections. Each dictionary contains bounding boxes,
        classes and labels.
        """
        prediction_dicts = []
        # GPUtil.showUtilization()
        if saliency_maps is None:
            saliency_maps = [None] * len(image_dataloader)
            DiL_scores = [None] * len(image_dataloader)
        if isinstance(detection_threshold, float):
            detection_threshold = [detection_threshold] * len(image_dataloader)
        for index,(input_tensor,saliency_map,DiL_score) in enumerate(tqdm(zip(image_dataloader,saliency_maps,DiL_scores), desc="Detecting objects in images")):
            outputs = self.predict(input_tensor,saliency_map,use_grad)
            prediction_dicts.append(self.process_preds(outputs, detection_threshold[index]))
        # GPUtil.showUtilization()
        del outputs
        torch.cuda.empty_cache()
        # GPUtil.showUtilization()
        return np.array(prediction_dicts)

    def predict(self,input_tensor,saliency_map,use_grad):
        if self.algorithm == "Faster_RCNN" or self.algorithm =='YOLOv5':
            if not use_grad:
                with autocast(enabled=True):
                    with torch.no_grad():
                        input_tensor = input_tensor.to(self.device)
                        outputs = self.model(input_tensor)
            else:
                input_tensor = input_tensor.to(self.device)
                outputs = self.model(input_tensor)
        else:
            if inference_detector is None:
                raise ModuleNotFoundError("MMDetection inference requested but 'mmdet' is not installed.")
            imgs = input_tensor.detach().cpu().numpy()*255
            imgs = imgs.squeeze(0)
            imgs = np.transpose(imgs,(1,2,0))
            imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
            result = inference_detector(self.model, imgs)
            outputs = self.mmdetection_to_faster_rcnn_result(result.pred_instances)
        return outputs

    def extract_prediction(self,outputs,detection_threshold = 0.8):
        """
        Helper function to predict, that extract detection as dictionary from a raw prediction (from a single frame).
        :param outputs: required, a row tensor prediction.
        :param detection_threshold: required, int. The confidence threshold upon the detections is set.
        :return: Dictionary containing detections (bounding boxes, classes and labels).
        """
        if self.model_dataset =='SuperStore':
            try:
                check = outputs[0]
            except:
                outputs = [outputs]
            pred_labels = [SuperStore_classes[pred_label] for pred_label in outputs[0]['labels'].cpu().numpy()]
        else:
            pred_labels = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_classes = outputs[0]['labels'].cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        boxes, classes, labels, indices, scores = self.filter_low_confidence_predictions(pred_classes, pred_labels,
                                                                                 pred_scores, pred_bboxes,
                                                                                 detection_threshold)

        return boxes, classes, labels, indices, scores

    def filter_low_confidence_predictions(self,pred_classes,pred_labels,pred_scores,pred_bboxes,detection_threshold):
        """
        Help function to 'extract_prediction' that filter low confidence predictions.
        :param pred_classes: required, numpy array of strings representing the detection classes.
        :param pred_labels: required, numpy array of ints representing the detection labels.
        :param pred_scores: required, numpy array of floats representing the detection confidence.
        :param pred_bboxes: required, numpy array of lists representing the detection bounding box.
        :param detection_threshold: required, int. The confidence threshold upon the detections is set.
        :return: The detections that their confidence was above the given threshold.
        """
        boxes, classes, labels, indices, scores = [], [], [], [], []
        for index in range(len(pred_scores)):
            if pred_scores[index] >= detection_threshold:
                boxes.append(pred_bboxes[index].astype(np.int32))
                classes.append(pred_classes[index])
                labels.append(pred_labels[index])
                indices.append(index)
                scores.append(pred_scores[index])
        boxes = np.int32(boxes)
        return boxes, classes, labels, indices, scores

    @staticmethod
    def extract_prediction_faster_rcnn_dict_format(outputs,threshold = 0.8):
        filter_element = len(outputs[0]['scores'][outputs[0]['scores']>threshold])
        outputs[0]['boxes'] = outputs[0]['boxes'][:filter_element]
        outputs[0]['labels'] = outputs[0]['labels'][:filter_element]
        outputs[0]['scores'] = outputs[0]['scores'][:filter_element]
        return outputs


    def extract_predictions_YOLO(self,results,detection_threshold):
        boxes = results[0][0]
        class_ids = results[1][0]
        labels = results[2][0]
        # Convert YOLO class ids (0-based) to category ids (1-based) used by the eval pipeline.
        classes = [int(class_id) + 1 for class_id in class_ids]
        scores = results[3][0]
        return boxes, classes, labels, None, scores


    def transform_classes_from_yolo_to_faster(self,yolo_labels):
        if self.model_dataset=='COCO':
            return [coco_names.index(label) for label in yolo_labels]
        else:
            return [SuperStore_classes.index(label) for label in yolo_labels]


    @staticmethod
    def transform_outputs_to_faster_rcnn_format(results,device):
        result_faster_rcnn_format = []
        result_dict = {'boxes': torch.empty(size=(0, 4), device=device),
                       'labels': torch.empty(size=(0,), device=device, dtype=torch.int64),
                       'scores': torch.empty(size=(0,), device=device)}
        if len(results.pred[0]) > 0:
            result = results.pandas().xyxy[0].sort_values('confidence', ascending=False)
            result_dict['labels'] = torch.tensor(self.process_yolo_pred(result['name']), device=device)
            result_dict['scores'] = torch.tensor(result['confidence'], device=device)
            result_dict['boxes'] = torch.stack(self.process_yolo_bbox(result[['xmin', 'ymin', 'xmax', 'ymax']]))
        result_faster_rcnn_format.append(result_dict)
        return result_faster_rcnn_format


    def mmdetection_to_faster_rcnn_result(self, pred_instances):

        result_dict = {'boxes': torch.empty(size=(0, 4), device=self.device),
                       'labels': torch.empty(size=(0,), device=self.device, dtype=torch.int64),
                       'scores': torch.empty(size=(0,), device=self.device)}
        result_dict['boxes'] = pred_instances['bboxes']
        if len(pred_instances['labels'])>0:
            result_dict['labels'] = torch.stack([self.adjust_mmdetection_prediction_clases(label) for label in pred_instances['labels']])
        result_dict['scores'] = pred_instances['scores']
        return [[result_dict]]


    def evaluate_predictions(self, dataset,predictions,decision_threshold):
        image_ids = [image_dict['id'] for image_dict in dataset.image_dict]
        preds_list = self.process_preds_for_eval(predictions,image_ids,decision_threshold)
        coco_detections = dataset.coco.loadRes(preds_list)
        dataset.eval(coco_detections,'bbox',list(range(0, len(coco_names))))

    def process_preds_for_eval(self,predictions,image_ids,decision_threshold):
        preds_list = []
        if isinstance(decision_threshold, float):
            decision_threshold = [decision_threshold] * len(image_ids)
        for index,(id,preds) in enumerate(tqdm(zip(image_ids,predictions), desc="Processing predictions for evaluation")):
            for i in range(len(preds['boxes'])):
                if float(preds['scores'][i]) > decision_threshold[index]:
                    pred_dict = {}
                    pred_dict['image_id'] = id
                    pred_dict['category_id'] = int(preds['classes'][i])
                    pred_dict['bbox'] = self.transform_bbox_to_yolo_format(preds['boxes'][i])
                    pred_dict['score'] = preds['scores'][i]
                    preds_list.append(pred_dict)

        return preds_list



    def plot_predictions(self, patched_images, image_dicts,output_path,mode='benign'):
        images = [patched_image.detach().cpu().numpy() * 255 for patched_image in patched_images]
        images_id = [image_dict['id'] for image_dict in image_dicts]
        if mode == 'benign':
            pred_dicts = [image_dict['benign_prediction'] for image_dict in image_dicts]

        else:
            pred_dicts = [image_dict['adversarial_prediction'] for image_dict in image_dicts]

        for image_id, patch_image, pred_dict in zip(images_id, images, pred_dicts):
            # print(f'Trying to plot {os.path.join(output_path, str(image_id))}')
            patch_image = np.squeeze(patch_image)
            patch_image = np.transpose(patch_image, (1, 2, 0)).astype(np.uint8).copy()
            plot_image_with_boxes(patch_image,
                                  boxes=pred_dict['boxes'],
                                  pred_cls=pred_dict['labels'],
                                  confidence=pred_dict['scores'],
                                  output_path=output_path,
                                  image_id=image_id)

    def transform_bbox_to_yolo_format(self, bbox_list):
        x1 = bbox_list[0]
        y1 = bbox_list[1]
        w = bbox_list[2] - x1
        h = bbox_list[3] - y1
        return [x1, y1, w, h]


    def add_pred_dict(self,boxes, classes, labels, indices, scores):
        """
        Helper function to predict, that receives detection raw data (bounding box, class etc.) and form a dictionary.
        :param boxes: required, numpy array of lists representing the detection bounding box.
        :param classes: required, numpy array of strings representing the detection classes.
        :param labels: required, numpy array of ints representing the detection labels.
        :param indices: required, numpy array of ints representing image indices.
        :return: Dictionary that represent the prediction.
        """
        pred_dict = {
            'boxes': boxes,
            'classes': classes,
            'labels': labels,
            'indices': indices,
            'scores': scores
        }
        return pred_dict

    def draw_boxes(self, boxes, labels, classes, image, scores=None):
        """
        Function that visualize a prediction (draw bounding boxes and classes) on a frame.
        :param boxes: required, numpy array of lists representing the detection bounding box.
        :param labels: required, numpy array of ints representing the detection labels.
        :param classes: required, numpy array of strings representing the detection classes.
        :param image: required. Image in a numpy format.
        :return: The Imge in numpy format with the prediction visualization.
        """

        for i, box in enumerate(boxes):
            color_idx = random.randint(0, len(classes) - 1)
            color = self.COLORS[color_idx]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 6
            )
            label_text = str(labels[i]) if i < len(labels) else str(classes[i])
            if scores is not None and i < len(scores):
                label_text = f"{label_text}: {float(scores[i]):.2f}"
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            text_x = x1 + 5
            text_y = min(y2 - 6, y1 + 20)
            cv2.putText(
                image,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
        return image

    @staticmethod
    def init_seeds(seed=42):
        # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
        # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
        import torch.backends.cudnn as cudnn
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

    def activate_image_manipulation_by_dil_score(self, detection_threshold, saliency_map,dil):
        if dil:
            if dil < 0.2:
                detection_threshold = 0.8
                saliency_map = None
                return detection_threshold,saliency_map
        return detection_threshold, saliency_map

    def process_preds(self, outputs, dynamic_detection_threshold=0.6):
        if self.algorithm == "Faster_RCNN" or self.algorithm == "MMDetection":
            boxes, classes, labels, indices, scores = self.extract_prediction(outputs[0], dynamic_detection_threshold)
        else:
            output = [i for i in outputs[0]]
            boxes, classes, labels, indices, scores = self.extract_predictions_YOLO(output, dynamic_detection_threshold)
        return self.add_pred_dict(boxes, classes, labels, indices, scores)

    def adjust_mmdetection_prediction_clases(self, label):
        if self.model_dataset=='COCO':
            class_label = mmdet_coco_classes[label]
            label_idx_in_coco_names = coco_names.index(class_label)
            return torch.tensor(label_idx_in_coco_names, device='cuda')
        else:

            return torch.tensor(int(label)+1, device='cuda')



class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
    	assign a score on how the current bounding boxes match it,
    		1. In IOU
    		2. In the classification score.
    	If there is not a large enough overlap, or the category changed,
    	assign a score of 0.

    	The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5, require_grad=False):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold
        self.require_grad = require_grad

    def __call__(self, model_outputs):
        if self.require_grad:
            return torch.sum(model_outputs.get("scores"))

        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output

class YOLOBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes,detection_threshold, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.detection_threshold = detection_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        boxes, colors, categories, names, confidences = Object_detection_model.extract_predictions_YOLO(model_outputs,self.detection_threshold)
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

    # def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    #     """Initialize a detector from config file.
    #
    #     Args:
    #         config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
    #             :obj:`Path`, or the config object.
    #         checkpoint (str, optional): Checkpoint path. If left as None, the model
    #             will not load any weights.
    #         cfg_options (dict): Options to override some settings in the used
    #             config.
    #
    #     Returns:
    #         nn.Module: The constructed detector.
    #     """
    #     if isinstance(config, (str, Path)):
    #         config = mmcv.Config.fromfile(config)
    #     elif not isinstance(config, mmcv.Config):
    #         raise TypeError('config must be a filename or Config object, '
    #                         f'but got {type(config)}')
    #     if cfg_options is not None:
    #         config.merge_from_dict(cfg_options)
    #     if 'pretrained' in config.model:
    #         config.model.pretrained = None
    #     elif 'init_cfg' in config.model.backbone:
    #         config.model.backbone.init_cfg = None
    #     config.model.train_cfg = None
    #     model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    #     if checkpoint is not None:
    #         checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    #         if 'CLASSES' in checkpoint.get('meta', {}):
    #             model.CLASSES = checkpoint['meta']['CLASSES']
    #         else:
    #             warnings.simplefilter('once')
    #             warnings.warn('Class names are not saved in the checkpoint\'s '
    #                           'meta data, use COCO classes by default.')
    #             model.CLASSES = get_classes('coco')
    #     model.cfg = config  # save the config in the model for convenience
    #     model.to(device)
    #     model.eval()
    #     return model
    #
    # @staticmethod
    # def fromfile(filename,
    #              use_predefined_variables=True,
    #              import_custom_modules=True):
    #     if isinstance(filename, Path):
    #         filename = str(filename)
    #     cfg_dict, cfg_text = Config._file2dict(filename,
    #                                            use_predefined_variables)
    #     if import_custom_modules and cfg_dict.get('custom_imports', None):
    #         import_modules_from_strings(**cfg_dict['custom_imports'])
    #     return Config(cfg_dict, cfg_text=cfg_text, filename=filename)



