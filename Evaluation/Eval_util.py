import copy
import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from torch import IntTensor, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

"""
This module is used to evaluate object detection models.
Includes object detection performance metrics such as: mAP, recall, False positive rate.
"""

class evaluation_module():
    def __init__(self, od_model,model_params,COCO_dataset_params,evaluation_params):
        self.model_params = model_params
        self.dataset_params = COCO_dataset_params
        self.od_model = od_model
        self.evaluation_params = evaluation_params


    def init_experiment_folder(self):

        """
        Function for crating output folders.
        :return: Create output folders.
        """
        time = datetime.datetime.now().strftime("%d-%m-%Y_%H;%M")
        output_path = f"{self.evaluation_params['experiment_folder']}/{time}"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        folder_param = {
            'saved_base_model_preds':'base_model_preds',
            'saved_manipulated_images':'manipulated_images',
            'saved_robust_model_preds_ddt_manipulated':'robust_model_preds_ddt_manipulated',
            'saved_robust_model_preds_ddt':'robust_model_preds_ddt',
            'saved_robust_model_preds_manipulated':'robust_model_preds_manipulated',
            'saved_XAI':'XAI'
        }
        [self.create_folder(output_path=output_path,dir_name=dir_name,param_name=param_name) for param_name,dir_name in folder_param.items()]

    def create_folder(self, output_path, dir_name, param_name):
        curr_dir = f'{output_path}/{dir_name}'
        Path(curr_dir).mkdir(parents=True, exist_ok=True)
        self.evaluation_params[param_name] = curr_dir

    def set_output_path(self, mode,dynamic_thresholds):
        if dynamic_thresholds:
            return self.evaluation_params['saved_robust_model_preds']
        return self.evaluation_params['saved_base_model_preds']

    @staticmethod
    def get_base_model_performance(gt_objects_counter, pred_objects_predicted):
        missed_objects_counter = 0
        for gt_number_of_objects,pred_number_of_objects in zip(gt_objects_counter,pred_objects_predicted):
            missed_objects_counter+= max(gt_number_of_objects-pred_number_of_objects,0)
        gt_sum = sum(gt_objects_counter)
        return 1-(missed_objects_counter/gt_sum)

    @staticmethod
    def get_model_fp(gt_objects_counter, pred_objects_predicted):
        fp_objects_counter = 0
        for gt_number_of_objects, pred_number_of_objects in zip(gt_objects_counter, pred_objects_predicted):
            fp_objects_counter += max(pred_number_of_objects-gt_number_of_objects, 0)
        pred_sum = sum(pred_objects_predicted)
        return fp_objects_counter / pred_sum

    @staticmethod
    def get_base_robustness_models_performance_difference(gt_objects_counter, base_pred_objects,robust_pred_objects):
        missed_base_objects_counter = 0
        missed_robust_objects_counter = 0
        for gt_number_of_objects, base_pred_number_of_objects,robust_pred_number_of_objects in zip(gt_objects_counter, base_pred_objects, robust_pred_objects):
            missed_base_objects_counter += max(gt_number_of_objects - base_pred_number_of_objects, 0)
            missed_robust_objects_counter += max(gt_number_of_objects - robust_pred_number_of_objects, 0)
        return 1 - (missed_robust_objects_counter / missed_base_objects_counter)

    def create_evaluation_dict(self,images,file_names,gt_objects_counter):
        eval_dict = {}
        gt_objects_counter = list(gt_objects_counter.values())[1:]
        gt_objects_counter = [int(x) for x in gt_objects_counter]
        for image,file_name,gt_object_counter in zip(images,file_names,gt_objects_counter):
            eval_dict[file_name] = {'image':image,'ground_truth_label':gt_object_counter}
        return eval_dict

    def mAP(self, predictions, annotations, file_names,images,mode):
        annotations['annotations'] = sorted(annotations['annotations'], key=lambda d: d['original_image_id'])
        if type(annotations['annotations'][0]['original_image_id'])==int:
            file_names = [int(file_name) for file_name in file_names]
        annotations['annotations'] = [d for d in annotations['annotations'] if
                                      d.get('original_image_id') in file_names]
        gt_boxes = self.adjust_gt_boxes(annotations)
        gt_classes = self.adjust_gt_classes(annotations)
        gt = self.transform_gt_to_dicts(gt_boxes,gt_classes)
        preds = self.adjust_preds(predictions,images,annotations,file_names,mode)
        # pred_boxes,pred_classes = self.adjust_pred_boxes_classes(predictions)
        # gt_boxes,gt_classes = self.arrange_annotations_order(gt_boxes,gt_classes,image_ids,file_names)
        self.calculate_map(gt,preds)
        # self.compute_map(gt_boxes,pred_boxes,gt_classes,pred_classes)

    def compute_iou(self,box1, box2):
        """Compute the Intersection Over Union (IOU) of two bounding boxes."""
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        inter_xmin = max(xmin1, xmin2)
        inter_ymin = max(ymin1, ymin2)
        inter_xmax = min(xmax1, xmax2)
        inter_ymax = min(ymax1, ymax2)
        inter_area = max(0, inter_xmax - inter_xmin + 1) * max(0, inter_ymax - inter_ymin + 1)
        box1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
        box2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou
    def compute_map(self,gt_boxes, pred_boxes, gt_classes, pred_classes):
        aps = []
        for i in range(len(gt_boxes)):
            if len(pred_boxes[i]) == 0:  # No predicted boxes for this image
                if len(gt_boxes[i]) == 0:  # No ground truth boxes either
                    aps.append(1)  # Perfect prediction
                else:  # There were objects, but none were detected
                    aps.append(0)  # No detections => AP = 0
                continue

            pred_box = np.array(pred_boxes[i])
            gt_box = np.array(gt_boxes[i])

            iou = self.compute_iou(pred_box[:,:-1], gt_box)
            overlap = iou.max(axis=1)
            overlap_gt = iou.max(axis=0)

            tp = (overlap >= 0.5) & (np.array(pred_classes[i]) == np.array(gt_classes[i])[overlap_gt.argmax(axis=0)])
            fp = ~tp

            scores = pred_box[:, -1]
            sort_idx = np.argsort(-scores)
            tp = tp[sort_idx]
            fp = fp[sort_idx]

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / (len(gt_box) + np.finfo(float).eps)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps)

            ap = average_precision_score(recalls, precisions)
            aps.append(ap)

        return np.mean(aps)

    def scale_bbox(self, bbox, original_size, target_size):
        """
        Scales a bounding box from its original image size to a target image size.

        Parameters:
        - bbox (tuple): A tuple containing the bounding box coordinates (x1, y1, x2, y2).
        - original_size (tuple): A tuple containing the original image size (width1, height1).
        - target_size (tuple): A tuple containing the target image size (width2, height2).

        Returns:
        - tuple: A tuple containing the scaled bounding box coordinates.
        """
        x1, y1, x2, y2 = bbox
        if x1<0:
            x1=1
        if y1<0:
            y1=1
        width1, height1 = original_size
        width2, height2 = target_size

        # Scale the bounding box coordinates
        x1_scaled = x1 * width2 / width1
        y1_scaled = y1 * height2 / height1
        x2_scaled = x2 * width2 / width1
        y2_scaled = y2 * height2 / height1

        return (x1_scaled, y1_scaled, x2_scaled, y2_scaled)

    def adjust_gt_boxes(self, annotations):
        gt_boxes = []
        base_image_id = -1
        new_box = []
        for anno in annotations['annotations']:
            current_image_id = anno['image_id']
            if current_image_id!=base_image_id:
                gt_boxes.append(new_box)
                base_image_id = current_image_id
                new_box = []
            new_box.append(anno['bbox'])
        gt_boxes.append(new_box)
        gt_boxes.pop(0)
        # gt_boxes.reverse()
        return gt_boxes

    def transform_gt_to_dicts(self, gt_boxes,gt_classes):
        dict_list = []
        for boxes,labels in zip(gt_boxes,gt_classes):
            new_dict = {
                "boxes": Tensor(boxes),
                "labels": IntTensor(labels),
            }
            dict_list.append(new_dict)
        return dict_list


    def adjust_gt_classes(self, annotations):
        gt_classes = []
        base_image_id = -1
        new_box = []
        for anno in annotations['annotations']:
            current_image_id = anno['image_id']
            if current_image_id != base_image_id:
                gt_classes.append(new_box)
                base_image_id = current_image_id
                new_box = []
            new_box.append(anno['category_id'])
        gt_classes.append(new_box)
        gt_classes.pop(0)
        # gt_classes.reverse()
        return gt_classes

    def adjust_pred_boxes_classes(self, predictions):
        pred_boxes = []
        pred_classes = []
        for pred in predictions:
            adj_boxes = []
            for boxes,score in zip(pred['boxes'],pred['scores']):
                boxes = np.append(boxes,score)
                adj_boxes.append(boxes)
            pred_boxes.append(adj_boxes)
            pred_classes.append(pred['classes'])
        return pred_boxes,pred_classes

    def adjust_preds(self,predictions,images,annotations,file_names,mode):
        pred_copy = copy.deepcopy(predictions)
        for idx,pred in enumerate(pred_copy):
            anno_dict = [d for d in annotations['annotations'] if d.get('original_image_id') == file_names[idx]][0]
            image_id = anno_dict['image_id']
            image_info = [d for d in annotations['images'] if d.get('id') == image_id][0]
            image_width,image_height = image_info['width'],image_info['height']
            if mode =='SuperStore':
                image_size = (images[idx].shape[3],images[idx].shape[2])
            else:
                image_size = (images[idx].shape[2], images[idx].shape[3])
            boxes = [self.scale_bbox(box,image_size,(image_width,image_height)) for box in pred['boxes']]
            pred['boxes'] = Tensor(boxes)
            # pred['boxes'] = Tensor(pred['boxes'])
            pred['labels'] = IntTensor(pred['classes'])
            pred['scores'] = Tensor(pred['scores'])
        return pred_copy.tolist()

    def calculate_map(self,gt,preds):
        metric = MeanAveragePrecision(iou_type="bbox")

        # Update metric with predictions and respective ground truth
        metric.update(preds, gt)

        # Compute the results
        result = metric.compute()
        print(f"mAP:{round(result['map_50'].item(),3)}")

    def filter_annotations(self, annotations, file_names_fool,file_names):
        images_dropped = [string for string in file_names if string not in file_names_fool]
        filtered_list = []
        for d in annotations['annotations']:
            match_found = False
            for img_id in file_names_fool:
                if d.get('original_image_id') == img_id:
                    match_found = True
                    break
            if match_found:
                filtered_list.append(d)
        return filtered_list


