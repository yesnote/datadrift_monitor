import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import numpy as np
import torch
import os
import GPUtil
from tqdm import tqdm
from XAI.saliency_map_generation import GradCAM,GradCAMPlusPlus,LayerCAM,GradCAMElementWise,HiResCAM,XGradCAM,EigenGradCAM,EigenCAM,find_yolo_layer
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

"""
This module is used to generate saliency maps and save them to a specific folder.
"""

# This module is inspired from Pytorch grad cam project
# https://github.com/jacobgil/pytorch-grad-cam
# MIT License
#
# Copyright (c) 2021 Jacob Gildenblat
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



class Explainer():
    def __init__(self, object_detection_model, XAI_method, model_algorithm):
        super().__init__()
        self.object_detection_model = object_detection_model
        self.model_algorithm = model_algorithm
        self.target_layers = self.set_target_layer()
        self.XAI_method = self.init_XAI_method(XAI_method)

    def set_target_layer(self):
        if self.model_algorithm == 'Faster_RCNN' or self.model_algorithm == 'SSD300':
            return [self.object_detection_model.model.rpn._modules['head']._modules['cls_logits']]
        else:
            target_layers = ['model.24.m.0','model.24.m.1','model.24.m.2']
            return [find_yolo_layer(self.object_detection_model.model,target_layer) for target_layer in target_layers]


    def init_XAI_method(self,XAI_method):
        XAI_dict = {
            'GradCAM': GradCAM,
            'GradCAM++': GradCAMPlusPlus,
            'EigenCAM': EigenCAM,
            'EigenGradCAM': EigenGradCAM,
            'GradCAMElementWise': GradCAMElementWise,
            'XGradCAM': XGradCAM,
            'HiresCAM': HiResCAM,
            'LayerCAM': LayerCAM
        }
        return XAI_dict.get(XAI_method, "No info available")(self.object_detection_model.model,
                                                             self.target_layers,
                                                             use_cuda=True,
                                                             model_algorithm=self.model_algorithm)

    def apply_explanations(self, images, saliency_based_on_objectness=True,eigen_smooth = False):
        """
        Wrapper function that applies saliency maps to multiple images.
        :param images: required. numpy array of images (in numpy array format).
        :param prediction_dicts: required. numpy array of dictionaries containing detections.
        Each dictionary contains bounding boxes, classes and labels.
        :return: Numpy array of heatmaps.
        """
        heatmaps = []
        prediction_dicts = []
        GPUtil.showUtilization()
        for image in tqdm(images, desc="Explain images"):
            if self.model_algorithm== 'Faster_RCNN':
                saliency_map,predictions = self.explain(image,eigen_smooth)
            else:
                saliency_map,predictions = self.explain(image.requires_grad_(requires_grad=True),saliency_based_on_objectness,eigen_smooth)
            heatmaps.append(saliency_map)
            prediction_dicts.append(predictions)
            del saliency_map
            # GPUtil.showUtilization()
        return heatmaps,prediction_dicts


    def explain(self,image,saliency_based_on_objectness = False,eigen_smooth=False):
        """
        Function that produce saliency map for a single frame.
        :param image: required. Image in a numpy format.
        :return: Saliency map as numpy array of a single frame.
        """
        if self.model_algorithm== "YOLOv5":
            image = image.to(self.object_detection_model.device)
            heatmap_cam,predictions = self.XAI_method(image, eigen_smooth=eigen_smooth, based_on_objectness=saliency_based_on_objectness)
        else:
            heatmap_cam,predictions = self.XAI_method(image, eigen_smooth=eigen_smooth)
        heatmap_cam = heatmap_cam[0, :]
        return heatmap_cam,predictions

    def visualize(self,original_images,heatmap_cams,prediction_dicts,output_path,bbox_renormalize=False,save=True,file_names=None):
        """
        Wrapper function that visualize multiple saliency maps (save the image using a given output path).
        :param original_images: required. numpy array of images (in numpy array format).
        :param heatmap_cams: required. numpy array of saliency maps (in numpy array format).
        :param prediction_dicts: required. numpy array of dictionaries containing detections.
        Each dictionary contains bounding boxes, classes and labels.
        :return: Numpy array of heatmaps.
        :param output_path: required. the output folder path which the saliency maps will be saved in.
        :return: --
        """
        if not save:
            self.explanations = []
        if not file_names:
            file_names = np.arange(len(original_images))
        for idx,(original_image,heatmap_cam,prediction_dict,file_name) in enumerate(zip(original_images,heatmap_cams,prediction_dicts,file_names)):
            curr_heat_map_output_path = os.path.join(output_path,f'{file_name}.jpg')
            self.visualize_explanation(original_image,heatmap_cam,prediction_dict['boxes'],prediction_dict['classes'],
                                                                    prediction_dict['labels'],curr_heat_map_output_path,
                                                                    bbox_renormalize,save)
        if not save:
            return self.explanations

    def visualize_explanation(self,original_image,heatmap_cam,boxes, classes,labels,output_path,bbox_renormalize,save):
        """
        Save a single heatmap using a given output path.
        :param original_image: required. Image in a numpy format.
        :param heatmap_cam: required. Saliency map in a numpy format.
        :param boxes: required, numpy array of lists representing the detection bounding box.
        :param classes: required, numpy array of strings representing the detection classes.
        :param labels: required, numpy array of ints representing the detection labels.
        :param output_path: required. the output file path which the saliency map will be saved in.
        :return: --
        """
        original_image = torch.squeeze(original_image).detach().cpu().numpy()
        original_image = np.transpose(original_image,(1,2,0))
        original_image = np.clip(original_image, 0.00, 1.00)
        if bbox_renormalize:
            heatmap_cam = self.renormalize_cam_in_bounding_boxes(boxes,heatmap_cam)
        cam_image = show_cam_on_image(original_image, heatmap_cam, use_rgb=True)
        image_with_bounding_boxes = self.object_detection_model.draw_boxes(boxes, labels, classes, cam_image)
        image_with_bounding_boxes = cv2.cvtColor(image_with_bounding_boxes, cv2.COLOR_BGR2RGB)
        if save:
            cv2.imwrite(output_path, image_with_bounding_boxes)
        else:
            return self.explanations.append(image_with_bounding_boxes)

    def renormalize_cam_in_bounding_boxes(self, boxes, heatmap_cam):
        renormalized_cam = np.zeros(heatmap_cam.shape, dtype=np.float32)
        images = []
        for x1, y1, x2, y2 in boxes:
            img = renormalized_cam * 0
            img[y1:y2, x1:x2] = scale_cam_image(heatmap_cam[y1:y2, x1:x2].copy())
            images.append(img)

        renormalized_cam = np.max(np.float32(images), axis=0)
        renormalized_cam = scale_cam_image(renormalized_cam)
        return renormalized_cam
