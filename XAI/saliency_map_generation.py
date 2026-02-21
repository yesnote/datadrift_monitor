import numpy as np
import torch
import cv2
import ttach as tta
import gc
from typing import Callable, List, Tuple
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from OD_models.Object_detector import Object_detection_model

"""
Helper module that generate a saliency map based on an object detection model layers output.
"""

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform, model_algorithm, img_size=None):
        self.model = model
        self.model_algorithm = model_algorithm
        if self.model_algorithm == "YOLOv5":
            self.gradients = dict()
            self.gradients['value'] = []
            self.activations = dict()
            self.activations['value'] = []
            self.form_hooks_yolo(target_layers)

        else:
            self.gradients = []
            self.activations = []
            for target_layer in target_layers:
                target_layer.register_forward_hook(self.save_activation)
                target_layer.register_backward_hook(self.save_gradient)

        self.reshape_transform = reshape_transform
        self.handles = []

    def form_hooks_yolo(self, target_layers):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_size = (640, 640)

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'].append(grad_output[0])
            return None

        def forward_hook(module, input, output):
            self.activations['value'].append(output)
            return None

        self.target_layers = target_layers
        [target_layer.register_forward_hook(forward_hook) for target_layer in self.target_layers]
        [target_layer.register_backward_hook(backward_hook) for target_layer in self.target_layers]

    def save_activation(self, module, input, output):
        activation = output
        try:
            self.activations.append(activation.cpu().detach())
        except:
            try:
                self.activations.append(activation[list(activation.keys())[len(activation)-1]])
            except:
                self.activations.append(activation[0][len(activation[0]) - 1])

    def save_gradient(self, module, input, output):
        self.gradients.append(output[0])

    def __call__(self, x,targets=None):
        if self.model_algorithm== "YOLOv5":
            return self.model(x)
        else:
            return self.model(x,targets)

    def release(self):
        for handle in self.handles:
            handle.remove()

class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True,
                 model_algorithm: str = "") -> None:

        self.model_algorithm = model_algorithm
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = self.init_transform_function(model_algorithm)
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform, model_algorithm=model_algorithm)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def init_transform_function(self, model_algorithm):
        if model_algorithm == 'Faster_RCNN':
            reshape_transform_function = self.fasterrcnn_reshape_transform
        else:
            reshape_transform_function = self.yolo_reshape_transform
        return reshape_transform_function

    def fasterrcnn_reshape_transform(self,x):
        target_size = x['pool'].size()[-2:]
        activations = []
        for key, value in x.items():
            activations.append(
                torch.nn.functional.interpolate(
                    torch.abs(value),
                    target_size,
                    mode='bilinear'))
        activations = torch.cat(activations, axis=1)
        return activations

    def yolo_reshape_transform(self,x):
        pass

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations
        cam = weighted_activations.sum(axis=1)
        # if eigen_smooth:
        #     if self.model_algorithm== 'YOLOv5':
        #         cam = get_2d_projection_new(weighted_activations)
        #     else:
        #         cam = get_2d_projection(weighted_activations)
        # else:
        #     cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:
        if self.model_algorithm!= "YOLOv5":
            self.activations_and_grads = ActivationsAndGradients(
                self.model, self.target_layers, self.reshape_transform, model_algorithm=self.model_algorithm)

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)
        self.init_gradient_and_activations()
        self.activations_and_grads.model.eval()
        if self.model_algorithm== "YOLOv5":
            predictions, logits,objectivness,x = self.activations_and_grads(input_tensor)
        else:
            outputs = self.activations_and_grads(input_tensor)
            # GPUtil.showUtilization()
            indices_to_remove = [0, 2, 4, 6, 8]
            self.activations_and_grads.activations = [tensor for i, tensor in
                                                      enumerate(self.activations_and_grads.activations) if
                                                      i not in indices_to_remove]
            predictions = Object_detection_model.extract_prediction_faster_rcnn_dict_format(outputs[0])

        if self.uses_gradients:
            self.model.zero_grad()
            if self.model_algorithm == "YOLOv5":
                if self.based_on_objectness:
                    saliency = objectivness[0]
                else:
                    # saliency = torch.stack([logit[cls] for logit, cls in zip(logits[0], predictions[1][0])])
                    saliency = torch.stack([max(logit) for logit in logits[0]])

                saliency = torch.sum(saliency)
                saliency.backward()
                self.activations_and_grads.gradients['value'].reverse()
            else:
                objectness_max = [objectness_feature_map.max(2)[0].max(2)[0] for objectness_feature_map in outputs[1]]
                objectness_loss = torch.sum(torch.cat(objectness_max))
                objectness_loss.backward()




        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)

        heat_map = self.aggregate_multi_layers(cam_per_layer)
        self.init_gradient_and_activations()
        del cam_per_layer
        gc.collect()
        # torch.cuda.empty_cache()
        return heat_map,predictions

    def compute_heat_map_activations(self,feature_map,objectness=True, logits=False):
        if objectness:
            feature_map = feature_map[0].max(dim=0)[0][:, :, 4]
        else:
            feature_map[:, :, :, :, 5:] = feature_map[:, :, :, :, 5:] * feature_map[:, :, :, :, 4:5]
            feature_map, j = feature_map[0].max(dim=0)[0].max(dim=2)
        if not logits:
            feature_map = feature_map.sigmoid()

        return feature_map.unsqueeze(0).detach().cpu().numpy()

    def init_gradient_and_activations(self):
        # del self.activations_and_grads.activations
        if self.model_algorithm == "YOLOv5":
            self.activations_and_grads.activations['value'] = []
            self.activations_and_grads.gradients['value'] = []
        else:
            self.activations_and_grads.activations = []
            self.activations_and_grads.gradients = []

    def remove_duplicates(self,tensor_list):
        # Convert tensors to bytes and store in a set to remove duplicates
        unique_tensors = set(tensor.numpy().tobytes() for tensor in tensor_list)

        # Convert bytes back to tensors
        unique_tensor_list = [torch.from_numpy(tensor) for tensor in unique_tensors]

        return unique_tensor_list

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def extract_activations_and_gradients(self):
        activations_list, grads_list = [], []
        if self.model_algorithm== "YOLOv5":
            activations_list = [activation.detach().cpu().numpy() for activation in
                               self.activations_and_grads.activations['value']]

            if self.uses_gradients:
                grads_list = self.activations_and_grads.gradients['value']
                grads_list = [g.cpu().data.numpy() for g in grads_list]
        else:
            activations_list = [a.cpu().data.numpy()
                                for a in self.activations_and_grads.activations]
            # activations_list = [a.detach().cpu().numpy()
            #                     for a in self.activations_and_grads.activations]
            activations_list = activations_list[len(activations_list)-2:]
            grads_list = [g.cpu().data.numpy()
                          for g in self.activations_and_grads.gradients]
            grads_list.reverse()
            grads_list = grads_list[len(grads_list)-2:]

        return activations_list,grads_list


    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list,grads_list = self.extract_activations_and_gradients()

        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []
        if self.model_algorithm== "YOLOv5":
            self.target_layers = self.activations_and_grads.target_layers
            iterate_over = self.target_layers
        # Loop over the saliency image from every layer
        else:
            iterate_over = activations_list
        for i in range(len(iterate_over)):
            if self.model_algorithm=="YOLOv5":
                target_layer = self.target_layers[i]
            else:
                target_layer = self.target_layers[0]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if self.uses_gradients:
                if i < len(grads_list):
                    layer_grads = grads_list[i]
            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
        del activations_list
        del cam
        del scaled
        return cam_per_target_layer

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        if self.model_algorithm== "Faster_RCNN":
            cam_per_target_layer = [cam/cam.max() for cam in cam_per_target_layer]
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        if self.model_algorithm== 'YOLOv5':
            result = np.amax(cam_per_target_layer, axis=1)
        else:
            # cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
            # result = np.mean(cam_per_target_layer, axis=1)
            result = np.amax(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False,
                 based_on_objectness = False) -> np.ndarray:

        self.based_on_objectness = based_on_objectness
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        cam,prediction_dict = self.forward(input_tensor,targets, eigen_smooth)
        if 'GradCAMPlusPlus' in str(type(self)) or 'XGradCAM' in str(type(self)):
            cam = 1 - cam
        if 'EigenGradCAM' in str(type(self)):
            cam = self.transform_eigen_grad_cam(cam)

        return cam,prediction_dict

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

    def transform_eigen_grad_cam(self, cam):
        if cam[0][0][0]==cam[0][0][1] and cam[0][0][1]==cam[0][0][2]:
            cam_default_value = cam[0][0][0]
        else:
            return cam
        cam[cam==cam_default_value] =0
        return cam


def process_activations(activation):
    number_of_anchors = 3
    output_per_anchor = activation.shape[1] // number_of_anchors
    bs, _, ny, nx = activation.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    activation = activation.view(bs, number_of_anchors, output_per_anchor, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    return activation


def find_yolo_layer(model, layer_name):
    """Find yolov5 layer to calculate GradCAM and GradCAM++

    Args:
        model: yolov5 model.
        layer_name (str): the name of layer with its hierarchical information.

    Return:
        target_layer: found layer
    """
    hierarchy = layer_name.split('.')
    target_layer = model.model._modules[hierarchy[0]]

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]
    return target_layer

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
            # img=1-img
        result.append(img)
    result = np.float32(result)

    return result

def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        # projection = np.abs(projection)
        projections.append(projection)
        del U
        del reshaped_activations
    return np.float32(projections)

def get_2d_projection_new(feature_map,objectness=True,logits = False):
    feature_map = torch.from_numpy(feature_map)
    feature_map = process_activations(feature_map)
    if objectness:
        feature_map = feature_map[0].max(dim=0)[0][:, :, 4]
    else:
        feature_map[:, :, :, :, 5:] = feature_map[:, :, :, :, 5:] * feature_map[:, :, :, :, 4:5]
        feature_map, j = feature_map[0].max(dim=0)[0].max(dim=2)
    if not logits:
        feature_map = feature_map.type(torch.float32)
        feature_map = feature_map.sigmoid()
    return feature_map.unsqueeze(0).detach().cpu().numpy()



class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None, model_algorithm=""):
        super(GradCAM,self).__init__(model, target_layers, use_cuda, reshape_transform,
                                     model_algorithm=model_algorithm)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))


class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None, model_algorithm=""):
        super(GradCAMPlusPlus, self).__init__(model, target_layers, use_cuda,
                                              reshape_transform, model_algorithm=model_algorithm)

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        # activations = np.expand_dims(activations,0) # Omer Added
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights

class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None, model_algorithm=""):
        super(EigenCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform,
                                       uses_gradients=True,
                                       model_algorithm=model_algorithm
                                       )

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        if self.model_algorithm== "YOLOv5":
            return get_2d_projection_new(activations)
        else:
            return get_2d_projection(activations)

class EigenGradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None, model_algorithm=""):
        super(EigenGradCAM, self).__init__(model, target_layers, use_cuda, reshape_transform,
                                           model_algorithm= model_algorithm)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        if self.model_algorithm == "YOLOv5":
            return get_2d_projection_new(grads * activations)
        else:
            return get_2d_projection(grads * activations)


class HiResCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None, model_algorithm=""):
        super(
            HiResCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            model_algorithm= model_algorithm
        )

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        elementwise_activations = grads * activations

        if eigen_smooth:
            print(
                "Warning: HiResCAM's faithfulness guarantees do not hold if smoothing is applied")
            cam = get_2d_projection(elementwise_activations)
        else:
            cam = elementwise_activations.sum(axis=1)
        return cam

class XGradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None, model_algorithm=""):
        super(XGradCAM,self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            model_algorithm= model_algorithm
        )

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = grads * activations / \
            (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))
        return weights

class GradCAMElementWise(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None, model_algorithm=""):
        super(
            GradCAMElementWise,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            model_algorithm=model_algorithm
        )

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        elementwise_activations = np.maximum(grads * activations, 0)

        if eigen_smooth:
            cam = get_2d_projection(elementwise_activations)
        else:
            cam = elementwise_activations.sum(axis=1)
        return cam

class LayerCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None,
            model_algorithm=""):
        super(
            LayerCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            model_algorithm=model_algorithm)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        spatial_weighted_activations = np.maximum(grads, 0) * activations

        if eigen_smooth:
            cam = get_2d_projection(spatial_weighted_activations)
        else:
            cam = spatial_weighted_activations.sum(axis=1)
        return cam

