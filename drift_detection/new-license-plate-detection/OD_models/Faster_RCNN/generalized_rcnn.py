"""
Implements the Generalized R-CNN framework
"""

import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
import GPUtil
import gc
import torch
from torch import nn, Tensor
from torchvision.utils import _log_api_usage_once
import torchvision.transforms as T
import imutils
from torchvision.transforms.functional import pad
import torchvision.transforms.functional as F


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        # self.activations_gradients = ActivationsAndGradients(self.rpn,
        #                                                      [self.rpn._modules['head']._modules['cls_logits']], )
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        # if saliency_map is not None:
        #     images,_ = self.mask_features(saliency_map,images,5)
        #     self.plot_image(images[0]*255)

        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        if isinstance(images, list):
            images = torch.cat(images)
        if len(images.shape)<4:
            images = images.unsqueeze(0)
        if images.shape[1]!=3:
            images = torch.permute(images,(0,3,1,2))
        original_images = torch.clone(images)
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # if saliency_map is not None:
        #     for idx, feature_key in enumerate(features.keys()):
        #         self.plot_feature_map(features[feature_key],idx)
        #         # features[feature_key],_ = self.mask_features(saliency_map,features[feature_key],idx)
        proposals, proposal_losses, objectness = self.rpn(images, features, targets)
        self.rpn.zero_grad()
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes,
                                                original_image_sizes)  # type: ignore[operator]
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        del proposals
        del proposal_losses
        del features
        gc.collect()
        # torch.cuda.empty_cache()
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections),objectness,original_images

    def normalize_3d_image(self,image):
        min_val = np.min(image)
        max_val = np.max(image)

        # If the range is already within [0, 255], return the original image
        if min_val >= 0 and max_val <= 255:
            return image

        # Shift the minimum value to 0
        image = image - min_val

        # Scale the image to the range [0, 255]
        image = image / (max_val - min_val) * 255

        # Clip values to ensure they are within the range [0, 255]
        image = np.clip(image, 0, 255)

        # Convert the image to the integer data type
        image = image.astype(np.uint8)

        return image

    def mask_features(self, saliency_map, image, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        saliency_map = imutils.resize(saliency_map, width=image.shape[3], height=image.shape[2])
        saliency_map = imutils.resize(saliency_map, height=image.shape[2])
        saliency_map_tensor = torch.from_numpy(saliency_map).float().to(device)
        if saliency_map.shape[0] != image.shape[2]:
            saliency_map_tensor = pad(saliency_map_tensor, (0, image.shape[2] - saliency_map.shape[0], 0, 0),
                                      fill=1.0)
        if saliency_map.shape[1] != image.shape[3]:
            saliency_map_tensor = pad(saliency_map_tensor, (0, 0, image.shape[3] - saliency_map.shape[1], 0),
                                      fill=1.0)
        # Create a mask based on the saliency map
        mask = (saliency_map_tensor < 0.6).unsqueeze(0).float()
        mode = '5'
        if mode == '1':
            new_feature_map = self.hide_patch(image, 1-mask)
        elif mode == '2':
            new_feature_map = self.increase_features_values_by_patch(image, mask)
        elif mode== '3':
            new_feature_map = self.decrease_patch_values_by_features(image, mask)
        elif mode=='4':
            new_feature_map = self.increase_features_values_by_patch(image, mask)
            new_feature_map = self.decrease_patch_values_by_features(new_feature_map, mask)
        elif mode == '5':
            # blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges = self.edge_detection(image)
            transform = T.GaussianBlur(kernel_size=(7, 13), sigma=(2.0, 3.0))
            blurred = transform(image)
            new_feature_map = blurred * mask
            new_feature_map = torch.add(image*(1-mask),new_feature_map)
        elif mode == '6':
            new_feature_map = saliency_map_tensor * image
        else:
            new_feature_map = image


        self.plot_feature_map(new_feature_map, idx)
        return new_feature_map, mask

    def hide_patch(self,image,mask):
        return image * mask

    def increase_features_values_by_patch(self,image,mask):
        masked_pixel_sum = torch.sum(image * mask, dim=(1, 2))
        masked_pixel_count = torch.sum(mask, dim=(1, 2))
        # num_elements = int(0.1 * masked_pixel_sum.numel())
        # # Find the top 10% maximum values
        # top_values, _ = torch.topk(masked_pixel_sum, num_elements)
        # average =torch.mean(top_values)
        # masked_pixel_sum_new = average.repeat(masked_pixel_sum.shape[1]).unsqueeze(0)
        average_value = masked_pixel_sum / masked_pixel_count

        # Broadcast the average value to the unmasked pixels
        average = (average_value[:, None, None] * (mask))
        modified_image_tensor = torch.add(image,average)
        return modified_image_tensor

    def decrease_patch_values_by_features(self,image,mask):
        mask = 1 - mask
        masked_pixel_sum = torch.sum(image * mask, dim=(1, 2))
        masked_pixel_count = torch.sum(mask, dim=(1, 2))

        # num_elements = int(0.1 * masked_pixel_sum.numel())
        # # Find the top 10% maximum values
        # top_values, _ = torch.topk(masked_pixel_sum, num_elements)
        # average = torch.mean(top_values)
        # masked_pixel_sum_new = average.repeat(masked_pixel_sum.shape[1]).unsqueeze(0)

        average_value = masked_pixel_sum / masked_pixel_count

        # compute average values
        average_pixels = average_value[:, None, None] * (1-mask)
        # Broadcast the average value to the masked pixels
        modified_image_tensor = torch.add(image, average_pixels)
        return modified_image_tensor

    def edge_detection(self,image):
        canny = CannyFilter()
        blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges = canny(image)
        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges


    def plot_feature_map(self,new_feature_map,idx):
        mask_image_numpy = new_feature_map.detach().cpu().numpy()
        mask_image_numpy = mask_image_numpy.mean(1)
        mask_image_numpy = np.transpose(mask_image_numpy, (1, 2, 0))
        mask_image_numpy = self.normalize_3d_image(mask_image_numpy)


    def plot_image(self, image):
        mask_image_numpy = image.detach().cpu().numpy()
        mask_image_numpy = mask_image_numpy.squeeze()
        mask_image_numpy = np.transpose(mask_image_numpy, (1, 2, 0))
        mask_image_numpy = self.normalize_3d_image(mask_image_numpy)
        mask_image_numpy = cv2.cvtColor(mask_image_numpy, cv2.COLOR_BGR2RGB)



class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False,
                                         device=self.device)
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False,
                                        device=self.device)
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)


        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False,
                                        device=self.device)
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)


        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False,
                                            device=self.device)
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False,
                                    device=self.device)
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)


    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1


        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges

def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
        kernel_angle = kernel_angle * is_diag  # because of the interpolation
        thin_kernels.append(kernel_angle)
    return thin_kernels

def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D

def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D