import torch
import torch.nn.functional as F


class YOLOGradCAM:
    """
    Grad-CAM for DiL (YOLO26).
    - Hook: backbone / neck conv feature
    - Target: sum of objectness outputs (Mo)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        self.gradient = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activation = out

        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def _aggregate_objectness(self, det_raw):
        """
        Implements Eq.(1) in DiL paper:
        sum of all objectness scores
        """
        score = 0.0

        if isinstance(det_raw, (list, tuple)):
            for out in det_raw:
                # out shape depends on head
                # YOLO26 one-to-one: (B, 300, 6)
                # assume objectness at index 4
                score += out[..., 4].sum()
        else:
            score = det_raw[..., 4].sum()

        return score

    def saliency(self, det_raw, input_shape):
        """
        Returns:
            cam: (1, 1, H, W)
        """
        # 1. aggregate objectness
        score = self._aggregate_objectness(det_raw)

        # 2. backward
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # 3. Grad-CAM
        grad = self.gradient          # (B, C, h, w)
        act = self.activation         # (B, C, h, w)

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 4. upsample to input size
        cam = F.interpolate(
            cam,
            size=input_shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # 5. normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam
