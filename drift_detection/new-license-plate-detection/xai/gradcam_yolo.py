import torch
import torch.nn.functional as F


class YOLOGradCAM:
    """
    Grad-CAM for DiL (YOLO26).
    - Hook: backbone / neck conv feature
    - Target: sum of objectness outputs
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
        Eq.(1) in DiL paper: sum of objectness scores
        Expected YOLO26 one-to-one head:
          det_raw: (B, N, 6)  -> [..., 4] = objectness
        """
        assert det_raw.dim() == 3, \
            f"Expected det_raw dim=3, got {det_raw.shape}"

        assert det_raw.size(-1) >= 5, \
            f"Expected last dim >=5 (bbox+obj+cls), got {det_raw.shape}"

        # objectness channel index = 4
        return det_raw[..., 4].sum()

    def saliency(self, det_raw, input_shape):
        """
        Returns:
            cam: (1, 1, H, W)
        """
        # reset states
        self.activation = None
        self.gradient = None

        # 1. aggregate objectness
        score = self._aggregate_objectness(det_raw)

        # 2. backward
        self.model.zero_grad()
        score.backward(retain_graph=True)

        assert self.activation is not None, "Activation not captured"
        assert self.gradient is not None, "Gradient not captured"

        # 3. Grad-CAM
        grad = self.gradient          # (B, C, h, w)
        act = self.activation         # (B, C, h, w)

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 4. upsample
        cam = F.interpolate(
            cam,
            size=input_shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # 5. normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam
