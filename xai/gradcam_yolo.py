import torch
import torch.nn.functional as F

class YOLOGradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.activations = []
        self.gradients = []

        for l in self.target_layers:
            l.register_forward_hook(self._forward_hook)
            l.register_backward_hook(self._backward_hook)

    def _forward_hook(self, m, i, o):
        self.activations.append(o)

    def _backward_hook(self, m, gi, go):
        self.gradients.append(go[0])

    def saliency(self, det_raw):
        """
        Eq.(1): sum(objectness)
        """
        score = 0.0
        for out in det_raw:
            score += out.sum()

        self.model.zero_grad()
        score.backward(retain_graph=True)

        cams = []
        for act, grad in zip(self.activations, self.gradients):
            w = grad.mean(dim=(2, 3), keepdim=True)
            cam = (w * act).sum(dim=1, keepdim=True)
            cams.append(F.relu(cam))

        cam = sum(F.interpolate(c, size=cams[0].shape[-2:], mode="bilinear", align_corners=False)
                  for c in cams)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam
