import torch

def compute_dil(cam, pred_boxes):
    CL = cam.sum()

    bg = torch.ones_like(cam)
    for x1, y1, x2, y2 in pred_boxes:
        bg[..., int(y1):int(y2), int(x1):int(x2)] = 0

    BL = (cam * bg).sum()
    return (BL / (CL + 1e-6)).item()
