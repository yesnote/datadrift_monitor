from utils.bbox import iou

def is_missed_detection(gt_boxes, pred_boxes, iou_thresh=0.5):
    if len(gt_boxes) == 0:
        return False

    for gt in gt_boxes:
        matched = False
        for pr in pred_boxes:
            if iou(gt, pr) >= iou_thresh:
                matched = True
                break
        if not matched:
            return True
    return False
