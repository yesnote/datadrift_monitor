import torch

from mmcv.ops import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.utils import filter_scores_and_topk
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.retina_head import RetinaHead


@HEADS.register_module()
class RetinaHeadECPAL(RetinaHead):
    def __init__(self, **kwargs):
        super(RetinaHeadECPAL, self).__init__(**kwargs)

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        if score_factor_list[0] is None:
            with_score_factors = False
        else:
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_class_scores = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for cls_score, bbox_pred, score_factor, priors in zip(
                cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                class_scores = cls_score.sigmoid()
            else:
                class_scores = cls_score.softmax(-1)[:, :-1]

            results = filter_scores_and_topk(
                class_scores, cfg.score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred,
                    priors=priors,
                    class_scores=class_scores))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            class_scores = filtered_results['class_scores']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_class_scores.append(class_scores)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       mlvl_class_scores,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           mlvl_class_scores,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)
        assert len(mlvl_scores) == len(mlvl_class_scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)
        mlvl_class_scores = torch.cat(mlvl_class_scores)

        if mlvl_score_factors is not None:
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if not with_nms:
            raise NotImplementedError(
                'ECPAL feature export requires NMS so final detections can '
                'be matched to pre-NMS candidates.')

        if mlvl_bboxes.numel() == 0:
            det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
            det_labels = mlvl_labels
            empty_features = self._empty_detection_features(det_bboxes)
            return dict(
                det_bboxes=det_bboxes,
                det_labels=det_labels,
                ecpal_features=empty_features,
                ecpal_miss_features=det_bboxes.new_zeros((2, )))

        det_bboxes, keep_idxs = batched_nms(
            mlvl_bboxes, mlvl_scores, mlvl_labels, cfg.nms)
        keep_idxs = keep_idxs[:cfg.max_per_img]
        det_bboxes = det_bboxes[:cfg.max_per_img]
        det_labels = mlvl_labels[keep_idxs]
        det_class_scores = mlvl_class_scores[keep_idxs]

        ecpal_features = self._compute_detection_features(
            mlvl_bboxes, mlvl_labels, mlvl_class_scores,
            det_bboxes[:, :4], det_labels, det_class_scores, cfg)
        ecpal_miss_features = self._compute_miss_features(
            mlvl_bboxes, mlvl_class_scores, det_bboxes[:, :4], cfg)

        return dict(
            det_bboxes=det_bboxes,
            det_labels=det_labels,
            ecpal_features=ecpal_features,
            ecpal_miss_features=ecpal_miss_features)

    def _empty_detection_features(self, det_bboxes):
        return dict(
            p_max=det_bboxes.new_zeros((0, )),
            A_cls=det_bboxes.new_zeros((0, )),
            n_sup=det_bboxes.new_zeros((0, )),
            mu_iou=det_bboxes.new_zeros((0, )))

    def _compute_detection_features(self, candidate_bboxes, candidate_labels,
                                    candidate_class_scores, det_bboxes,
                                    det_labels, det_class_scores, cfg):
        if det_bboxes.numel() == 0:
            return self._empty_detection_features(det_bboxes)

        p_max = det_class_scores.max(dim=1)[0]
        if candidate_bboxes.numel() == 0:
            return dict(
                p_max=p_max,
                A_cls=det_bboxes.new_zeros((det_bboxes.shape[0], )),
                n_sup=det_bboxes.new_zeros((det_bboxes.shape[0], )),
                mu_iou=det_bboxes.new_zeros((det_bboxes.shape[0], )))

        support_iou_thr = float(cfg.get('support_iou_thr', 0.5))
        box_equal_tol = float(cfg.get('support_box_equal_tol', 1e-6))

        overlaps = bbox_overlaps(candidate_bboxes, det_bboxes)
        max_overlaps, assigned_det_inds = overlaps.max(dim=1)
        same_box = (det_bboxes[:, None, :] - candidate_bboxes[None, :, :])
        same_box = same_box.abs().max(dim=2)[0].transpose(0, 1) <= box_equal_tol

        det_indices = torch.arange(
            det_bboxes.shape[0], device=det_bboxes.device)
        support_mask = (
            (assigned_det_inds[:, None] == det_indices[None, :])
            & (max_overlaps[:, None] >= support_iou_thr)
            & (~same_box)
        ).transpose(0, 1)
        support_counts = support_mask.sum(dim=1).to(dtype=det_bboxes.dtype)

        label_matches = det_labels[:, None] == candidate_labels[None, :]
        same_class_counts = (support_mask & label_matches).sum(dim=1)
        same_class_counts = same_class_counts.to(dtype=det_bboxes.dtype)

        denom = support_counts.clamp(min=1.0)
        A_cls = same_class_counts / denom
        det_candidate_overlaps = overlaps.transpose(0, 1)
        mu_iou = (
            det_candidate_overlaps * support_mask.to(dtype=overlaps.dtype)
        ).sum(dim=1)
        mu_iou = mu_iou / denom
        return dict(
            p_max=p_max,
            A_cls=A_cls,
            n_sup=support_counts,
            mu_iou=mu_iou)

    def _compute_miss_features(self, candidate_bboxes, candidate_class_scores,
                               det_bboxes, cfg):
        if candidate_bboxes.numel() == 0:
            return det_bboxes.new_zeros((2, ))

        candidate_scores = candidate_class_scores.max(dim=1)[0]
        if det_bboxes.numel() == 0:
            rho = candidate_scores.new_zeros((candidate_bboxes.shape[0], ))
        else:
            rho = bbox_overlaps(candidate_bboxes, det_bboxes).max(dim=1)[0]
        residual = 1.0 - rho
        R_amt = residual.sum()
        eps = float(cfg.get('miss_eps', 1e-12))
        R_prob = (residual * candidate_scores).sum() / (R_amt + eps)
        return torch.stack([R_amt, R_prob])
