import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class ECPALFeatureCocoDataset(CocoDataset):
    def _det2json(self, results):
        """Convert ECPAL compact feature results to image-level json."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            if not isinstance(result, dict):
                raise TypeError(
                    'ECPALFeatureCocoDataset expects dict results, got %r' %
                    (type(result), ))

            bboxes = np.asarray(result['det_bboxes'])
            labels = np.asarray(result['det_labels'])
            features = result['ecpal_features']
            miss_features = np.asarray(result['ecpal_miss_features'])

            final_detections = []
            for i in range(bboxes.shape[0]):
                data = dict()
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['category_id'] = self.cat_ids[int(labels[i])]
                data['score'] = float(bboxes[i][4])
                data['p_max'] = float(features['p_max'][i])
                data['A_cls'] = float(features['A_cls'][i])
                data['n_sup'] = float(features['n_sup'][i])
                data['mu_iou'] = float(features['mu_iou'][i])
                final_detections.append(data)

            json_results.append(
                dict(
                    image_id=img_id,
                    final_detections=final_detections,
                    miss_features=dict(
                        R_amt=float(miss_features[0]),
                        R_prob=float(miss_features[1]))))
        return json_results

    def results2json(self, results, outfile_prefix):
        result_files = dict()
        if isinstance(results[0], dict):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.json'
            mmcv.dump(json_results, result_files['bbox'])
        else:
            raise TypeError('invalid type of ECPAL feature results')
        return result_files
