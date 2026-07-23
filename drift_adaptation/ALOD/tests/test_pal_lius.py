import unittest

from methods.common.matching import match_detections_to_ground_truth
from methods.pal.acquisition import select_lius_images


class MatchingTest(unittest.TestCase):
    def test_greedy_tp_fp_matching(self):
        ground_truth = {
            'annotations': [
                {'id': 1, 'image_id': 10, 'category_id': 1, 'bbox': [0, 0, 10, 10]},
            ]
        }
        detections = [
            {'image_id': 10, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'score': 0.9},
            {'image_id': 10, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'score': 0.8},
            {'image_id': 10, 'category_id': 2, 'bbox': [0, 0, 10, 10], 'score': 0.7},
        ]

        matched = match_detections_to_ground_truth(detections, ground_truth, iou_threshold=0.5)

        self.assertEqual([record['target'] for record in matched], [1, 0, 0])
        self.assertEqual(matched[0]['matched_gt_id'], 1)

    def test_spatial_no_match_is_fp(self):
        ground_truth = {
            'annotations': [
                {'id': 1, 'image_id': 10, 'category_id': 1, 'bbox': [0, 0, 10, 10]},
            ]
        }
        detections = [
            {'image_id': 10, 'category_id': 1, 'bbox': [30, 30, 10, 10], 'score': 0.9},
        ]

        matched = match_detections_to_ground_truth(detections, ground_truth, iou_threshold=0.5)

        self.assertEqual(matched[0]['target'], 0)
        self.assertIsNone(matched[0]['matched_gt_id'])


class PalLiusSamplerTest(unittest.TestCase):
    def test_selects_unique_unlabeled_images(self):
        labeled_pool = {
            'images': [{'id': 1}, {'id': 2}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10]},
                {'id': 2, 'image_id': 2, 'category_id': 2, 'bbox': [20, 20, 8, 8]},
            ],
            'categories': [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}],
        }
        unlabeled_pool = {
            'images': [{'id': 3}, {'id': 4}, {'id': 5}],
            'categories': labeled_pool['categories'],
        }
        labeled_detections = [
            {'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'score': 0.95, 'pre_nms_count': 3},
            {'image_id': 1, 'category_id': 1, 'bbox': [50, 50, 5, 5], 'score': 0.30, 'pre_nms_count': 30},
            {'image_id': 2, 'category_id': 2, 'bbox': [20, 20, 8, 8], 'score': 0.90, 'pre_nms_count': 4},
            {'image_id': 2, 'category_id': 2, 'bbox': [60, 60, 5, 5], 'score': 0.25, 'pre_nms_count': 25},
        ]
        unlabeled_detections = [
            {'image_id': 3, 'category_id': 1, 'bbox': [0, 0, 9, 9], 'score': 0.55, 'pre_nms_count': 10},
            {'image_id': 4, 'category_id': 2, 'bbox': [0, 0, 9, 9], 'score': 0.50, 'pre_nms_count': 11},
            {'image_id': 5, 'category_id': 1, 'bbox': [0, 0, 9, 9], 'score': 0.10, 'pre_nms_count': 2},
        ]

        result = select_lius_images(
            labeled_pool=labeled_pool,
            unlabeled_pool=unlabeled_pool,
            labeled_detections=labeled_detections,
            unlabeled_detections=unlabeled_detections,
            budget=2,
            iou_threshold=0.5,
            seed=0,
        )

        selected = result['selected_image_ids']
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(set(selected)), 2)
        self.assertTrue(set(selected).issubset({3, 4, 5}))
        self.assertEqual(result['matched_detection_count'], 4)
        self.assertEqual(result['scored_detection_count'], 3)


if __name__ == '__main__':
    unittest.main()
