import unittest

from methods.pal.guide import compute_cwie, compute_rcdi, compute_rcsp
from methods.pal.acquisition import select_full_pal_images


class PalGuideScoreTest(unittest.TestCase):
    def test_cwie_rcdi_and_rcsp_terms(self):
        detections = [
            {'image_id': 1, 'category_id': 1, 'class_scores': [0.8, 0.2]},
            {'image_id': 1, 'category_id': 2, 'class_scores': [0.5, 0.5]},
        ]
        class_weights = {1: 0.25, 2: 0.75}

        cwie = compute_cwie(detections, class_weights)
        rcdi = compute_rcdi([1, 2], class_weights)
        rcsp = compute_rcsp([1, 2, 3], {
            1: [1.0, 0.0],
            2: [0.0, 1.0],
            3: [1.0, 0.0],
        })

        self.assertGreater(cwie, 0.0)
        self.assertAlmostEqual(rcdi, 1.0)
        self.assertAlmostEqual(rcsp[1], 1.0)
        self.assertAlmostEqual(rcsp[2], 1.0)
        self.assertAlmostEqual(rcsp[3], 0.0)


class PalFullSamplerTest(unittest.TestCase):
    def test_full_pal_selects_unique_images_with_guide_scores(self):
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
            {
                'image_id': 3, 'category_id': 1, 'bbox': [0, 0, 9, 9],
                'score': 0.55, 'pre_nms_count': 10, 'class_scores': [0.7, 0.3],
            },
            {
                'image_id': 4, 'category_id': 2, 'bbox': [0, 0, 9, 9],
                'score': 0.50, 'pre_nms_count': 11, 'class_scores': [0.2, 0.8],
            },
            {
                'image_id': 5, 'category_id': 1, 'bbox': [0, 0, 9, 9],
                'score': 0.10, 'pre_nms_count': 2, 'class_scores': [0.6, 0.4],
            },
        ]

        result = select_full_pal_images(
            labeled_pool=labeled_pool,
            unlabeled_pool=unlabeled_pool,
            labeled_detections=labeled_detections,
            unlabeled_detections=unlabeled_detections,
            budget=2,
            iou_threshold=0.5,
            seed=0,
            embedding_source='detection',
        )

        selected = result['selected_image_ids']
        self.assertEqual(result['mode'], 'full')
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(set(selected)), 2)
        self.assertTrue(set(selected).issubset({3, 4, 5}))
        self.assertGreater(len(result['candidate_scores']), 0)
        self.assertIn('pal_score', result['candidate_scores'][0])
        self.assertIn('guide_cwie', result['candidate_scores'][0])
        self.assertIn('guide_rcdi', result['candidate_scores'][0])
        self.assertIn('guide_rcsp', result['candidate_scores'][0])


if __name__ == '__main__':
    unittest.main()
