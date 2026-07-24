import json
import tempfile
import unittest
from pathlib import Path

from methods.common.coco_pool import load_coco_pool, read_coco_json, write_coco_pool_split
from methods.common.detections import load_detection_records
from methods.common.results import acquisition_result
from methods.common.selection import fill_to_budget


class CommonSelectionTest(unittest.TestCase):
    def test_fill_to_budget_preserves_selected_and_refills_deterministically(self):
        selected = [3, 1, 3]
        candidates = [1, 2, 3, 4, 5]

        output = fill_to_budget(selected, candidates, budget=4, seed=0)

        self.assertEqual(output[:2], [3, 1])
        self.assertEqual(len(output), 4)
        self.assertEqual(len(set(output)), 4)
        self.assertTrue(set(output).issubset(set(candidates)))


class CommonCocoPoolTest(unittest.TestCase):
    def test_load_coco_pool_accepts_path_or_dict(self):
        data = {'images': [{'id': 1}], 'categories': []}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'pool.json'
            path.write_text(json.dumps(data), encoding='utf-8')

            self.assertEqual(load_coco_pool(path), data)
            self.assertIs(load_coco_pool(data), data)

    def test_write_coco_pool_split_controls_labeled_annotations(self):
        oracle = {
            'images': [{'id': 2}, {'id': 1}, {'id': 3}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 1},
                {'id': 2, 'image_id': 2, 'category_id': 1},
            ],
            'categories': [{'id': 1, 'name': 'a'}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            labeled_path = Path(tmpdir) / 'labeled.json'
            unlabeled_path = Path(tmpdir) / 'unlabeled.json'

            write_coco_pool_split(
                oracle,
                labeled_image_ids=[1, 2],
                unlabeled_image_ids=[3],
                out_labeled_json=labeled_path,
                out_unlabeled_json=unlabeled_path,
                labeled_include_annotations=False,
            )

            labeled = read_coco_json(labeled_path)
            unlabeled = read_coco_json(unlabeled_path)

        self.assertEqual([image['id'] for image in labeled['images']], [1, 2])
        self.assertNotIn('annotations', labeled)
        self.assertEqual([image['id'] for image in unlabeled['images']], [3])


class CommonDetectionTest(unittest.TestCase):
    def test_load_detection_records_validates_required_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'detections.json'
            path.write_text(
                json.dumps({'results': [{'image_id': 1, 'score': 0.5}]}),
                encoding='utf-8',
            )

            records = load_detection_records(path, required_keys=('image_id', 'score'))

        self.assertEqual(records, [{'image_id': 1, 'score': 0.5}])


class CommonResultsTest(unittest.TestCase):
    def test_acquisition_result_builds_common_envelope(self):
        result = acquisition_result(
            method='pal',
            stage='guide',
            round_index=1,
            budget=2,
            selected_image_ids=[10, 11],
            inputs={'unlabeled_pool_json': 'u.json'},
            outputs={'labeled_pool_json': 'l.json'},
            metrics={'matched_detection_count': 4},
        )

        self.assertEqual(result['selected_count'], 2)
        self.assertEqual(result['inputs']['unlabeled_pool_json'], 'u.json')
        self.assertEqual(result['metrics']['matched_detection_count'], 4)


if __name__ == '__main__':
    unittest.main()
