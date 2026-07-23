import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from methods.pal.embeddings import (
    DetectionEmbeddingBackend,
    build_image_embeddings,
    read_embedding_cache,
    read_embeddings_json,
    read_embeddings_npy,
    stack_embeddings,
    write_embedding_cache,
    write_embeddings_json,
    write_embeddings_npy,
)
from methods.pal.vit_embeddings import read_coco_image_paths


class PalDetectionEmbeddingTest(unittest.TestCase):
    def test_detection_backend_is_deterministic_and_includes_empty_images(self):
        detections = [
            {
                'image_id': 2,
                'category_id': 1,
                'score': 0.8,
                'pre_nms_count': 4,
                'class_scores': [0.2, 0.8],
            },
            {
                'image_id': 1,
                'category_id': 2,
                'score': 0.5,
                'pre_nms_count': 2,
                'class_scores': [0.7, 0.3],
            },
            {
                'image_id': 2,
                'category_id': 1,
                'score': 0.4,
                'pre_nms_count': 8,
                'class_scores': [0.6, 0.4],
            },
        ]
        backend = DetectionEmbeddingBackend(num_classes=2)

        first = backend.embed_records(detections, image_ids=[1, 2, 3])
        second = backend.embed_records(detections, image_ids=[1, 2, 3])

        self.assertEqual(list(first.keys()), [1, 2, 3])
        self.assertEqual(first[1].shape, (9, ))
        np.testing.assert_allclose(first[1], second[1])
        np.testing.assert_allclose(first[2], second[2])
        np.testing.assert_allclose(first[3], np.zeros(9, dtype=np.float32))

    def test_detection_backend_falls_back_to_category_one_hot(self):
        detections = [
            {'image_id': 1, 'category_id': 5, 'score': 0.9, 'pre_nms_count': 3},
            {'image_id': 2, 'category_id': 7, 'score': 0.4, 'pre_nms_count': 5},
        ]

        embeddings = build_image_embeddings(detections, backend='fallback')

        self.assertEqual(sorted(embeddings), [1, 2])
        self.assertEqual(embeddings[1].shape, (9, ))
        self.assertGreater(float(np.linalg.norm(embeddings[1])), 0.0)
        self.assertGreater(float(np.linalg.norm(embeddings[2])), 0.0)

    def test_stack_embeddings_uses_requested_image_order(self):
        embeddings = {
            1: np.asarray([1.0, 0.0], dtype=np.float32),
            2: np.asarray([0.0, 1.0], dtype=np.float32),
        }

        image_ids, matrix = stack_embeddings(embeddings, image_ids=[2, 1])

        self.assertEqual(image_ids, [2, 1])
        np.testing.assert_allclose(matrix, [[0.0, 1.0], [1.0, 0.0]])


class PalEmbeddingCacheTest(unittest.TestCase):
    def test_json_cache_roundtrip_preserves_image_ids_and_values(self):
        embeddings = {
            1: np.asarray([1.0, 0.0], dtype=np.float32),
            'two': np.asarray([0.5, 0.5], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'embeddings.json'

            write_embeddings_json(embeddings, path)
            loaded = read_embeddings_json(path)

        self.assertEqual(set(loaded), {1, 'two'})
        np.testing.assert_allclose(loaded[1], embeddings[1])
        np.testing.assert_allclose(loaded['two'], embeddings['two'])

    def test_npy_cache_roundtrip_preserves_image_ids_and_values(self):
        embeddings = {
            1: np.asarray([1.0, 0.0, 0.5], dtype=np.float32),
            3: np.asarray([0.2, 0.3, 0.4], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'embeddings.npy'

            write_embeddings_npy(embeddings, path)
            loaded = read_embeddings_npy(path)

        self.assertEqual(set(loaded), {1, 3})
        np.testing.assert_allclose(loaded[1], embeddings[1])
        np.testing.assert_allclose(loaded[3], embeddings[3])

    def test_generic_cache_dispatch_uses_suffix(self):
        embeddings = {
            1: np.asarray([1.0], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / 'embeddings.json'
            npy_path = Path(tmpdir) / 'embeddings.npy'

            write_embedding_cache(embeddings, json_path)
            write_embedding_cache(embeddings, npy_path)
            json_loaded = read_embedding_cache(json_path)
            npy_loaded = read_embedding_cache(npy_path)

        np.testing.assert_allclose(json_loaded[1], embeddings[1])
        np.testing.assert_allclose(npy_loaded[1], embeddings[1])


class PalVitEmbeddingPathTest(unittest.TestCase):
    def test_collects_unique_relative_image_paths_from_coco_pools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_json = root / 'first.json'
            second_json = root / 'second.json'
            with first_json.open('w', encoding='utf-8') as handle:
                json.dump({
                    'images': [
                        {'id': 2, 'file_name': 'JPEGImages/000002.jpg'},
                        {'id': 1, 'file_name': 'JPEGImages/000001.jpg'},
                    ],
                }, handle)
            with second_json.open('w', encoding='utf-8') as handle:
                json.dump({
                    'images': [
                        {'id': 1, 'file_name': 'JPEGImages/000001.jpg'},
                        {'id': 'extra', 'file_name': 'JPEGImages/extra.jpg'},
                    ],
                }, handle)

            records = read_coco_image_paths([first_json, second_json], root / 'VOCdevkit')

        self.assertEqual([record.image_id for record in records], [2, 1, 'extra'])
        self.assertEqual(
            [record.file_name for record in records],
            ['JPEGImages/000002.jpg', 'JPEGImages/000001.jpg', 'JPEGImages/extra.jpg'])
        self.assertEqual(records[0].path, root / 'VOCdevkit' / 'JPEGImages/000002.jpg')

    def test_rejects_absolute_coco_file_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            annotation_json = root / 'pool.json'
            with annotation_json.open('w', encoding='utf-8') as handle:
                json.dump({
                    'images': [
                        {'id': 1, 'file_name': str(root / 'absolute.jpg')},
                    ],
                }, handle)

            with self.assertRaisesRegex(ValueError, 'file_name must be relative'):
                read_coco_image_paths([annotation_json], root)


if __name__ == '__main__':
    unittest.main()
