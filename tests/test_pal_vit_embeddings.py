import json
import tempfile
import unittest
from pathlib import Path

from methods.pal.vit_embeddings import (
    DEFAULT_GOOGLE_VIT_MODEL,
    missing_image_paths,
    read_coco_image_paths,
    resolve_coco_image_path,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle)


class PalVitEmbeddingPathTest(unittest.TestCase):
    def test_reads_unique_coco_image_paths_in_annotation_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'VOCdevkit'
            image_path = root / 'VOC2007' / 'JPEGImages' / '000001.jpg'
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b'placeholder')

            first = Path(tmpdir) / 'first.json'
            second = Path(tmpdir) / 'second.json'
            _write_json(first, {
                'images': [
                    {'id': 1, 'file_name': 'VOC2007/JPEGImages/000001.jpg'},
                    {'id': 2, 'file_name': 'VOC2012/JPEGImages/000002.jpg'},
                ],
            })
            _write_json(second, {
                'images': [
                    {'id': 1, 'file_name': 'VOC2007/JPEGImages/000001.jpg'},
                    {'id': 3, 'file_name': 'VOC2012/JPEGImages/000003.jpg'},
                ],
            })

            records = read_coco_image_paths([first, second], root)

        self.assertEqual([record.image_id for record in records], [1, 2, 3])
        self.assertEqual(records[0].file_name, 'VOC2007/JPEGImages/000001.jpg')
        self.assertTrue(str(records[0].path).endswith('VOC2007\\JPEGImages\\000001.jpg')
                        or str(records[0].path).endswith('VOC2007/JPEGImages/000001.jpg'))

    def test_conflicting_duplicate_image_id_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'VOCdevkit'
            ann = Path(tmpdir) / 'annotations.json'
            _write_json(ann, {
                'images': [
                    {'id': 1, 'file_name': 'VOC2007/JPEGImages/000001.jpg'},
                    {'id': 1, 'file_name': 'VOC2012/JPEGImages/000001.jpg'},
                ],
            })

            with self.assertRaises(ValueError):
                read_coco_image_paths([ann], root)

    def test_rejects_absolute_or_escaping_file_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'VOCdevkit'
            with self.assertRaises(ValueError):
                resolve_coco_image_path(root, str(Path(tmpdir).resolve()))
            with self.assertRaises(ValueError):
                resolve_coco_image_path(root, '../outside.jpg')

    def test_missing_image_paths_preserves_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'VOCdevkit'
            ann = Path(tmpdir) / 'annotations.json'
            _write_json(ann, {
                'images': [
                    {'id': 'a', 'file_name': 'a.jpg'},
                    {'id': 'b', 'file_name': 'b.jpg'},
                ],
            })

            records = read_coco_image_paths([ann], root)
            missing = missing_image_paths(records)

        self.assertEqual([path.name for path in missing], ['a.jpg', 'b.jpg'])

    def test_default_google_vit_model_is_explicit(self):
        self.assertEqual(DEFAULT_GOOGLE_VIT_MODEL, 'google/vit-base-patch16-224-in21k')


if __name__ == '__main__':
    unittest.main()
