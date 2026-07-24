import json
import tempfile
import unittest
from pathlib import Path

from tools.common.voc import build_voc0712_oracle, ensure_voc_active_learning


def _write_voc_sample(voc_root: Path, sample_id: str, class_name: str) -> None:
    (voc_root / 'Annotations').mkdir(parents=True, exist_ok=True)
    (voc_root / 'ImageSets' / 'Main').mkdir(parents=True, exist_ok=True)
    (voc_root / 'JPEGImages').mkdir(parents=True, exist_ok=True)
    (voc_root / 'ImageSets' / 'Main' / 'trainval.txt').write_text(
        sample_id + '\n',
        encoding='utf-8',
    )
    (voc_root / 'Annotations' / ('%s.xml' % sample_id)).write_text(
        '''<annotation>
  <size><width>100</width><height>80</height></size>
  <object>
    <name>{class_name}</name>
    <difficult>0</difficult>
    <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>11</xmax><ymax>22</ymax></bndbox>
  </object>
</annotation>'''.format(class_name=class_name),
        encoding='utf-8',
    )


class VocPreparationTest(unittest.TestCase):
    def test_builds_tiny_voc0712_oracle_and_initial_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vocdevkit = root / 'data' / 'VOCdevkit'
            _write_voc_sample(vocdevkit / 'VOC2007', '000001', 'cat')
            _write_voc_sample(vocdevkit / 'VOC2012', '000002', 'dog')

            oracle = build_voc0712_oracle(vocdevkit)

            self.assertEqual(len(oracle['images']), 2)
            self.assertEqual(len(oracle['annotations']), 2)
            self.assertEqual(oracle['images'][0]['file_name'], 'VOC2007/JPEGImages/000001.jpg')

            result = ensure_voc_active_learning(
                {
                    'type': 'voc0712',
                    'vocdevkit': 'data/VOCdevkit',
                    'oracle_output': 'data/VOC0712/annotations/trainval_0712.json',
                    'split_output_dir': 'data/active_learning/voc',
                    'n_labeled': 1,
                    'n_diff': 1,
                    'seed': 0,
                },
                root,
            )

            labeled_path = Path(result['split_paths'][0]['labeled'])
            unlabeled_path = Path(result['split_paths'][0]['unlabeled'])
            labeled = json.loads(labeled_path.read_text(encoding='utf-8'))
            unlabeled = json.loads(unlabeled_path.read_text(encoding='utf-8'))

        self.assertEqual(result['status'], 'ready')
        self.assertEqual(len(labeled['images']), 1)
        self.assertEqual(len(unlabeled['images']), 1)
        self.assertEqual(labeled['annotations'][0]['bbox'], [0, 1, 10, 20])


if __name__ == '__main__':
    unittest.main()
