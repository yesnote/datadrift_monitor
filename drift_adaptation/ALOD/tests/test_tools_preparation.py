import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from configs.catalog import build_experiment_config, resolve_experiment
from tools.common.preparation import prepare_required_inputs


class PreparationOrchestrationTest(unittest.TestCase):
    def test_pal_full_prepares_dataset_pretrained_and_embeddings(self):
        selection = resolve_experiment(method='pal:guide', detector='retinanet', dataset='voc')
        cfg = build_experiment_config(selection)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch('tools.common.preparation.ensure_voc_active_learning') as voc, \
                    patch('tools.common.preparation.ensure_pretrained') as pretrained, \
                    patch('tools.common.preparation.ensure_pal_embeddings') as embeddings:
                voc.return_value = {'component': 'dataset'}
                pretrained.return_value = {'component': 'pretrained'}
                embeddings.return_value = {'component': 'pal_embeddings'}

                results = prepare_required_inputs(cfg, root)

        self.assertEqual([result['component'] for result in results],
                         ['dataset', 'pretrained', 'pal_embeddings'])
        voc.assert_called_once()
        pretrained.assert_called_once()
        embeddings.assert_called_once()

    def test_pal_lius_does_not_prepare_embeddings(self):
        selection = resolve_experiment(method='pal:lius', detector='retinanet', dataset='voc')
        cfg = build_experiment_config(selection)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch('tools.common.preparation.ensure_voc_active_learning') as voc, \
                    patch('tools.common.preparation.ensure_pretrained') as pretrained, \
                    patch('tools.common.preparation.ensure_pal_embeddings') as embeddings:
                voc.return_value = {'component': 'dataset'}
                pretrained.return_value = {'component': 'pretrained'}

                results = prepare_required_inputs(cfg, root)

        self.assertEqual([result['component'] for result in results],
                         ['dataset', 'pretrained'])
        embeddings.assert_not_called()


if __name__ == '__main__':
    unittest.main()
