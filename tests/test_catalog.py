import unittest
from pathlib import Path

from configs.catalog import build_experiment_config, resolve_experiment, resolve_method_alias
from tools.run_active_learning import build_round_plan


class CatalogResolutionTest(unittest.TestCase):
    def test_pal_full_combo_resolves_to_full_config(self):
        selection = resolve_experiment(
            method='pal:guide',
            detector='retinanet',
            dataset='voc',
        )

        self.assertIsNotNone(selection)
        self.assertEqual(selection.method, 'pal')
        self.assertEqual(selection.method_alias, 'pal:guide')
        self.assertEqual(selection.cfg_overrides['pal_mode'], 'full')
        self.assertEqual(selection.preset.name, 'pal-retinanet-voc')

        cfg = build_experiment_config(selection)
        self.assertEqual(cfg['pal_mode'], 'full')
        self.assertEqual(cfg['pal_embedding_source'], 'external')
        self.assertEqual(cfg['pal_embedding_path'], 'work_dirs/pal_embeddings/voc_google_vit_embeddings.npy')
        self.assertEqual(cfg['output_dir'], 'work_dirs/retinanet_voc_pal_7rounds_5percent_to_20percent')
        self.assertEqual(cfg['train_config'], 'configs/alod_mmdet/retinanet_voc_train_26e.py')

    def test_pal_lius_combo_resolves_to_lius_config(self):
        selection = resolve_experiment(
            method='pal:lius',
            detector='retinanet',
            dataset='voc',
        )

        self.assertIsNotNone(selection)
        self.assertEqual(selection.method, 'pal')
        self.assertEqual(selection.method_alias, 'pal:lius')
        self.assertEqual(selection.cfg_overrides['pal_mode'], 'lius')
        self.assertEqual(selection.preset.name, 'pal-lius-retinanet-voc')

        cfg = build_experiment_config(selection)
        self.assertEqual(cfg['pal_mode'], 'lius')
        self.assertNotIn('pal_embedding_path', cfg)
        self.assertEqual(cfg['output_dir'], 'work_dirs/retinanet_voc_pal_lius_7rounds_5percent_to_20percent')

    def test_preset_uses_own_default_method_alias(self):
        full = resolve_experiment(preset='pal-retinanet-voc')
        lius = resolve_experiment(preset='pal-lius-retinanet-voc')

        self.assertIsNotNone(full)
        self.assertIsNotNone(lius)
        self.assertEqual(full.method, 'pal')
        self.assertEqual(full.method_alias, 'pal')
        self.assertEqual(lius.method, 'pal')
        self.assertEqual(lius.method_alias, 'pal:lius')

    def test_method_alias_normalization(self):
        method, overrides = resolve_method_alias('pal/full')

        self.assertEqual(method, 'pal')
        self.assertEqual(overrides['pal_mode'], 'full')

    def test_ppal_catalog_config_builds_sampler_defaults(self):
        selection = resolve_experiment(method='ppal', detector='retinanet', dataset='voc')
        cfg = build_experiment_config(selection)

        self.assertEqual(cfg['train_config'], 'configs/alod_mmdet/retinanet_voc_train_26e.py')
        self.assertEqual(cfg['uncertainty_sampler_config']['type'], 'DCUSSampler')
        self.assertEqual(cfg['diversity_sampler_config']['type'], 'DiversitySampler')
        self.assertEqual(cfg['uncertainty_sampler_config']['oracle_annotation_path'], cfg['oracle_path'])

    def test_pal_round_plan_contains_expected_steps(self):
        selection = resolve_experiment(method='pal:guide', detector='retinanet', dataset='voc')
        cfg = build_experiment_config(selection)
        plans = build_round_plan(
            cfg,
            Path(cfg['output_dir']),
            selection.method,
            round_index=1,
        )

        self.assertEqual(
            [plan.name for plan in plans],
            [
                'train_round_01',
                'eval_round_01',
                'pal_labeled_inference_round_01',
                'pal_unlabeled_inference_round_01',
                'pal_acquisition_round_01',
            ],
        )


if __name__ == '__main__':
    unittest.main()
