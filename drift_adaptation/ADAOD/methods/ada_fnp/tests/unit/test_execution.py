import json
import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from methods.ada_fnp.acquisition.records import RawAdaFnpScore
from methods.ada_fnp.execution import (
    ExecutionDependencyError,
    ExecutionServices,
    FnpmSession,
)
from methods.ada_fnp.execution.backend import (
    MmdetExecutionBackend,
    MmdetRuntime,
    _pool_samples_by_image_id,
    _single_dataset_loader,
    validate_detector_resume_checkpoint,
)
from methods.ada_fnp.execution.executors import create_executor_registry
from methods.ada_fnp.manifest import MANIFEST
from methods.ada_fnp.models.fnpm import FalseNegativePredictionModule
from methods.ada_fnp.phases import resolve_detector_phase
from methods.common.data.cityscapes import CATEGORY_IDS, CITYSCAPES_CLASSES
from methods.common.data.image_identity import SampleIdentity
from methods.common.data.pool import PoolState
from methods.common.engine import (
    ArtifactStore,
    ExecutionContext,
    RunStateStore,
    StageRunner,
    load_executor_factory,
)
from methods.common.contracts import StageSpec
from tools.common.config import compose_config


def test_fnpm_loader_keeps_the_configured_domain_batch_size():
    loader = _single_dataset_loader({'type': 'CocoDataset'}, batch_size=4)
    assert loader['batch_size'] == 4
    assert loader['drop_last'] is True


class _FakeBackend:
    def train_detector(
        self, stage, context, phase, checkpoint_path, resume_from
    ):
        del stage, context, phase, resume_from
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b'detector checkpoint')
        return checkpoint_path

    def create_fnpm_session(self, stage, context, checkpoint_path):
        del stage, context, checkpoint_path
        fnpm = FalseNegativePredictionModule(in_channels=1)
        teacher = nn.Identity()

        def extractor(model, batch):
            del model
            features = torch.full((1, 1, 1, 1), float(batch + 1))
            return features, torch.zeros(1)

        return FnpmSession(
            fnpm=fnpm,
            teacher=teacher,
            source_batch_provider=lambda iteration: iteration,
            teacher_batch_extractor=extractor,
        )

    def score_pool(self, stage, context, samples):
        del stage, context
        return tuple(
            RawAdaFnpScore(
                sample=sample,
                false_negative=float(index + 1),
                localization=float(index + 1),
                entropy=float(index + 1),
                diversity=float(index + 1),
                source_domain_probability=0.5,
                detection_count=1,
            )
            for index, sample in enumerate(samples)
        )

    def evaluate(self, stage, context, checkpoint_path):
        del stage, context
        assert checkpoint_path.is_file()
        return {'AP50': 52.0}


def _write_fake_dataset(context):
    cache = (
        context.repository_root / context.config['runtime']['dataset_cache_root'] /
        context.config['scenario']
    )
    cache.mkdir(parents=True, exist_ok=True)
    images = [
        {
            'id': index + 1,
            'sample_id': '{}:frame-{:03d}'.format(
                context.config['dataset']['target']['train_sample_id_namespace'], index
            ),
            'file_name': 'frame-{:03d}.png'.format(index),
            'width': 1,
            'height': 1,
        }
        for index in range(5)
    ]
    categories = [
        {'id': CATEGORY_IDS[name], 'name': name, 'supercategory': 'object'}
        for name in CITYSCAPES_CLASSES
    ]
    (cache / 'target_train_unlabeled.json').write_text(
        json.dumps({'images': images, 'annotations': [], 'categories': categories}),
        encoding='utf-8',
    )
    (cache / 'target_train_oracle.json').write_text(
        json.dumps({'images': images, 'annotations': [], 'categories': categories}),
        encoding='utf-8',
    )
    return {'fingerprint': 'fixture'}


def _context(tmp_path):
    config = compose_config(
        MANIFEST,
        overrides={
            'acquisition': {'total_budget': 5},
            'dataset': {'target': {'expected_train_images': 5}},
            'fnpm': {'iterations_per_round': 2},
        },
    )
    run_directory = tmp_path / 'work_dirs' / 'run'
    run_directory.mkdir(parents=True)
    state_store = RunStateStore(run_directory / 'state.json')
    return config, ExecutionContext(
        config=config,
        repository_root=tmp_path,
        run_directory=run_directory,
        state_store=state_store,
        artifact_store=ArtifactStore(run_directory),
    )


def test_manifest_factory_runs_complete_five_round_fixture(tmp_path):
    config, context = _context(tmp_path)

    def prepare_asset(current_context):
        path = current_context.repository_root / 'work_dirs/pretrained/vgg16_caffe.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture pretrained')
        return path

    services = ExecutionServices(
        backend=_FakeBackend(),
        asset_preparer=prepare_asset,
        dataset_preparer=_write_fake_dataset,
    )
    registry = create_executor_registry(context, services=services)
    plan = MANIFEST.plan_factory(config)

    StageRunner(registry, context.state_store, context).run(plan)

    state = context.state_store.load()
    assert state.status == 'complete'
    assert len(state.completed_stages) == 29
    assert state.active_round == 5
    assert state.global_detector_iteration == 40000
    assert state.pool_artifact_id is not None
    assert state.detector_checkpoint_artifact_id is not None
    assert state.fnpm_checkpoint_artifact_id is not None
    assert (context.run_directory / 'artifacts/evaluation.json').is_file()
    score_artifact = json.loads(
        (context.run_directory / 'artifacts/rounds/01/scores.json').read_text(
            encoding='utf-8'
        )
    )
    assert set(score_artifact['records'][0]['fields']) == {
        'raw', 'normalized', 'source_domain_probability',
        'detection_count', 'final_score',
    }
    labeled = json.loads(
        (context.run_directory / 'datasets/target_train_labeled_round_05.json')
        .read_text(encoding='utf-8')
    )
    assert len(labeled['images']) == 5

    (context.run_directory / 'artifacts/evaluation.json').write_text(
        '{}', encoding='utf-8'
    )
    with pytest.raises(RuntimeError, match='verification'):
        StageRunner(registry, context.state_store, context).run(plan)


def test_manifest_loads_execution_factory_before_mmdet_registrations(tmp_path):
    _, context = _context(tmp_path)
    factory = load_executor_factory(MANIFEST)
    registry = factory(context)
    assert registry.resolve('ada_fnp.train_detector')
    assert registry.resolve('common.select')


def test_unavailable_mmdet_backend_fails_without_writing_placeholder(tmp_path):
    _, context = _context(tmp_path)
    checkpoint = context.run_directory / 'checkpoints/detector.pth'
    stage = StageSpec('train', 'ada_fnp.train_detector')

    def unavailable_runtime():
        raise ExecutionDependencyError('missing test runtime')

    with pytest.raises(ExecutionDependencyError):
        MmdetExecutionBackend(runtime_loader=unavailable_runtime).train_detector(
            stage, context, object(), checkpoint, None
        )

    assert not checkpoint.exists()


class _BranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.Linear(1, 1, bias=False)
        self.teacher = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.student.weight, 2.0)
        nn.init.constant_(self.teacher.weight, 0.0)


class _FakeRunner:
    def __init__(self, config):
        self.config = config
        self.model = _BranchModel()
        self.trained = False
        self.iter = 0

    def train(self):
        self.trained = True
        self.iter = self.config['train_cfg']['max_iters']
        with torch.no_grad():
            self.model.student.weight.fill_(3.0)

    def save_checkpoint(self, out_dir, filename, **kwargs):
        self.saved_kwargs = kwargs
        path = Path(out_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'optimizer': {'state': {}, 'param_groups': []},
                'param_schedulers': [{}],
                'meta': kwargs['meta'],
            },
            path,
        )


def _runtime_config():
    source = dict(type='CocoDataset', ann_file='source_train.json')
    unlabeled = dict(type='CocoDataset', ann_file='target_train_unlabeled.json')
    labeled = dict(type='CocoDataset', ann_file='target_train_labeled.json')
    return {
        'custom_imports': {},
        'env_cfg': {'cudnn_benchmark': False},
        'model': {
            'detector': {'backbone': {'pretrained_checkpoint': 'old.pth'}},
            'enable_unsupervised_loss': False,
        },
        'stage_overrides': {
            'initial': {
                'train_dataloader': {
                    'sampler': {'seed': 0},
                    'dataset': {'datasets': [source, unlabeled]},
                },
                'model': {'enable_unsupervised_loss': False},
                'custom_hooks': [],
            },
            'adaptation': {
                'train_dataloader': {
                    'sampler': {'seed': 0},
                    'dataset': {'datasets': [source, labeled, unlabeled]},
                },
                'model': {'enable_unsupervised_loss': True},
                'custom_hooks': [{'type': 'MeanTeacherHook'}],
            },
        },
        'train_cfg': {'max_iters': 40000, 'val_interval': 5000},
        'default_hooks': {
            'checkpoint': {'interval': 5000, 'by_epoch': False}
        },
        'param_scheduler': [
            {'type': 'LinearLR', 'begin': 0, 'end': 400},
            {'type': 'MultiStepLR', 'milestones': [30000, 35000]},
        ],
        'val_cfg': {},
        'val_dataloader': {},
        'val_evaluator': {},
        'test_cfg': {},
        'test_dataloader': {
            'dataset': {'type': 'CocoDataset', 'ann_file': 'target_val.json'}
        },
        'test_evaluator': {'type': 'PTVOCMetric'},
    }


def test_mmdet_train_adapter_preserves_global_schedule_and_atomic_checkpoint(
    tmp_path,
):
    _, context = _context(tmp_path)
    cache = (
        context.repository_root / context.config['runtime']['dataset_cache_root'] /
        context.config['scenario']
    )
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'source_train.json').write_text('{}', encoding='utf-8')
    samples = tuple(
        SampleIdentity('foggy-cityscapes.beta-0.02.train', 'frame-{:03d}'.format(index))
        for index in range(5)
    )
    target_pool_cache = {
        'info': {},
        'images': [
            {'id': index + 1, 'sample_id': sample.qualified_id}
            for index, sample in enumerate(samples)
        ],
        'annotations': [],
        'categories': [],
    }
    (cache / 'target_train_unlabeled.json').write_text(
        json.dumps(target_pool_cache), encoding='utf-8'
    )
    pool_round_zero = PoolState.initialize(samples, total_budget=5)
    context.artifact_store.write_json(
        'artifacts/pool/round_00.json',
        pool_round_zero.to_dict(),
        'target_pool_state',
        'prepare_datasets',
    )
    built = []

    def build_runner(config):
        runner = _FakeRunner(config)
        built.append(runner)
        return runner

    runtime = MmdetRuntime(
        load_config=lambda path: copy.deepcopy(_runtime_config()),
        import_custom_modules=lambda config: None,
        build_runner=build_runner,
        build_model=lambda config: None,
        build_dataloader=lambda config, seed: (),
        load_model_checkpoint=lambda model, path: None,
    )
    backend = MmdetExecutionBackend(
        runtime_loader=lambda: runtime, require_cuda=False
    )
    stage = StageSpec('train_0_5000', 'ada_fnp.train_detector')
    checkpoint = context.run_directory / 'checkpoints/detector_05000.pth'

    written = backend.train_detector(
        stage,
        context,
        resolve_detector_phase(0, 5000, 0),
        checkpoint,
        None,
    )

    assert written == checkpoint
    assert checkpoint.is_file()
    assert not checkpoint.with_name('.detector_05000.tmp.pth').exists()
    runner = built[0]
    assert runner.trained
    assert runner.config['resume'] is False
    assert runner.config['load_from'] is None
    assert runner.config['train_cfg']['max_iters'] == 5000
    assert runner.config['param_scheduler'][1]['milestones'] == [30000, 35000]
    assert runner.config['val_cfg'] is None
    assert torch.equal(runner.model.teacher.weight, runner.model.student.weight)
    assert runner.saved_kwargs['save_optimizer'] is True
    assert runner.saved_kwargs['save_param_scheduler'] is True

    state = context.state_store.load()
    state.active_round = 1
    context.state_store.save(state)
    context.artifact_store.write_json(
        'artifacts/pool/round_01.json',
        pool_round_zero.acquire((samples[0],)).to_dict(),
        'target_pool_state',
        'select_round_01',
    )
    labeled_manifest = (
        context.run_directory / 'datasets/target_train_labeled_round_01.json'
    )
    labeled_manifest.parent.mkdir(parents=True, exist_ok=True)
    labeled_manifest.write_text('{}', encoding='utf-8')
    adaptation_checkpoint = (
        context.run_directory / 'checkpoints/detector_10000.pth'
    )
    backend.train_detector(
        StageSpec('train_5000_10000', 'ada_fnp.train_detector'),
        context,
        resolve_detector_phase(5000, 10000, 1),
        adaptation_checkpoint,
        checkpoint,
    )

    adaptation = built[1]
    assert adaptation.config['resume'] is True
    assert adaptation.config['load_from'] == str(checkpoint)
    assert adaptation.config['train_cfg']['max_iters'] == 10000
    assert adaptation.config['model']['enable_unsupervised_loss'] is True
    assert adaptation.config['custom_hooks'] == [{'type': 'MeanTeacherHook'}]
    datasets = adaptation.config['train_dataloader']['dataset']['datasets']
    assert datasets[1]['ann_file'] == str(labeled_manifest)
    dynamic_unlabeled = Path(datasets[2]['ann_file'])
    assert dynamic_unlabeled.name == 'target_train_unlabeled_pool_01.json'
    dynamic_payload = json.loads(dynamic_unlabeled.read_text(encoding='utf-8'))
    assert [image['sample_id'] for image in dynamic_payload['images']] == [
        sample.qualified_id for sample in samples[1:]
    ]
    assert dynamic_payload['annotations'] == []
    assert set(_pool_samples_by_image_id(
        dynamic_unlabeled, samples[1:]
    ).values()) == set(samples[1:])
    assert not torch.equal(
        adaptation.model.teacher.weight, adaptation.model.student.weight
    )


def test_detector_resume_checkpoint_requires_exact_reproducibility_state(
    tmp_path,
):
    model = _BranchModel()
    checkpoint = tmp_path / 'resume.pth'
    valid = {
        'state_dict': model.state_dict(),
        'optimizer': {'state': {}, 'param_groups': []},
        'param_schedulers': [{}],
        'meta': {'global_iteration': 5000},
    }
    torch.save(valid, checkpoint)
    validate_detector_resume_checkpoint(checkpoint, model, (5000,))

    invalid = copy.deepcopy(valid)
    invalid['state_dict']['student.weight'] = torch.zeros(2, 1)
    torch.save(invalid, checkpoint)
    with pytest.raises(ValueError, match='tensor shapes'):
        validate_detector_resume_checkpoint(checkpoint, model, (5000,))

    invalid = copy.deepcopy(valid)
    del invalid['optimizer']
    torch.save(invalid, checkpoint)
    with pytest.raises(ValueError, match='missing: optimizer'):
        validate_detector_resume_checkpoint(checkpoint, model, (5000,))

    invalid = copy.deepcopy(valid)
    invalid['param_schedulers'] = []
    torch.save(invalid, checkpoint)
    with pytest.raises(ValueError, match='nonempty param-scheduler'):
        validate_detector_resume_checkpoint(checkpoint, model, (5000,))

    invalid = copy.deepcopy(valid)
    invalid['meta']['global_iteration'] = 4999
    torch.save(invalid, checkpoint)
    with pytest.raises(ValueError, match='global iteration'):
        validate_detector_resume_checkpoint(checkpoint, model, (5000,))


def test_mmdet_evaluator_resolves_prefixed_pt_voc_ap50_as_percent(tmp_path):
    _, context = _context(tmp_path)
    cache = (
        context.repository_root / context.config['runtime']['dataset_cache_root'] /
        context.config['scenario']
    )
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'target_val.json').write_text('{}', encoding='utf-8')

    class EvalRunner:
        def test(self):
            return {'pt_voc/mAP': 52.0, 'pt_voc/AP50': 52.0}

    runtime = MmdetRuntime(
        load_config=lambda path: copy.deepcopy(_runtime_config()),
        import_custom_modules=lambda config: None,
        build_runner=lambda config: EvalRunner(),
        build_model=lambda config: None,
        build_dataloader=lambda config, seed: (),
        load_model_checkpoint=lambda model, path: None,
    )
    backend = MmdetExecutionBackend(
        runtime_loader=lambda: runtime, require_cuda=False
    )

    assert backend.evaluate(
        StageSpec('evaluate', 'common.evaluate'),
        context,
        context.run_directory / 'detector.pth',
    ) == {'AP50': 52.0}
