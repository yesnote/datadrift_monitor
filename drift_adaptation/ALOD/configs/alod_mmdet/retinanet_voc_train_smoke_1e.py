_base_ = './retinanet_voc_train_26e.py'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ],
)

lr_config = dict(policy='step', warmup='linear', warmup_iters=1, warmup_ratio=0.001, step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=1, by_epoch=True)
