'''Forty-thousand-iteration schedule used by ADA-FNP comparisons.'''

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(
        max_norm=10.0,
        norm_type=2.0,
        error_if_nonfinite=True,
    ),
    optimizer=dict(
        type='SGD',
        lr=0.02,
        momentum=0.9,
        weight_decay=0.0001,
    ),
)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        begin=0,
        end=400,
        by_epoch=False,
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=40000,
        by_epoch=False,
        milestones=[30000, 35000],
        gamma=0.1,
    ),
]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,
    val_interval=5000,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
