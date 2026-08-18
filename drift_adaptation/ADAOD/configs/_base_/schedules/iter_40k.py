'''Reusable 40k-iteration SGD schedule.'''

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.02,
        momentum=0.9,
        weight_decay=0.0001,
    ),
)

param_scheduler = [
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
