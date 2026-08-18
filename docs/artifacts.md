# Run artifacts

Generated files live at `work_dirs/<method>/<scenario>/<run>`. Each run stores
an immutable resolved configuration and plan manifest, mutable atomic state,
checkpoints, round score shards, merged scores, selections, pool transitions,
and final evaluation metrics.

JSON artifacts are key-sorted, written to a temporary sibling, flushed, and
atomically replaced. Their SHA256 is recorded in an artifact reference.
Resume verifies completed outputs before skipping them. Checkpoints preserve
model, optimizer, scheduler, scaler, sampler, global iteration, and Python,
NumPy, PyTorch, and CUDA random state.

Converted dataset annotations are generated beneath
`work_dirs/.dataset_cache`; no generated annotation is written into `data`.
