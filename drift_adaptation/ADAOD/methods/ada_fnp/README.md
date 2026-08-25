# ADA-FNP

This package implements Active Domain Adaptation with False Negative
Prediction for the Cityscapes to Foggy Cityscapes `beta=0.02` scenario. The
only supported detector is Faster R-CNN with a BN-free VGG16 backbone.

## Reproduction contract

The serial plan contains 29 stages. It first prepares the pretrained asset and
dataset cache, then trains source-supervised plus source/target adversarial
adaptation through detector iteration 5,000. The student detector and domain
discriminator are copied exactly to the teacher at that boundary. At detector
iterations 5k, 10k, 15k, 20k, and 25k, the workflow:

1. trains the `FalseNegativePredictor` for 2,000 iterations;
2. scores every sample in the currently committed unlabeled pool;
3. selects the round budget and commits the pool transition;
4. reveals annotations only for selected samples; and
5. continues detector training from the preceding global iteration.

The final teacher is evaluated after iteration 40k. Detector checkpoints
preserve the optimizer and parameter scheduler. False-negative predictor
checkpoints retain only the completed round's model weights; every round uses
a fresh optimizer and cosine scheduler, and an interrupted predictor round is
rerun from its beginning. Before MMEngine continues a detector segment, ADAOD
also
requires an exact model-key and tensor-shape match, nonempty optimizer and
parameter-scheduler sections, and the expected saved global iteration.
Inference checkpoint loading is strict.

In the paper, this predictor is called the False Negative Prediction Module
(FNPM). The implementation uses the more direct class name
`FalseNegativePredictor`; subsequent documentation uses “false-negative
predictor” for the component.

## Package map

- `schedule.py` is the single source for detector segments, acquisition
  milestones, round length, and budget resolution.
- `models/detector.py` contains `AdaFnpDetector` and
  `AdaFnpDetectorBranch`; `models/domain_adaptation.py` contains
  `AdaFnpDomainDiscriminator`; `models/mc_dropout_roi_head.py` contains
  `AdaFnpMonteCarloDropoutRoIHead`; and
  `models/false_negative_predictor.py` contains `FalseNegativePredictor`.
- `probabilistic_teacher_augmentation.py` contains
  `ProbabilisticTeacherStrongAugmentation`.
- `training/false_negative_matching.py`,
  `training/false_negative_training.py`, and
  `training/pseudo_labeling.py` own their corresponding training operations.
- `acquisition/mc_dropout.py` and `acquisition/scoring.py` own acquisition
  inference and score computation.
- `execution/stages.py` registers the serial method stages;
  `mmdet_backend.py` implements detector, predictor, scoring, and evaluation
  runtime work; `mmdet_config.py` materializes MMDetection configs;
  `mmdet_checkpoints.py` handles strict detector checkpoints; and
  `run_files.py` owns run-local paths, pool manifests, and checkpoint lookup.

## PT-compatible detector and input path

- Caffe BGR preprocessing: mean `[103.53, 116.28, 123.675]`, unit standard
  deviation, no RGB conversion, and padding divisor 1.
- Resize to a 600-pixel short edge with a 1,333-pixel long-edge cap.
- One shared random flip is applied before branching. The strong target branch
  adds only PT photometric augmentation, so weak and strong views have the
  same geometry.
- VGG16 freezes the first two stages and loads only convolution weights from
  the checksum-pinned `vgg16_caffe.pth` asset. This initialization runs only
  for the initial 0-to-5k build; later stages load the full detector
  checkpoint directly.
- RPN anchor sizes are 128, 256, and 512 at stride 16 with integer-pixel
  centers; the positive fraction is 0.25. Train proposals use 12,000 pre-NMS
  and 2,000 post-NMS entries; test uses 6,000 and 1,000.
- RoIAlign is aligned, uses a 7-by-7 output and adaptive sampling. The two RoI
  fully connected layers have width 1,024, use Detectron2 C2 Xavier
  initialization, and bbox regression is class-specific.
- SGD uses learning rate 0.02, momentum 0.9, and weight decay 0.0001. Linear
  warm-up runs for 400 iterations from factor 0.001; learning rate drops by
  0.1 at iterations 30k and 35k.

The dataset uses exactly eight labels in PT registry order: `truck`, `car`,
`rider`, `person`, `train`, `motorcycle`, `bicycle`, and `bus`. Labels such as
`persongroup` and `cargroup` are excluded rather than converted to crowd
instances.

Source and selected target samples receive PT strong photometric augmentation,
as specified by Supplementary Equations S2 and S9. Unlabeled target samples
retain the weak-teacher and strong-student views used for pseudo-labeling.

## ADA-FNP scoring decisions

False-negative targets use score-ordered, class-aware, one-to-one matching at
IoU 0.5 or greater over at most 100 post-NMS detections. MC Dropout uses ten
fixed-proposal RoI passes with dropout 0.1 after both fully connected layers.
Mean foreground probability selects the class for each proposal. The matching
class-specific bbox-head delta is averaged and decoded once, while its
unbiased delta variance is retained through one class-aware NMS call.
Pseudo-labels require both mean delta variance at most 0.1 and foreground
confidence at least 0.5.

The acquisition artifact stores the four raw components, their normalized
values, source-domain probability, detection count, final product, and the MC
delta-variance and pseudo-label threshold metadata. Images
with no detections receive final score zero. Each scoring and training stage
materializes `target_train_unlabeled_pool_NN.json` from the committed pool, so
previously acquired samples cannot re-enter the unlabeled loader.

`FalseNegativePredictor` ends in Softplus and regresses raw false-negative
counts with mean squared error. This preserves non-negative values above one,
although Figure 4 labels the final activation as Sigmoid; the figure conflicts
with an unrestricted count target. Detector optimization applies PT's global
gradient-norm-10 clipping. Non-finite gradients stop the run before the
optimizer can update the detector.

## Evaluation

`Detectron2PascalVocMetric` reproduces PT's Detectron2/Pascal-VOC path for the
generated zero-based, half-open boxes: detection coordinates are serialized to
one decimal place, scores to three decimals, IoU must be strictly greater than
0.5, legacy `+1` box arithmetic is disabled, and VOC2012 continuous-area AP50
is returned on the 0--100 percentage scale.

## Commands and current validation boundary

```powershell
python -m tools.run_adaod --method ada-fnp --budget-percent 1 --seed 0
```

Without `--run-directory`, each run is written below
`work_dirs/runs/ada-fnp/cityscapes-to-foggy/faster-rcnn-vgg16/seed-<seed>/`
in a local-time `MM-DD-YYYY_HH_mm` directory. Passing `--run-directory`
continues to select the exact repository-relative output path.

The pinned Caffe VGG16 checkpoint is SHA-256 verified when present and
downloaded automatically when missing. A mismatched cached file stops the run
instead of being overwritten or used silently.

The interactive terminal contains one `tqdm` line. During detector and
false-negative predictor optimization its only scalar is total `loss`.
MMEngine still writes the complete 50-iteration records to its timestamped
`.log` and `vis_data/scalars.json`. The initialization segment records the
five `source.*` detector values and `domain.loss_adv`; adaptation segments
also require and record the five `target_labeled.*` values and the two
`target_unlabeled_strong.*` classification losses. Missing required keys are
an execution error rather than a silently shortened log.

Execution is connected to MMEngine/MMDetection runners for detector training,
false-negative predictor training, pool scoring, selection, reveal, and final
teacher evaluation. Model stages fail explicitly when the pinned OpenMMLab
packages or CUDA are unavailable. Completion of the 40k command and
paper-level AP50 parity remain unverified.

Manifest API version 2, run-state schema version 2, and the descriptive model,
transform, metric, and stage-executor registry names are breaking changes.
An earlier or interrupted run must be started again in a fresh run directory.
The CLI has no interrupted-run resume mode and refuses to overwrite a nonempty
run directory.
