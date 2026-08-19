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

1. trains FNPM for 2,000 iterations;
2. scores every sample in the currently committed unlabeled pool;
3. selects the round budget and commits the pool transition;
4. reveals annotations only for selected samples; and
5. resumes detector training from the preceding global iteration.

The final teacher is evaluated after iteration 40k. Detector checkpoints
preserve the optimizer and parameter scheduler. FNPM checkpoints preserve its
model, fresh per-round optimizer and cosine scheduler, local iteration, and
random state. `--resume` verifies the saved resolved-config fingerprint and
completed artifact hashes before continuing. Before MMEngine restores a
detector segment, ADAOD also requires an exact model-key and tensor-shape
match, nonempty optimizer and parameter-scheduler sections, and the expected
saved global iteration. Inference checkpoint loading is strict.

## PT-compatible detector and input path

- Caffe BGR preprocessing: mean `[103.53, 116.28, 123.675]`, unit standard
  deviation, no RGB conversion, and padding divisor 1.
- Resize to a 600-pixel short edge with a 1,333-pixel long-edge cap.
- One shared random flip is applied before branching. The strong target branch
  adds only PT photometric augmentation, so weak and strong views have the
  same geometry.
- VGG16 freezes the first two stages and loads only convolution weights from
  the checksum-pinned `vgg16_caffe.pth` asset.
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

Source and selected target samples use the weak view only; weak-teacher and
strong-student views are created only for the unlabeled target branch. PT's
trainer applies both views to labeled/source supervision, whereas ADA-FNP
Figure 2 and Equations 13--14 describe the weak/strong split for unlabeled
target data. The baseline records this as an explicit paper-first decision.

## ADA-FNP scoring decisions

False-negative targets use score-ordered, class-aware, one-to-one matching at
IoU 0.5 or greater over at most 100 post-NMS detections, with no
additional confidence threshold. MC Dropout uses ten fixed-proposal RoI passes
with dropout 0.1 after both fully connected layers. Class probabilities and
class-specific boxes are averaged before one multiclass NMS; localization
variance follows the selected proposal/class pair.

The acquisition artifact stores the four raw components, their normalized
values, source-domain probability, detection count, and final product. Images
with no detections receive final score zero. Each scoring and training stage
materializes `target_train_unlabeled_pool_NN.json` from the committed pool, so
previously acquired samples cannot re-enter the unlabeled loader.

FNPM ends in Softplus and regresses raw false-negative counts with mean squared
error. This preserves non-negative values above one, although Figure 4 labels
the final activation as Sigmoid; the figure conflicts with an unrestricted
count target. Detector optimization does not apply PT's gradient-norm-10
clipping because ADA-FNP does not specify it. Both choices are reproduction
assumptions, not confirmed paper details.

## Evaluation

`PTVOCMetric` reproduces PT's Detectron2/Pascal-VOC path for the generated
zero-based, half-open boxes: detection coordinates are serialized to one
decimal place, scores to three decimals, IoU must be strictly greater than
0.5, legacy `+1` box arithmetic is disabled, and VOC2012 continuous-area AP50
is returned on the 0--100 percentage scale.

## Commands and current validation boundary

```powershell
python -m tools.run_adaod --method ada-fnp --budget-percent 1 --seed 0
python -m tools.run_adaod --method ada-fnp --budget-percent 1 --seed 0 --resume
python -m tools.run_adaod --method ada-fnp --budget-percent 1 --seed 0 --offline
```

Non-dry execution is connected to MMEngine/MMDetection runners for detector
training, FNPM training, pool scoring, selection, reveal, and final teacher
evaluation. Model stages fail explicitly when the pinned OpenMMLab packages or
CUDA are unavailable. The pinned CUDA/MMCV gate, full model construction,
direct real-batch loss/backward, one official C-to-F Runner iteration,
checkpoint creation, and checkpoint resume through the next iteration pass.
The 40k command is technically ready to run; completion and paper-level AP50
parity remain unverified.
