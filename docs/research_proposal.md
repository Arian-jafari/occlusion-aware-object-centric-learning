# Research proposal

## Motivation

Object identity association is uncertain when an object is visually similar to another object, moves substantially, or reappears after a long absence. A matching algorithm that must always choose an existing identity can turn weak evidence into an identity error. Selective classification offers a simple alternative: estimate the probability that a candidate pair is correct and reject assignments below a defined threshold.

This project studies that decision in an object-centric video setting with oracle object masks. Fixing the detections makes it possible to examine association confidence without mixing segmentation mistakes into the result.

## Research gap being investigated

This is a focused empirical question about the relationship between pairwise confidence, rejection, and identity continuity after absence. The project does not claim that memory-based association, uncertainty, or selective prediction are new concepts. It asks whether a simple probability model and explicit rejection policy produce a useful precision/coverage trade-off on DAVIS 2017.

## Research question and hypothesis

**Research question:** Can selective identity assignment using logistic-regression probability estimates, evaluated for calibration on held-out data, reduce identity switches when objects reappear after occlusion, compared with forced matching and fixed similarity-threshold matching?

**Hypothesis:** Rejecting weak assignments will lower incorrect associations and identity switches for some operating thresholds. Rejection can create new predicted identities, so fragmentation may rise. The results must therefore report both errors and coverage rather than treating higher precision as a complete tracking improvement.

## Proposed method

1. Extract frozen DINOv2 ViT-S/14 spatial patch tokens from each frame.
2. Pool the features of each ground-truth mask into an L2-normalized object embedding.
3. Keep predicted-identity memory with a normalized prototype from the bounded recent appearance history, center, bounding box, observation time, and area.
4. Compute appearance, center displacement, bounding-box IoU, area-ratio, memory-age, and best-versus-competing similarity features for every retained identity/current detection pair.
5. Generate candidate pairs by replaying training sequences with thresholded cosine association, allowing weak detections to start new identities as in selective inference. Label pair correctness from ground-truth identities after feature generation; do not put those labels in the feature vector.
6. Fit a standardized logistic regression with training-sequence candidates only.
7. At inference, exclude below-threshold edges before one-to-one assignment, include unmatched identity/detection choices with dummy nodes, and create new predicted identities for unmatched detections.

## Baselines

- Cosine similarity with forced Hungarian matching.
- Cosine Hungarian matching with a configurable fixed acceptance threshold.
- A weighted appearance, center-displacement, and box-overlap score with a fixed threshold.

All methods use the same frame masks, cached frozen features, deterministic object ordering, and memory expiration rule in the main comparison.

## Dataset and split

Use DAVIS 2017 Semi-supervised TrainVal 480p from the [official DAVIS page](https://davischallenge.org/davis2017/code.html). The loader reads the official `ImageSets/2017/train.txt` and `ImageSets/2017/val.txt` files (or equivalent split files under the supported `ImageSets/480p` or `ImageSets` layouts). It requires both files and rejects overlap. The run manifest records the exact sequence names read from these files. Training sequences fit the calibrator; validation sequences are held out from fitting and are used for evaluation and calibration analysis. The current TrainVal configuration has no independent labeled test split, so these validation results are not an untouched test estimate.

Ground-truth instance masks serve as detections. Label 0 is background. The dataset is not downloaded automatically, and no test-dev or test-challenge split is used.

## Metrics

Report per-sequence and aggregate identity switches, track fragmentations, correct/incorrect accepted associations, confidence rejections, assignment conflicts, no-candidate detections, new predicted identities, association precision/recall/F1, association coverage, and false association rate. For logistic probability estimates on validation candidate pairs, report per-sequence and pooled Brier score, configurable-bin ECE, reliability diagrams, and the number of candidate pairs. The candidate-pair precision/coverage curve is distinct from the additional assignment-level curve, which re-runs Hungarian assignment at each cutoff.

An identity switch is a predicted-ID change between consecutive observed frames for one ground-truth object; its first observation is not a switch. Fragmentation counts a changed identity after an intervening valid annotated frame in which the object is absent. Candidate gaps crossing missing or invalid dataset frames are not described as natural occlusion. No artificial observation removal is implemented in this version.

## Ablations

- Appearance only; appearance plus center displacement; appearance plus center displacement and box IoU.
- Raw cosine threshold versus logistic-probability selective decisions. Calibration quality is measured separately with Brier/ECE because cosine is not a probability and logistic regression probabilities are not assumed to be calibrated.
- Memory gaps of 5, 15, 30, and 60 missing frames, with a training-only calibrator fit for each duration.
- Validation confidence-threshold sweeps as candidate-pair and assignment-level diagnostic precision/coverage curves. The configured operating threshold is fixed independently of validation results.
- Natural reappearance outcomes grouped by missing-frame ranges. Artificially removed observations are excluded.

## Expected contributions

- A modular, reproducible oracle-mask pipeline for studying object association.
- An auditable implementation of pair-feature calibration, Hungarian assignment, and explicit selective rejection.
- An empirical analysis of association precision, confidence calibration, identity continuity, fragmentation, memory duration, and natural reappearance gaps on the official DAVIS split.

These are expected study artifacts, not established findings. Whether the observed results justify publication is an empirical and novelty assessment outside this proposal.

## Risks and limitations

Ground-truth masks isolate association but make the protocol easier and narrower than a deployed tracking system. DINOv2 remains frozen. Logistic regression is intentionally simple; class weighting can affect probability estimates and is evaluated on held-out pairs. Thresholded-cosine training replay reduces but does not eliminate memory-state shift at calibrated inference: changed association decisions alter future prototypes and candidate features. A single dataset does not establish domain generalization. Rejection may trade incorrect associations for fragmentation. Missing/invalid dataset frames reduce the amount of evaluable natural-occlusion evidence. The project does not implement IDF1/HOTA or artificial occlusion removal.
