# Methodology

## Scope

The target is object identity association given ground-truth instance masks. Each nonzero instance label in a valid annotation supplies one object mask and therefore one detection. Background is label 0. The mask label is used to pool pixels, create training targets, and evaluate associations; inference receives frame-local detection IDs, embeddings, and geometry without ground-truth IDs.

This protocol does not evaluate segmentation, object discovery, detector recall, or a complete multi-object tracker.

## DAVIS loading and frame validity

The loader resolves the expected `JPEGImages/480p`, `Annotations/480p`, and official train/validation split files. Both split files are mandatory and must be nonempty, contain unique safe sequence names, and be disjoint. A sequence/frame pair is matched by filename stem. An image without an annotation, an annotation without an image, a decode failure, a non-single-channel/noninteger mask, a negative label, or a dimension mismatch is reported. Invalid matched frames are skipped with an error record; a sequence with no valid matched frames is marked skipped. Inspection streams frames so it does not retain a full sequence in RAM.

## Frozen spatial features

The default backbone is the official pretrained `facebookresearch/dinov2` `dinov2_vits14` model. It is loaded in evaluation mode, frozen, and called under inference mode. Image dimensions are resized while preserving aspect ratio and rounded to multiples of the 14-pixel patch size. RGB values are normalized with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. CUDA uses FP16 autocast when enabled; CPU and non-autocast paths use FP32.

Only `x_norm_patchtokens` are used. The CLS/global token is rejected as an object representation. For each mask, a binary coverage image is resized to the feature image dimensions using area resampling. Its fractional occupancy is averaged inside each 14×14 patch. Patch tokens are averaged with those occupancy weights and L2-normalized. Empty, non-finite, or zero-norm results return `None` and are reported by extraction. Partial coverage preserves small objects that touch very few patches.

Each cache key includes schema version, sequence, frame, mask object label, backbone repository/name, resize and normalization settings, input size, device type, and precision mode. Cache files are atomic NumPy arrays outside the repository. No checkpoint is copied into source control.

## Detections and deterministic ordering

For every mask object, geometry consists of pixel-center centroid, half-open `(x1,y1,x2,y2)` box, and pixel area. Detections are ordered by centroid, box, area, and a mask-bitmap tie-breaker. The resulting local detection ID does not depend on the ground-truth instance label. A separate mapping from detection IDs to ground-truth labels is held by calibration/evaluation code and is never passed to the association decision.

## Memory

Each predicted identity stores its own generated predicted ID, latest normalized embedding, previous center/box/area, most recent observation frame, age, missing-frame count, and bounded optional appearance history. With maximum gap `N`, an identity can match a reappearance after `N` complete absent frames. It expires before an observation after more than `N` missing frames. Rejection does not delete the old identity; the detection receives a new predicted ID and the old identity stays until expiry.

## Association features

| Feature | Definition | Range |
|---|---|---:|
| Cosine similarity | Cosine between latest memory embedding and current embedding | `[-1, 1]` |
| Center displacement | Euclidean centroid distance divided by image diagonal, clipped | `[0, 1]` |
| Bounding-box IoU | IoU between previous and current half-open boxes | `[0, 1]` |
| Area ratio | `min(previous area,current area)/max(...)` | `[0, 1]` |
| Observation age | Frames since prior observation divided by `max_gap+1`, clipped | `[0, 1]` |
| Similarity margin | Candidate cosine minus best competing identity cosine, divided by two; zero if no competitor | `[-1, 1]` |

The geometry uses the last observed center and box without learned motion extrapolation. Feature generation returns one vector per active-memory/current-detection pair and does not make an assignment decision.

## Decision rules

All methods use one-to-one Hungarian assignment over the current active identities and detections.

- **Cosine forced:** maximize cosine and accept every selected pair.
- **Cosine threshold:** maximize cosine and reject selected pairs below the configured cosine threshold.
- **Motion + appearance:** combine normalized cosine, center proximity, box IoU, and optionally area ratio by nonnegative configurable weights; reject selected scores below a fixed threshold.
- **Calibrated selective:** score all candidate feature rows with the fitted logistic regression, maximize those probabilities, accept above the configured confidence threshold, and explicitly reject selected assignments below it.

Each rejected detection starts a new predicted identity. Unassigned detections also start new identities. Existing unassigned identities remain in memory until expiry.

## Training examples and probability model

Candidate data are generated by replaying each training sequence with forced-cosine matching. For a pair, the target is positive if the predicted memory identity's previous ground-truth owner matches the current detection's ground-truth object. Ground-truth owner bookkeeping is external to the feature vector and is consulted only for target creation. This trains on the same forced-matching memory distribution for every run. StandardScaler plus logistic regression is fit with configurable class weighting (default `balanced`), iteration limit, and random state. Both classes must be present. Validation sequence names are explicitly checked against the training set, and validation labels never enter fitting.

Logistic regression provides a probability estimate; it does not guarantee that those probabilities are calibrated. Brier score, ECE, and reliability diagrams are computed only on held-out validation candidate pairs.

## Metrics and reporting

Identity switches count predicted-ID changes between consecutive observed frames of one ground-truth object; the first observation is excluded. A fragmentation counts an identity change after at least one intervening valid annotated frame in which the object is absent. An interval with missing/invalid source frames is not counted as natural occlusion fragmentation. Candidate coverage in the precision/coverage curve is accepted candidate pairs divided by all validation candidate pairs; reported association coverage is accepted Hungarian assignments divided by all selected Hungarian decisions before rejection.

Association precision is correct accepted pairs divided by accepted pairs. Recall is correct accepted pairs divided by current object observations that had an active memory identity of that same object. False association rate is incorrect accepted pairs divided by accepted pairs. Counts and denominators are written so these rates can be audited.

IDF1 and HOTA are not reported: this project has no validated implementation for this oracle-mask protocol. No approximate substitutes are called official metrics.
