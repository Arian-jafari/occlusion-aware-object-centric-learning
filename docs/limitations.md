# Limitations

1. **Oracle masks:** Ground-truth masks are used as detections. Segmentation quality, detector recall, object discovery, and detector identity errors are outside the evaluation.
2. **Frozen representation:** DINOv2 ViT-S/14 is frozen. This project does not train or claim to improve a visual representation.
3. **Simple confidence model:** The initial confidence estimator is standardized logistic regression. It may not represent complex interactions or remain calibrated under distribution shift.
4. **Single dataset:** DAVIS 2017 alone cannot establish generalization to other video domains, camera conditions, object categories, or annotation styles.
5. **Rejection/fragmentation trade-off:** Rejection can lower incorrect accepted associations while starting new predicted identities and increasing fragmentation.
6. **Precision is incomplete:** Higher association precision alone does not imply better identity continuity or tracking. Recall, coverage, identity switches, fragmentation, and counts must be considered together.
7. **Calibration is separate:** Brier score, ECE, and reliability diagrams assess candidate probabilities. They are not substitutes for tracking metrics, and logistic regression probabilities are not assumed calibrated.
8. **No validation fitting:** The calibrator must be trained using training sequences only. Validation labels are reserved for final candidate calibration and tracking evaluation; threshold selection on validation is diagnostic only.
9. **Natural and artificial absence:** This version analyzes natural object absence only. Artificial observation removal is not implemented and must not be mixed with natural-occlusion measurements if added.
10. **Novelty:** A successful experiment does not establish publication-level novelty. The contribution is an empirical investigation, not a claim to be the first method using memory, uncertainty, or temporal association.

Additional practical limitations include a possible train/inference candidate-distribution shift, sensitivity to mask quality despite oracle annotations, finite DAVIS sequence counts, and the fact that the simplified last-box motion feature does not extrapolate velocity. IDF1 and HOTA are omitted until a validated implementation and protocol are added.
