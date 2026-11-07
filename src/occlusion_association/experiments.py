"""Leakage-aware experiment orchestration over cached DAVIS features."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .association import AssociationMethod, OnlineAssociator
from .association_features import FEATURE_NAMES
from .calibration import (
    CalibrationCandidates,
    LogisticAssociationCalibrator,
    fit_logistic_calibrator,
    generate_calibration_candidates,
)
from .dataset import DAVISDataset, DatasetIssue, FrameData
from .evaluation import (
    FrameEvaluationLog,
    assignment_precision_coverage_curve,
    calibration_metrics,
    candidate_targets_and_probabilities,
    evaluate_sequences,
    occlusion_gap_metrics,
    precision_coverage_curve,
)
from .feature_cache import FeatureCache
from .features import build_preprocessing_descriptor, load_cached_frame_embeddings
from .memory import ObjectMemory
from .object_representation import build_object_detections


@dataclass
class MethodRun:
    """Frame logs, metrics, and pair-/detection-level outputs for one method."""

    method_name: str
    per_sequence_metrics: list[dict[str, Any]]
    aggregate_metrics: dict[str, Any]
    frame_logs: list[FrameEvaluationLog]
    association_rows: list[dict[str, Any]]
    detection_rows: list[dict[str, Any]]
    issues: list[DatasetIssue]


def _frame_index(frame_name: str, ordinal: int) -> int:
    return int(frame_name) if frame_name.isdigit() else ordinal


def _stream_frames(
    dataset: DAVISDataset,
    sequence_name: str,
    frame_limit: int | None,
    issues: list[DatasetIssue],
) -> Iterable[FrameData]:
    iterator = dataset.iter_frames(sequence_name, issue_sink=issues)
    if frame_limit is None:
        yield from iterator
    else:
        yield from islice(iterator, frame_limit)


def _run_method(
    dataset: DAVISDataset,
    sequence_names: Sequence[str],
    cache: FeatureCache,
    *,
    method_name: str,
    method: AssociationMethod,
    config: Mapping[str, Any],
    calibrator: LogisticAssociationCalibrator | None = None,
    max_gap: int | None = None,
    confidence_threshold: float | None = None,
    motion_weights: Mapping[str, float] | None = None,
    frame_limit: int | None = None,
) -> MethodRun:
    """Run a single method over the same masked detections and cached features."""

    dataset_config = config["dataset"]
    feature_config = config["features"]
    memory_config = config["memory"]
    association_config = config["association"]
    gap = int(memory_config["max_gap"] if max_gap is None else max_gap)
    preprocessing = build_preprocessing_descriptor(
        int(feature_config["input_size"]),
        device=str(feature_config["device"]),
        use_mixed_precision=bool(feature_config["use_mixed_precision"]),
    )
    backbone_name = (
        f"{feature_config['torch_hub_repository']}:{feature_config['backbone']}"
    )
    frame_logs: list[FrameEvaluationLog] = []
    association_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    issues: list[DatasetIssue] = []
    processed_frames: dict[str, int] = {}

    for sequence_name in sequence_names:
        processed_frames[sequence_name] = 0
        memory = ObjectMemory(
            max_gap=gap,
            appearance_history_size=int(memory_config["appearance_history_size"]),
        )
        online = OnlineAssociator(
            memory,
            method,
            cosine_threshold=float(association_config["cosine_threshold"]),
            fixed_score_threshold=float(association_config["fixed_score_threshold"]),
            confidence_threshold=float(
                association_config["confidence_threshold"]
                if confidence_threshold is None
                else confidence_threshold
            ),
            motion_weights=motion_weights or association_config["weights"],
            calibrator=calibrator,
        )
        identity_gt_owner: dict[int, int] = {}
        for ordinal, frame in enumerate(
            _stream_frames(dataset, sequence_name, frame_limit, issues)
        ):
            processed_frames[sequence_name] += 1
            embeddings, missing = load_cached_frame_embeddings(
                frame,
                cache,
                backbone_name=backbone_name,
                preprocessing=preprocessing,
            )
            if missing:
                raise FileNotFoundError(
                    f"Missing cached features for {sequence_name}/{frame.frame_name}, "
                    f"objects {missing}; run scripts/extract_features.py first"
                )
            detections, gt_by_detection, invalid = build_object_detections(
                frame.mask, embeddings
            )
            if invalid:
                raise ValueError(
                    f"Objects lack valid embeddings for {sequence_name}/{frame.frame_name}: {invalid}"
                )
            owner_before = identity_gt_owner.copy()
            result = online.step(
                detections,
                image_size=frame.image.size,
                frame_index=_frame_index(frame.frame_name, ordinal),
            )
            frame_logs.append(
                FrameEvaluationLog(
                    sequence=sequence_name,
                    frame_name=frame.frame_name,
                    frame_index=_frame_index(frame.frame_name, ordinal),
                    gt_id_by_detection=gt_by_detection,
                    identity_gt_owner_before=owner_before,
                    result=result,
                )
            )
            accepted_identity_by_detection = {
                assignment.detection_id: assignment.predicted_identity_id
                for assignment in result.decision.accepted
            }
            candidate_count_by_detection = {detection_id: 0 for detection_id in gt_by_detection}
            eligible_count_by_detection = {detection_id: 0 for detection_id in gt_by_detection}
            for (predicted_id, detection_id), pair_outcome in result.decision.pair_outcomes.items():
                candidate_count_by_detection[detection_id] += 1
                if pair_outcome != "below_threshold":
                    eligible_count_by_detection[detection_id] += 1
            for detection_id, outcome in result.decision.detection_outcomes.items():
                detection_rows.append(
                    {
                        "method": method_name,
                        "sequence": sequence_name,
                        "frame": frame.frame_name,
                        "detection_id": detection_id,
                        "outcome": outcome,
                        "candidate_pair_count": candidate_count_by_detection[detection_id],
                        "eligible_candidate_pair_count": eligible_count_by_detection[detection_id],
                        "accepted_identity_id": accepted_identity_by_detection.get(detection_id),
                        "new_predicted_identity_id": result.new_identity_by_detection.get(
                            detection_id
                        ),
                    }
                )
            selected = {
                (assignment.predicted_identity_id, assignment.detection_id): assignment
                for assignment in result.decision.assignments
            }
            for pair, feature in result.pair_features.items():
                predicted_id, detection_id = pair
                assignment = selected.get(pair)
                score = result.decision.candidate_scores[pair]
                row = {
                    "method": method_name,
                    "sequence": sequence_name,
                    "frame": frame.frame_name,
                    "predicted_identity_id": predicted_id,
                    "detection_id": detection_id,
                    "pair_score": score,
                    "score_kind": "probability" if method == "calibrated_selective" else "association_score",
                    "hungarian_selected": assignment is not None,
                    "accepted_for_memory": bool(assignment and assignment.accepted),
                    "pair_outcome": result.decision.pair_outcomes[pair],
                    "detection_outcome": result.decision.detection_outcomes[detection_id],
                    "detection_rejected": result.decision.detection_outcomes[detection_id]
                    == "confidence_rejection",
                    "detection_assignment_conflict": result.decision.detection_outcomes[detection_id]
                    == "assignment_conflict",
                    "pair_correct": int(
                        owner_before.get(predicted_id) == gt_by_detection[detection_id]
                    ),
                }
                row.update(feature.to_dict())
                association_rows.append(row)

            for expired_id in result.expired_identity_ids:
                identity_gt_owner.pop(expired_id, None)
            for detection_id, predicted_id in result.predicted_identity_by_detection.items():
                identity_gt_owner[predicted_id] = gt_by_detection[detection_id]

    per_sequence, aggregate = evaluate_sequences(frame_logs)
    reported_sequences = {row["sequence"] for row in per_sequence}
    for sequence_name in sequence_names:
        if sequence_name not in reported_sequences:
            per_sequence.append(
                {
                    "sequence": sequence_name,
                    "identity_switches": 0,
                    "track_fragmentations": 0,
                    "correctly_associated_identity_transitions": 0,
                    "correct_associations": 0,
                    "incorrect_associations": 0,
                    "rejected_associations": 0,
                    "confidence_rejections": 0,
                    "assignment_conflicts": 0,
                    "no_candidate_detections": 0,
                    "new_predicted_identities": 0,
                    "eligible_gt_transitions": 0,
                    "association_assignment_decisions": 0,
                    "accepted_associations": 0,
                    "association_precision": float("nan"),
                    "association_recall": float("nan"),
                    "association_f1": float("nan"),
                    "association_coverage": float("nan"),
                    "false_association_rate": float("nan"),
                    "brier_score": float("nan"),
                    "ece": float("nan"),
                    "calibration_candidate_pairs": 0,
                    "association_candidate_pairs": 0,
                    "calibration_positive_candidate_pairs": 0,
                    "calibration_negative_candidate_pairs": 0,
                    "calibration_status": "not_applicable_no_processed_frames",
                    "occlusion_gaps": [],
                    "candidate_labels": [],
                    "candidate_probabilities": [],
                }
            )
    per_sequence.sort(key=lambda row: row["sequence"])
    for row in per_sequence:
        row["method"] = method_name
        row["frames_processed"] = processed_frames[row["sequence"]]
        row["skipped"] = processed_frames[row["sequence"]] == 0
    aggregate["method"] = method_name
    aggregate["sequence_count"] = len(sequence_names)
    aggregate["skipped_sequence_count"] = sum(
        count == 0 for count in processed_frames.values()
    )
    return MethodRun(
        method_name=method_name,
        per_sequence_metrics=per_sequence,
        aggregate_metrics=aggregate,
        frame_logs=frame_logs,
        association_rows=association_rows,
        detection_rows=detection_rows,
        issues=issues,
    )


def _training_streams(
    dataset: DAVISDataset,
    sequence_names: Sequence[str],
    frame_limit: int | None,
    issue_log: list[DatasetIssue],
):
    for sequence_name in sequence_names:
        local_issues: list[DatasetIssue] = []

        def frames(name=sequence_name, sink=local_issues):
            yield from _stream_frames(dataset, name, frame_limit, sink)
            issue_log.extend(sink)

        yield sequence_name, frames()


def _fit_for_gap(
    dataset: DAVISDataset,
    train_names: Sequence[str],
    val_names: Sequence[str],
    cache: FeatureCache,
    config: Mapping[str, Any],
    gap: int,
    frame_limit: int | None,
    issue_log: list[DatasetIssue],
) -> tuple[CalibrationCandidates, LogisticAssociationCalibrator]:
    features_config = config["features"]
    memory_config = config["memory"]
    candidates = generate_calibration_candidates(
        _training_streams(dataset, train_names, frame_limit, issue_log),
        cache,
        backbone_name=f"{features_config['torch_hub_repository']}:{features_config['backbone']}",
        preprocessing=build_preprocessing_descriptor(
            int(features_config["input_size"]),
            device=str(features_config["device"]),
            use_mixed_precision=bool(features_config["use_mixed_precision"]),
        ),
        max_gap=gap,
        appearance_history_size=int(memory_config["appearance_history_size"]),
        memory_policy="cosine_threshold",
        cosine_threshold=float(config["association"]["cosine_threshold"]),
    )
    calibrator = fit_logistic_calibrator(
        candidates,
        training_sequences=train_names,
        validation_sequences=val_names,
        class_weight=config["calibration"]["class_weight"],
        max_iter=int(config["calibration"]["max_iter"]),
        random_state=int(config["calibration"]["random_state"]),
    )
    calibrator.metadata["validation_sequences_checked"] = list(val_names)
    calibrator.metadata["max_memory_gap"] = gap
    return candidates, calibrator


def run_experiment_suite(
    dataset: DAVISDataset,
    cache: FeatureCache,
    config: Mapping[str, Any],
    *,
    train_sequence_names: Sequence[str] | None = None,
    validation_sequence_names: Sequence[str] | None = None,
    frame_limit: int | None = None,
    training_candidates: CalibrationCandidates | None = None,
    fitted_calibrator: LogisticAssociationCalibrator | None = None,
) -> dict[str, Any]:
    """Run the main comparison, requested ablations, and validation analyses."""

    train_names = list(
        dataset.train_sequences if train_sequence_names is None else train_sequence_names
    )
    val_names = list(
        dataset.validation_sequences
        if validation_sequence_names is None
        else validation_sequence_names
    )
    invalid_train_names = set(train_names) - set(dataset.train_sequences)
    invalid_validation_names = set(val_names) - set(dataset.validation_sequences)
    if invalid_train_names:
        raise ValueError(
            "Calibration training sequences must belong to the official training split: "
            f"{sorted(invalid_train_names)}"
        )
    if invalid_validation_names:
        raise ValueError(
            "Evaluation sequences must belong to the official validation split: "
            f"{sorted(invalid_validation_names)}"
        )
    if set(train_names) & set(val_names):
        raise ValueError("Training and validation sequences must be disjoint")
    all_issues: list[DatasetIssue] = []
    if (training_candidates is None) != (fitted_calibrator is None):
        raise ValueError("Pass both training_candidates and fitted_calibrator, or neither")
    if training_candidates is None:
        train_candidates, calibrator = _fit_for_gap(
            dataset,
            train_names,
            val_names,
            cache,
            config,
            gap=int(config["memory"]["max_gap"]),
            frame_limit=frame_limit,
            issue_log=all_issues,
        )
    else:
        train_candidates = training_candidates
        calibrator = fitted_calibrator
        if train_candidates.sequence_names - set(train_names):
            raise ValueError("Provided calibration candidates contain non-training sequences")
        expected_generation_policy = (
            "cosine_threshold;cosine_threshold="
            f"{float(config['association']['cosine_threshold']):g}"
        )
        if train_candidates.generation_policy != expected_generation_policy:
            raise ValueError(
                "Provided calibration candidates use a different memory replay policy: "
                f"expected {expected_generation_policy!r}, "
                f"got {train_candidates.generation_policy!r}"
            )
        if set(calibrator.metadata.get("training_sequences", ())) != set(train_names):
            raise ValueError("Provided calibrator training sequences do not match this run")
        if calibrator.metadata.get("validation_sequences_used_for_fit"):
            raise ValueError("Provided calibrator metadata shows validation data was used for fitting")
        calibrator_policy = calibrator.metadata.get("candidate_generation_policy")
        if calibrator_policy is not None and calibrator_policy != expected_generation_policy:
            raise ValueError(
                "Provided calibrator was fit with a different candidate memory replay policy: "
                f"expected {expected_generation_policy!r}, got {calibrator_policy!r}"
            )

    association_config = config["association"]
    base_weights = dict(association_config["weights"])
    method_specs = [
        ("cosine_forced", "cosine_forced", None, None),
        ("cosine_threshold", "cosine_threshold", None, None),
        ("motion_appearance", "motion_appearance", None, base_weights),
        ("calibrated_selective", "calibrated_selective", calibrator, None),
    ]
    runs: dict[str, MethodRun] = {}
    association_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    per_sequence_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for output_name, method, model, weights in method_specs:
        run = _run_method(
            dataset,
            val_names,
            cache,
            method_name=output_name,
            method=method,  # type: ignore[arg-type]
            config=config,
            calibrator=model,
            motion_weights=weights,
            frame_limit=frame_limit,
        )
        runs[output_name] = run
        metrics_rows.append(run.aggregate_metrics)
        per_sequence_rows.extend(run.per_sequence_metrics)
        association_rows.extend(run.association_rows)
        detection_rows.extend(run.detection_rows)
        all_issues.extend(run.issues)

    proposed_logs = runs["calibrated_selective"].frame_logs
    val_labels, val_probabilities = candidate_targets_and_probabilities(proposed_logs)
    calibration_report = calibration_metrics(
        val_labels,
        val_probabilities,
        ece_bins=int(config["evaluation"]["ece_bins"]),
    )
    calibration_report["evaluation_sequences"] = val_names
    calibration_report["calibrator_training_sequences"] = train_names
    calibration_report["feature_names"] = list(calibrator.metadata["feature_names"])
    calibration_report["metric_scope"] = "pooled_validation_candidate_pairs"
    calibration_report["probability_interpretation"] = (
        "Logistic-regression probability estimates; calibration is assessed on held-out data."
    )
    curve = precision_coverage_curve(
        val_labels,
        val_probabilities,
        config["evaluation"]["confidence_thresholds"],
    )
    assignment_curve = assignment_precision_coverage_curve(
        proposed_logs,
        config["evaluation"]["confidence_thresholds"],
    )

    appearance_ablation_specs = [
        ("appearance_only", {"appearance": 1.0, "center_displacement": 0.0, "bbox_iou": 0.0, "area_ratio": 0.0}),
        ("appearance_plus_center", {"appearance": 1.0, "center_displacement": 0.25, "bbox_iou": 0.0, "area_ratio": 0.0}),
        ("appearance_center_iou", {"appearance": 1.0, "center_displacement": 0.25, "bbox_iou": 0.25, "area_ratio": 0.0}),
    ]
    appearance_rows = []
    for name, weights in appearance_ablation_specs:
        run = _run_method(
            dataset,
            val_names,
            cache,
            method_name=name,
            method="motion_appearance",
            config=config,
            motion_weights=weights,
            frame_limit=frame_limit,
        )
        row = dict(run.aggregate_metrics)
        row.update({"ablation": "appearance_vs_motion", "weights": weights})
        appearance_rows.append(row)
        per_sequence_rows.extend(run.per_sequence_metrics)
        all_issues.extend(run.issues)

    memory_rows = []
    memory_gap_values = [int(value) for value in config["evaluation"]["memory_gaps"]]
    main_gap = int(config["memory"]["max_gap"])
    for gap in memory_gap_values:
        if gap == main_gap:
            gap_calibrator = calibrator
        else:
            _, gap_calibrator = _fit_for_gap(
                dataset,
                train_names,
                val_names,
                cache,
                config,
                gap=gap,
                frame_limit=frame_limit,
                issue_log=all_issues,
            )
        run = _run_method(
            dataset,
            val_names,
            cache,
            method_name="calibrated_selective",
            method="calibrated_selective",
            config=config,
            calibrator=gap_calibrator,
            max_gap=gap,
            frame_limit=frame_limit,
        )
        row = dict(run.aggregate_metrics)
        row["max_memory_gap"] = gap
        memory_rows.append(row)
        all_issues.extend(run.issues)

    main_aggregate = {row["method"]: row for row in metrics_rows}
    calibration_ablation = [
        {
            "comparison": "raw_cosine_threshold",
            **main_aggregate["cosine_threshold"],
            "brier_score": None,
            "ece": None,
            "calibration_analysis": "not applicable: raw cosine is not a probability",
        },
        {
            "comparison": "logistic_probability_selective",
            **main_aggregate["calibrated_selective"],
            "brier_score": calibration_report["brier_score"],
            "ece": calibration_report["ece"],
            "calibration_analysis": "held-out validation Brier/ECE reported separately",
        },
    ]
    gap_results = {
        name: occlusion_gap_metrics(run.per_sequence_metrics)
        for name, run in runs.items()
    }
    for row in metrics_rows:
        if row["method"] == "calibrated_selective":
            row["brier_score"] = calibration_report["brier_score"]
            row["ece"] = calibration_report["ece"]
        else:
            row["brier_score"] = float("nan")
            row["ece"] = float("nan")
    return {
        "metrics": metrics_rows,
        "per_sequence_metrics": per_sequence_rows,
        "association_predictions": association_rows,
        "detection_decisions": detection_rows,
        "calibration_candidates": [
            {
                "sequence": log.sequence,
                "frame": log.frame_name,
                "predicted_identity_id": predicted_id,
                "detection_id": detection_id,
                "correct": int(
                    log.identity_gt_owner_before.get(predicted_id)
                    == log.gt_id_by_detection[detection_id]
                ),
                "probability": log.result.decision.candidate_scores[(predicted_id, detection_id)],
            }
            for log in proposed_logs
            for predicted_id, detection_id in log.result.pair_features
        ],
        "training_calibration_candidates": [
            {
                **metadata,
                "correct": int(label),
                **{
                    feature_name: float(value)
                    for feature_name, value in zip(FEATURE_NAMES, feature_row.tolist())
                },
            }
            for metadata, label, feature_row in zip(
                train_candidates.metadata,
                train_candidates.labels.tolist(),
                train_candidates.features,
            )
        ],
        "calibration_metrics": calibration_report,
        "precision_coverage_curve": curve,
        "precision_coverage_curve_level": "candidate_pair",
        "assignment_precision_coverage_curve": assignment_curve,
        "occlusion_gap_metrics": gap_results,
        "appearance_ablation": appearance_rows,
        "calibration_ablation": calibration_ablation,
        "memory_duration_ablation": memory_rows,
        "train_candidate_summary": {
            "candidate_pairs": int(train_candidates.labels.size),
            "positive_candidates": int(np.sum(train_candidates.labels == 1)),
            "negative_candidates": int(np.sum(train_candidates.labels == 0)),
            "sequences": sorted(train_candidates.sequence_names),
            "candidate_generation_policy": train_candidates.generation_policy,
            "memory_policy_note": (
                "Training replay uses thresholded cosine memory to include rejection-induced "
                "new identities. Inference uses calibrated probability decisions, so memory "
                "states can still differ when geometric features change those decisions."
            ),
        },
        "calibrator_metadata": calibrator.metadata,
        "calibrator": calibrator,
        "dataset_issues": [
            issue.__dict__
            for issue in {
                (
                    issue.severity,
                    issue.code,
                    issue.message,
                    issue.sequence,
                    issue.frame,
                    issue.path,
                ): issue
                for issue in all_issues
            }.values()
        ],
        "sequence_split": {"train": train_names, "validation": val_names},
        "smoke_test": frame_limit is not None,
    }
