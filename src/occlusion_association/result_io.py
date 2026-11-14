"""Structured experiment table/metadata export helpers."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import write_json, write_yaml
from .visualization import generate_plots


def _write_plot_inputs(results: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results["metrics"]).to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(results["precision_coverage_curve"]).to_csv(
        directory / "precision_coverage_curve.csv", index=False
    )
    pd.DataFrame(results["memory_duration_ablation"]).to_csv(
        directory / "memory_duration_ablation.csv", index=False
    )
    pd.DataFrame(results["tracking_confidence_threshold_metrics"]).to_csv(
        directory / "tracking_confidence_thresholds.csv", index=False
    )
    write_json(directory / "calibration_metrics.json", results["calibration_metrics"])
    write_json(directory / "occlusion_gap_metrics.json", results["occlusion_gap_metrics"])


def _write_summary(results: dict[str, Any], directory: Path) -> None:
    lines = [
        "# Experiment summary",
        "",
        f"Smoke-test mode: **{results.get('smoke_test', False)}**",
        "",
        "Ground-truth DAVIS masks were used as detections. Segmentation quality was not evaluated.",
        "",
        "## Main validation results",
        "",
        "| Method | Identity switches | Fragmentations | Association precision | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["metrics"]:
        precision = row.get("association_precision")
        coverage = row.get("association_coverage")
        precision_text = f"{precision:.4f}" if precision is not None and pd.notna(precision) else "NA"
        coverage_text = f"{coverage:.4f}" if coverage is not None and pd.notna(coverage) else "NA"
        lines.append(
            f"| {row['method']} | {row['identity_switches']} | {row['track_fragmentations']} | "
            f"{precision_text} | {coverage_text} |"
        )
    lines.extend(
        [
            "",
            "## Full-tracking confidence threshold sweep",
            "",
            "Each row replays the complete validation sequences with a fresh identity memory at that threshold. These validation results are diagnostic and do not select the operating threshold.",
            "",
            "| Confidence threshold | Identity switches | Fragmentations | Association precision | Coverage |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["tracking_confidence_threshold_metrics"]:
        precision = row.get("association_precision")
        coverage = row.get("association_coverage")
        precision_text = f"{precision:.4f}" if precision is not None and pd.notna(precision) else "NA"
        coverage_text = f"{coverage:.4f}" if coverage is not None and pd.notna(coverage) else "NA"
        lines.append(
            f"| {row['confidence_threshold']:.3f} | {row['identity_switches']} | "
            f"{row['track_fragmentations']} | {precision_text} | {coverage_text} |"
        )
    calibration = results["calibration_metrics"]
    lines.extend(
        [
            "",
            "## Held-out calibration",
            "",
            f"Candidate pairs: {calibration.get('n_candidates')}",
            f"Brier score: {calibration.get('brier_score')}",
            f"ECE: {calibration.get('ece')}",
            "",
            "Raw cosine is a similarity score, not a probability; Brier and ECE are reported only for logistic probabilities.",
            "The candidate-pair precision/coverage curve thresholds pairs independently. The assignment-level curve reruns one-to-one assignment at each cutoff on the logged frame states.",
            "Confidence rejection means no candidate met threshold; assignment conflict means an eligible candidate lost the global one-to-one assignment. These outcomes are counted separately.",
            "",
            "## Scope and limits",
            "",
            "Ground-truth masks are detections; segmentation is not evaluated. Validation sequences were held out from calibrator fitting. The current TrainVal configuration has no independent labeled test split, so these are validation results. These results do not establish generalization beyond DAVIS or publication-level novelty.",
        ]
    )
    (directory / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_plots_from_results(
    results: dict[str, Any],
    output_directory: str | Path,
) -> list[Path]:
    """Plot in-memory experiment outputs through the standard plotting code."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="occlusion_association_plots_") as temp_dir:
        temp_root = Path(temp_dir)
        _write_plot_inputs(results, temp_root)
        generated = generate_plots(temp_root)
        figure_directory = destination / "figures"
        figure_directory.mkdir(parents=True, exist_ok=True)
        copied = []
        for source in generated:
            target = figure_directory / source.name
            shutil.copy2(source, target)
            copied.append(target)
        return copied


def export_experiment_results(
    results: dict[str, Any],
    experiment_directory: str | Path,
    *,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    make_plots: bool = True,
) -> Path:
    """Write metrics, candidates, ablations, calibration, model, and figures."""

    directory = Path(experiment_directory)
    directory.mkdir(parents=True, exist_ok=True)
    log_directory = directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / "run.log"
    if not log_path.exists():
        log_path.write_text("Experiment results exported.\n", encoding="utf-8")
    if config is not None:
        write_yaml(directory / "config.yaml", config)
    if metadata is not None:
        write_json(directory / "metadata.json", metadata)

    metrics = [dict(row) for row in results["metrics"]]
    calibration = results["calibration_metrics"]
    for row in metrics:
        if row["method"] == "calibrated_selective":
            row["brier_score"] = calibration["brier_score"]
            row["ece"] = calibration["ece"]
        else:
            row["brier_score"] = None
            row["ece"] = None
    pd.DataFrame(metrics).to_csv(directory / "metrics.csv", index=False)

    per_sequence = [
        {
            key: value
            for key, value in row.items()
            if key not in {"occlusion_gaps", "candidate_labels", "candidate_probabilities"}
        }
        for row in results["per_sequence_metrics"]
    ]
    pd.DataFrame(per_sequence).to_csv(directory / "per_sequence_metrics.csv", index=False)
    pd.DataFrame(results["association_predictions"]).to_csv(
        directory / "association_predictions.csv", index=False
    )
    pd.DataFrame(results["detection_decisions"]).to_csv(
        directory / "detection_decisions.csv", index=False
    )
    pd.DataFrame(results["calibration_candidates"]).to_csv(
        directory / "calibration_candidates.csv", index=False
    )
    pd.DataFrame(results["training_calibration_candidates"]).to_csv(
        directory / "training_calibration_candidates.csv", index=False
    )
    pd.DataFrame(results["precision_coverage_curve"]).to_csv(
        directory / "precision_coverage_curve.csv", index=False
    )
    pd.DataFrame(results["assignment_precision_coverage_curve"]).to_csv(
        directory / "assignment_precision_coverage_curve.csv", index=False
    )
    pd.DataFrame(results["appearance_ablation"]).to_csv(
        directory / "appearance_ablation.csv", index=False
    )
    pd.DataFrame(results["calibration_ablation"]).to_csv(
        directory / "calibration_ablation.csv", index=False
    )
    pd.DataFrame(results["memory_duration_ablation"]).to_csv(
        directory / "memory_duration_ablation.csv", index=False
    )
    pd.DataFrame(results["tracking_confidence_threshold_metrics"]).to_csv(
        directory / "tracking_confidence_thresholds.csv", index=False
    )
    pd.DataFrame(results["tracking_confidence_threshold_per_sequence_metrics"]).to_csv(
        directory / "tracking_confidence_thresholds_per_sequence.csv", index=False
    )
    write_json(directory / "calibration_metrics.json", calibration)
    write_json(directory / "occlusion_gap_metrics.json", results["occlusion_gap_metrics"])
    write_json(directory / "dataset_loading_issues.json", results["dataset_issues"])
    write_json(directory / "calibrator_metadata.json", results["calibrator_metadata"])
    write_json(directory / "train_candidate_summary.json", results["train_candidate_summary"])
    results["calibrator"].save(directory / "calibrator.joblib")
    _write_summary(results, directory)
    if make_plots:
        generate_plots_from_results(results, directory)
    return directory
