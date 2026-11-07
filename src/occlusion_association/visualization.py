"""Publication-style PNG/PDF figures from exported experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_NAMES = {
    "cosine_forced": "Cosine forced",
    "cosine_threshold": "Cosine threshold",
    "motion_appearance": "Motion + appearance",
    "calibrated_selective": "Calibrated selective",
    "appearance_only": "Appearance only",
    "appearance_plus_center": "Appearance + center",
    "appearance_center_iou": "Appearance + center + IoU",
}


def _save_figure(figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")


def _apply_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.figsize": (7.2, 4.6),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def generate_plots(experiment_directory: str | Path) -> list[Path]:
    """Generate all standard figures, or report that experiments must run first."""

    experiment_directory = Path(experiment_directory)
    metrics_path = experiment_directory / "metrics.csv"
    calibration_path = experiment_directory / "calibration_metrics.json"
    curve_path = experiment_directory / "precision_coverage_curve.csv"
    gap_path = experiment_directory / "occlusion_gap_metrics.json"
    memory_path = experiment_directory / "memory_duration_ablation.csv"
    if not metrics_path.is_file():
        print(
            f"No experiment results found in {experiment_directory}. "
            "Run scripts/run_experiments.py first; no placeholder plots were created."
        )
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style(plt)
    figure_directory = experiment_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(metrics_path)
    created: list[Path] = []

    for column, title, ylabel, filename in (
        ("identity_switches", "Identity switches by association method", "Identity switches", "identity_switches"),
        ("track_fragmentations", "Track fragmentation by association method", "Track fragmentations", "fragmentations"),
        ("association_precision", "Association precision by method", "Association precision", "association_precision"),
        ("association_coverage", "Association coverage by method", "Association coverage", "association_coverage"),
        ("brier_score", "Held-out Brier score by probabilistic method", "Brier score (lower is better)", "brier_score"),
        ("ece", "Held-out expected calibration error by probabilistic method", "ECE (lower is better)", "ece"),
    ):
        if column not in metrics.columns:
            continue
        values = metrics.dropna(subset=[column])
        if values.empty:
            continue
        labels = [METHOD_NAMES.get(str(method), str(method)) for method in values["method"]]
        figure, axis = plt.subplots()
        axis.bar(labels, values[column].astype(float), color="#4C78A8")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Association method")
        axis.tick_params(axis="x", rotation=18)
        figure.tight_layout()
        destination = figure_directory / filename
        _save_figure(figure, destination)
        plt.close(figure)
        created.extend([destination.with_suffix(".png"), destination.with_suffix(".pdf")])

    if calibration_path.is_file():
        report = json.loads(calibration_path.read_text(encoding="utf-8"))
        bins = [item for item in report.get("reliability", []) if item["count"] > 0]
        if bins:
            confidence = [item["mean_confidence"] for item in bins]
            accuracy = [item["empirical_accuracy"] for item in bins]
            figure, axis = plt.subplots()
            axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
            axis.plot(confidence, accuracy, marker="o", color="#E45756", label="Calibrated selective")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_title("Reliability diagram on held-out validation candidate pairs")
            axis.set_xlabel("Mean predicted association probability")
            axis.set_ylabel("Empirical association accuracy")
            axis.legend(frameon=False)
            figure.tight_layout()
            destination = figure_directory / "reliability_diagram"
            _save_figure(figure, destination)
            plt.close(figure)
            created.extend([destination.with_suffix(".png"), destination.with_suffix(".pdf")])

    if curve_path.is_file():
        curve = pd.read_csv(curve_path).sort_values("coverage")
        if not curve.empty:
            figure, axis = plt.subplots()
            axis.plot(curve["coverage"], curve["precision"], marker="o", color="#F58518", label="Logistic probability")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.02)
            axis.set_title("Candidate-pair precision–coverage curve")
            axis.set_xlabel("Candidate coverage")
            axis.set_ylabel("Candidate association precision")
            axis.legend(frameon=False)
            figure.tight_layout()
            destination = figure_directory / "precision_coverage"
            _save_figure(figure, destination)
            plt.close(figure)
            created.extend([destination.with_suffix(".png"), destination.with_suffix(".pdf")])

    if gap_path.is_file():
        gap_report = json.loads(gap_path.read_text(encoding="utf-8"))
        figure, axis = plt.subplots()
        for method, points in gap_report.items():
            x = [item["gap_range"] for item in points]
            y = [item["identity_consistency"] for item in points]
            axis.plot(x, y, marker="o", label=METHOD_NAMES.get(method, method))
        axis.set_ylim(0, 1.02)
        axis.set_title("Natural reappearance identity consistency by absence gap")
        axis.set_xlabel("Frames absent between observations")
        axis.set_ylabel("Fraction retaining the same predicted identity")
        axis.legend(frameon=False)
        figure.tight_layout()
        destination = figure_directory / "occlusion_gap_performance"
        _save_figure(figure, destination)
        plt.close(figure)
        created.extend([destination.with_suffix(".png"), destination.with_suffix(".pdf")])

    if memory_path.is_file():
        memory = pd.read_csv(memory_path)
        if not memory.empty:
            figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
            axes[0].plot(memory["max_memory_gap"], memory["identity_switches"], marker="o", label="Calibrated selective")
            axes[0].set_title("Identity switches")
            axes[0].set_xlabel("Maximum missing-frame gap")
            axes[0].set_ylabel("Identity switches")
            axes[1].plot(memory["max_memory_gap"], memory["association_precision"], marker="o", label="Calibrated selective")
            axes[1].set_title("Association precision")
            axes[1].set_xlabel("Maximum missing-frame gap")
            axes[1].set_ylabel("Association precision")
            for axis in axes:
                axis.legend(frameon=False)
            figure.suptitle("Performance versus memory duration")
            figure.tight_layout()
            destination = figure_directory / "memory_duration"
            _save_figure(figure, destination)
            plt.close(figure)
            created.extend([destination.with_suffix(".png"), destination.with_suffix(".pdf")])

    return created
