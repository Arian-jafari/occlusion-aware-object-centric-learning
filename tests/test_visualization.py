from __future__ import annotations

from pathlib import Path

from occlusion_association.visualization import generate_plots


def test_plotting_reports_missing_results_without_creating_placeholders(
    tmp_path: Path, capsys
) -> None:
    output = generate_plots(tmp_path / "not_run")

    assert output == []
    assert "Run scripts/run_experiments.py first" in capsys.readouterr().out
    assert not (tmp_path / "not_run" / "figures").exists()
