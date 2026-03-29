#!/usr/bin/env python3
"""Generate the current report-ready quantitative figure pack."""

from __future__ import annotations

from pathlib import Path

from plot_cross_dataset_transfer import plot as plot_cross_dataset_transfer
from plot_cross_dataset_transfer import _load_rows as load_cross_rows
from plot_stability_metrics import plot as plot_stability
from plot_stability_metrics import _load_summary as load_stability_rows


def main() -> None:
    report_dir = Path("artifacts/figures/report")
    report_dir.mkdir(parents=True, exist_ok=True)

    plot_stability(
        load_stability_rows(Path("artifacts/reports/stability_summary.csv")),
        report_dir / "offline_stability_comparison.png",
    )
    plot_cross_dataset_transfer(
        load_cross_rows(Path("artifacts/reports/cross_dataset_summary.csv")),
        report_dir / "cross_dataset_transfer_summary.png",
    )

    print("[ok] generated report figure pack")
    print(f"[info] output_dir={report_dir}")


if __name__ == "__main__":
    main()
