"""Measure weighted correlation between sliding patch averages in Poisson posteriors."""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "vendor" / "PosteriorBench"))

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from poisson_reference_correlations import weighted_lag_correlations, write_csv
from posteriorbench.datasets import get_dataset


WINDOW_SIZES = (1, 4, 8, 16)
WINDOW_COLORS = {1: "#414d59", 4: "#08748a", 8: "#bf651b", 16: "#7b3d9d"}


def summed_area_tables(fields):
    """Store each field's cumulative sum with a zero row and column at the origin."""
    sample_count, height, width = fields.shape
    totals = np.zeros((sample_count, height + 1, width + 1), dtype=np.float64)
    totals[:, 1:, 1:] = fields
    np.cumsum(totals, axis=1, out=totals)
    np.cumsum(totals, axis=2, out=totals)
    return totals


def sliding_patch_means(totals, size):
    """Average each square window that fits entirely in the field, at stride one."""
    window_sums = totals[:, size:, size:] - totals[:, :-size, size:]
    window_sums -= totals[:, size:, :-size]
    window_sums += totals[:, :-size, :-size]
    return window_sums / (size * size)


def summarize_curves(curves):
    summary = []
    for (size, direction), cases in curves.items():
        low, median, high = np.quantile(cases, [0.1, 0.5, 0.9], axis=0)
        for distance, (p10, p50, p90) in enumerate(zip(low, median, high), start=size):
            summary.append({
                "window_size_pixels": size,
                "direction": direction,
                "center_distance_pixels": distance,
                "cases": len(cases),
                "p10": p10,
                "median": p50,
                "p90": p90,
            })
    return summary


def plot_curves(path, summary, case_count):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10.5, 4.3), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharex=True, sharey=True)
    for axis, direction in zip(axes, ("horizontal", "vertical")):
        for size in WINDOW_SIZES:
            rows = [row for row in summary if row["window_size_pixels"] == size
                    and row["direction"] == direction]
            distances = [row["center_distance_pixels"] for row in rows]
            color = WINDOW_COLORS[size]
            label = "1 × 1 pixel" if size == 1 else f"{size} × {size} pixels"
            axis.fill_between(distances, [row["p10"] for row in rows],
                              [row["p90"] for row in rows], color=color, alpha=0.13,
                              edgecolor="none")
            axis.plot(distances, [row["median"] for row in rows], color=color,
                      linestyle="--" if size == 1 else "-", linewidth=1.9, label=label)
        axis.axhline(0, color="0.55", linewidth=0.7, zorder=0)
        axis.set_xlim(1, max(row["center_distance_pixels"] for row in summary))
        axis.set_title(f"{direction.capitalize()} window pairs", loc="left")
        axis.set_xlabel("Center-to-center distance (pixels)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean weighted Pearson correlation")
    axes[1].legend(frameon=False, loc="upper right", title="Window size")
    figure.suptitle(f"Patch correlations in {case_count} Poisson posteriors\n"
                    "Lines show case medians and shading shows the middle 80% of cases")
    figure.savefig(path, dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-source", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    dataset = get_dataset("poisson")
    cases = dataset.discover(args.case_source)
    records = []
    curves = defaultdict(list)
    for index, case in enumerate(cases, start=1):
        reference = dataset.load_reference(case)
        fields = reference.samples
        if 2 * max(WINDOW_SIZES) > min(fields.shape[1:]):
            raise ValueError(f"Field grid is too small for non-overlapping windows in {case.stem}")
        totals = summed_area_tables(fields)
        for size in WINDOW_SIZES:
            means = fields if size == 1 else sliding_patch_means(totals, size)
            for direction, (correlations, pair_counts) in weighted_lag_correlations(
                means, reference.weights
            ).items():
                # Windows start overlapping when their center separation is smaller than their width.
                non_overlapping = slice(size - 1, None)
                correlations = correlations[non_overlapping]
                pair_counts = pair_counts[non_overlapping]
                curves[size, direction].append(correlations)
                for distance, (correlation, count) in enumerate(zip(correlations, pair_counts),
                                                                 start=size):
                    records.append({
                        "case_id": case.stem,
                        "split": case.split,
                        "window_size_pixels": size,
                        "direction": direction,
                        "center_distance_pixels": distance,
                        "mean_correlation": correlation,
                        "window_pairs": count,
                    })
        if index % 20 == 0 or index == len(cases):
            print(f"Measured {index}/{len(cases)} cases", flush=True)

    summary = summarize_curves(curves)
    args.output.mkdir(parents=True)
    write_csv(args.output / "per_case.csv", records)
    write_csv(args.output / "summary.csv", summary)
    plot_curves(args.output / "patch_correlation_curves.png", summary, len(cases))
    print(f"Wrote {len(records)} correlation estimates to {args.output}", flush=True)


if __name__ == "__main__":
    main()
