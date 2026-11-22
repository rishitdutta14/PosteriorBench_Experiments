"""Measure weighted spatial correlations in the published Poisson posteriors."""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "vendor" / "PosteriorBench"))

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from posteriorbench.datasets import get_dataset
from posteriorbench.evaluation.metrics import normalize_weights


def directional_pair_means(standardized, valid):
    """Average correlations for all pairs separated along the last axis."""
    width = standardized.shape[-1]
    fft_length = next_fast_len(2 * width - 1, real=True)
    lags = slice(1, width)

    # Zero padding gives linear correlations, so grid edges never wrap together.
    spectrum = rfft(standardized, n=fft_length, axis=-1)
    power = np.einsum("nlf,nlf->f", spectrum.real, spectrum.real)
    power += np.einsum("nlf,nlf->f", spectrum.imag, spectrum.imag)
    correlation_sums = irfft(power, n=fft_length)[lags]

    mask_spectrum = rfft(valid, n=fft_length, axis=-1)
    mask_power = np.sum(np.abs(mask_spectrum) ** 2, axis=0)
    pair_counts = np.rint(irfft(mask_power, n=fft_length)[lags]).astype(int)
    if np.any(pair_counts <= 0):
        raise ValueError("No valid pixel pairs at one or more distances")
    return correlation_sums / pair_counts, pair_counts


def weighted_lag_correlations(fields, weights):
    """Compute Pearson correlations across samples using their posterior weights."""
    fields = np.asarray(fields, dtype=np.float64)
    weights = normalize_weights(weights, len(fields))
    centered = fields - np.average(fields, axis=0, weights=weights)
    weighted_centered = centered * np.sqrt(weights)[:, None, None]
    deviations = np.linalg.norm(weighted_centered, axis=0)
    valid = deviations > 1e-12
    standardized = np.divide(weighted_centered, deviations, out=np.zeros_like(fields), where=valid)

    return {
        "horizontal": directional_pair_means(standardized, valid),
        "vertical": directional_pair_means(standardized.swapaxes(1, 2), valid.T),
    }


def summarize_curves(curves):
    summary = []
    for direction, cases in curves.items():
        low, median, high = np.quantile(cases, [0.1, 0.5, 0.9], axis=0)
        for distance, (p10, p50, p90) in enumerate(zip(low, median, high), start=1):
            summary.append({
                "direction": direction,
                "distance_pixels": distance,
                "cases": len(cases),
                "p10": p10,
                "median": p50,
                "p90": p90,
            })
    return summary


def plot_curves(path, curves, summary):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 4.1), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharey=True)
    for axis, (direction, cases) in zip(axes, curves.items()):
        rows = [row for row in summary if row["direction"] == direction]
        lags = [row["distance_pixels"] for row in rows]
        individual_lines = axis.plot(lags, np.array(cases).T, color="#627687", alpha=0.10,
                                     linewidth=0.7, zorder=1)
        interval = axis.fill_between(lags, [row["p10"] for row in rows],
                                     [row["p90"] for row in rows], color="#79bdc9",
                                     alpha=0.45, zorder=2)
        median_line, = axis.plot(lags, [row["median"] for row in rows], color="#154e63",
                                 linewidth=2, zorder=3)
        axis.axhline(0, color="0.45", linewidth=0.8, linestyle="--", zorder=0)
        axis.set_xlim(lags[0], lags[-1])
        axis.set_title(f"{direction.capitalize()} pixel pairs", loc="left")
        axis.set_xlabel("Distance (pixels)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean weighted Pearson correlation")
    axes[1].legend([median_line, interval, individual_lines[0]],
                   ["Case median", "Middle 80% of cases", "Individual cases"],
                   frameon=False, loc="upper right")
    case_count = len(next(iter(curves.values())))
    figure.suptitle(f"Spatial dependence in {case_count} published Poisson posteriors")
    figure.savefig(path, dpi=200)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
        for direction, (correlations, pair_counts) in weighted_lag_correlations(
            reference.samples, reference.weights
        ).items():
            curves[direction].append(correlations)
            for distance, (correlation, count) in enumerate(zip(correlations, pair_counts), start=1):
                records.append({
                    "case_id": case.stem,
                    "split": case.split,
                    "direction": direction,
                    "distance_pixels": distance,
                    "mean_correlation": correlation,
                    "pixel_pairs": count,
                })
        if index % 20 == 0 or index == len(cases):
            print(f"Measured {index}/{len(cases)} cases", flush=True)

    summary = summarize_curves(curves)
    args.output.mkdir(parents=True)
    write_csv(args.output / "per_case.csv", records)
    write_csv(args.output / "summary.csv", summary)
    plot_curves(args.output / "correlation_curves.png", curves, summary)
    print(f"Wrote {len(records)} correlation estimates to {args.output}", flush=True)


if __name__ == "__main__":
    main()
