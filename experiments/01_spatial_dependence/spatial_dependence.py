"""Measure PosteriorBench's response to changes in Poisson spatial dependence."""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "vendor" / "PosteriorBench"))

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from posteriorbench.datasets import get_dataset
from posteriorbench.evaluation.metrics import (
    DEFAULT_N_PROJECTIONS,
    DEFAULT_PROJECTION_SEED,
    evaluate_distribution,
)


BLOCK_SIZES = (16, 4, 1)
LAGS = (1, 4, 16, 32)
DIRECTIONS = ("horizontal", "vertical")
LAG_KEYS = tuple(f"{direction}_{lag}" for direction in DIRECTIONS for lag in LAGS)
METRIC_LABELS = {
    "mean_rel_l2": "Posterior mean error",
    "std_rel_l2": "Posterior standard deviation error",
    "mmd": "Maximum mean discrepancy",
    "swd": "Sliced Wasserstein distance",
    "spectral_rel_geomean": "Radial spectral error",
}
COLORS = {16: "#15616d", 4: "#9c5d12", 1: "#912c60"}
ENSEMBLE_SIZE = 100
BOOTSTRAP_DRAWS = 5000


def shuffle_blocks(fields, block_size, rng):
    """Permute sample identities independently inside each fixed spatial block."""
    sample_count, height, width = fields.shape
    shuffled = np.empty_like(fields)
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            rows = slice(row, row + block_size)
            cols = slice(col, col + block_size)
            shuffled[:, rows, cols] = fields[rng.permutation(sample_count), rows, cols]
    return shuffled


def lag_correlations(fields):
    """Average across-sample Pearson correlations for pairs at each directional lag."""
    centered = fields - fields.mean(axis=0)
    std = fields.std(axis=0)
    valid = std > 1e-12
    standardized = centered / np.where(valid, std, 1.0)
    result = {}
    for direction in DIRECTIONS:
        values = standardized if direction == "horizontal" else standardized.swapaxes(1, 2)
        mask = valid if direction == "horizontal" else valid.T
        for lag in LAGS:
            pairs = mask[:, :-lag] & mask[:, lag:]
            products = np.einsum("nhw,nhw->hw", values[:, :, :-lag], values[:, :, lag:])
            result[f"{direction}_{lag}"] = float(products[pairs].mean() / len(fields))
    return result


def score_case(job):
    case_id, case_index, fields, reference_weights, repeats, seed, normalization = job
    fields = fields.astype(np.float64)
    equal_weights = np.full(ENSEMBLE_SIZE, 1 / ENSEMBLE_SIZE)
    records = []

    for repeat in range(repeats):
        rng = np.random.default_rng(np.random.SeedSequence([seed, case_index, repeat]))
        anchor, baseline = fields[
            rng.choice(len(fields), size=(2, ENSEMBLE_SIZE), p=reference_weights)
        ]
        baseline_scores = evaluate_distribution(
            baseline, anchor, equal_weights, equal_weights, **normalization
        )
        baseline_corr = lag_correlations(baseline)
        sorted_baseline = np.sort(baseline, axis=0)

        for block_size in BLOCK_SIZES:
            shuffled = shuffle_blocks(baseline, block_size, rng)
            if not np.array_equal(np.sort(shuffled, axis=0), sorted_baseline):
                raise ValueError(f"A pixel's values changed for {case_id}, repeat {repeat}")
            shuffled_scores = evaluate_distribution(
                shuffled, anchor, equal_weights, equal_weights, **normalization
            )
            shuffled_corr = lag_correlations(shuffled)
            scores = list(baseline_scores.values()) + list(shuffled_scores.values())
            correlations = list(baseline_corr.values()) + list(shuffled_corr.values())
            if not np.isfinite(scores + correlations).all():
                raise ValueError(f"Nonfinite result for {case_id}, repeat {repeat}")

            record = {
                "case_id": case_id,
                "repeat": repeat,
                "block_size": block_size,
                "correlation_loss": float(np.mean([
                    abs(baseline_corr[key]) - abs(shuffled_corr[key]) for key in LAG_KEYS
                ])),
            }
            for key in METRIC_LABELS:
                record[f"baseline_{key}"] = baseline_scores[key]
                record[f"shuffled_{key}"] = shuffled_scores[key]
                record[f"change_{key}"] = shuffled_scores[key] - baseline_scores[key]
            for key in LAG_KEYS:
                record[f"baseline_corr_{key}"] = baseline_corr[key]
                record[f"shuffled_corr_{key}"] = shuffled_corr[key]
            records.append(record)

    return records


def average_repeats(records):
    groups = {}
    for record in records:
        groups.setdefault((record["case_id"], record["block_size"]), []).append(record)
    measures = [key for key in records[0] if key not in {"case_id", "block_size", "repeat"}]
    return [
        {
            "case_id": case_id,
            "block_size": block_size,
            **{key: float(np.mean([row[key] for row in rows])) for key in measures},
        }
        for (case_id, block_size), rows in groups.items()
    ]


def summarize_cases(case_rows, seed):
    rng = np.random.default_rng(seed)
    case_count = len(case_rows) // len(BLOCK_SIZES)
    bootstrap_cases = rng.integers(case_count, size=(BOOTSTRAP_DRAWS, case_count))
    summary = {}
    for block_size in BLOCK_SIZES:
        rows = [row for row in case_rows if row["block_size"] == block_size]
        stats = {}
        keys = ["correlation_loss", *(f"change_{key}" for key in METRIC_LABELS)]
        for key in keys:
            values = np.array([row[key] for row in rows])
            lower, upper = np.quantile(values[bootstrap_cases].mean(axis=1), [0.025, 0.975])
            stats[key] = {
                "mean": float(values.mean()),
                "ci95": [float(lower), float(upper)],
                "fraction_positive": float(np.mean(values > 0)),
                "max_absolute": float(np.abs(values).max()),
            }
        summary[str(block_size)] = stats
    return summary


def plot_metric_response(output, split, case_rows, summary):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2, sharex=True)
    for axis, (key, label) in zip(axes.flat, METRIC_LABELS.items()):
        axis.axhline(0, color="0.65", linewidth=0.8)
        for block_size, color in COLORS.items():
            rows = [row for row in case_rows if row["block_size"] == block_size]
            loss = summary[str(block_size)]["correlation_loss"]["mean"]
            metric = summary[str(block_size)][f"change_{key}"]
            axis.scatter(
                [row["correlation_loss"] for row in rows],
                [row[f"change_{key}"] for row in rows],
                s=12,
                alpha=0.23,
                color=color,
            )
            axis.vlines(loss, *metric["ci95"], color=color, linewidth=1.6)
            axis.scatter(loss, metric["mean"], s=42, color=color, edgecolor="white", zorder=3)
        axis.set_title(label, loc="left")
        axis.set_ylabel("Change in score")
        axis.grid(axis="y", alpha=0.2)
        axis.set_xlabel("Drop in absolute lag correlation")
        axis.tick_params(labelbottom=True)
    legend_axis = axes.flat[-1]
    legend_axis.set_axis_off()
    for block_size, color in COLORS.items():
        legend_axis.scatter([], [], color=color, label=f"{block_size} × {block_size} pixels")
    legend_axis.legend(title="Patch size", loc="center", frameon=False)
    figure.suptitle(f"Metric response on Poisson {split} cases\n"
                    "Large points show means with 95% case bootstrap intervals")
    figure.savefig(output / "metric_response.png", dpi=180)


def plot_lag_correlations(output, split, case_rows):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 3.6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharey=True)
    baseline_rows = [row for row in case_rows if row["block_size"] == BLOCK_SIZES[0]]
    for axis, direction in zip(axes, DIRECTIONS):
        baseline = [np.mean([row[f"baseline_corr_{direction}_{lag}"] for row in baseline_rows])
                    for lag in LAGS]
        axis.plot(LAGS, baseline, "o-", color="0.2", label="Unshuffled")
        for block_size, color in COLORS.items():
            rows = [row for row in case_rows if row["block_size"] == block_size]
            shuffled = [np.mean([row[f"shuffled_corr_{direction}_{lag}"] for row in rows])
                        for lag in LAGS]
            axis.plot(LAGS, shuffled, "o-", color=color, label=f"{block_size} × {block_size}")
        axis.set_title(direction.capitalize())
        axis.set_xlabel("Pixel separation")
        axis.set_xscale("log", base=2)
        axis.set_xticks(LAGS, labels=LAGS)
        axis.axhline(0, color="0.7", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean correlation")
    axes[1].legend(title="Patch size", frameon=False)
    figure.suptitle(f"Spatial dependence on Poisson {split} cases")
    figure.savefig(output / "lag_correlations.png", dpi=180)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-source", type=Path, default=Path("data"))
    parser.add_argument("--training-metadata", type=Path,
                        default=Path("data/PDEFieldDataset_hf/Poisson/metadata.json"))
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.seed < 0 or args.workers < 1:
        parser.error("repeats and workers must be positive, and seed must be nonnegative")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    metadata = json.loads(args.training_metadata.read_text())
    field = next(name for name, channels in metadata["field_channels"].items() if channels == ["f"])
    index = metadata["field_names"].index(field)
    normalization = {
        "swd_normalization_mean": metadata["stats"]["mean"][index],
        "swd_normalization_std": metadata["stats"]["std"][index],
    }
    dataset = get_dataset("poisson")
    cases = [(index, case) for index, case in enumerate(dataset.discover(args.case_source))
             if case.split == args.split]
    if not cases:
        parser.error(f"No cases in the {args.split} split")

    def jobs():
        for index, case in cases:
            reference = dataset.load_reference(case)
            yield (case.stem, index, reference.samples, reference.weights,
                   args.repeats, args.seed, normalization)

    records = []
    print(f"Scoring {len(cases)} {args.split} cases with {args.repeats} repeats", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as executor:
        for index, result in enumerate(executor.map(score_case, jobs()), start=1):
            records.extend(result)
            if index == 1 or index % 10 == 0 or index == len(cases):
                print(f"Finished {index}/{len(cases)} cases", flush=True)

    case_rows = average_repeats(records)
    summary = summarize_cases(case_rows, args.seed)
    args.output.mkdir(parents=True)
    write_csv(args.output / "per_repeat.csv", records)
    write_csv(args.output / "per_case.csv", case_rows)
    report = {
        "dataset": "poisson",
        "split": args.split,
        "cases": len(cases),
        "repeats_per_case": args.repeats,
        "ensemble_size": ENSEMBLE_SIZE,
        "seed": args.seed,
        "block_sizes": BLOCK_SIZES,
        "lags_in_pixels": LAGS,
        "correlation_loss": "mean drop in absolute mean Pearson correlation across directions and lags",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "swd_projections": DEFAULT_N_PROJECTIONS,
        "swd_projection_seed": DEFAULT_PROJECTION_SEED,
        "swd_normalization": normalization,
        "max_absolute_moment_score_changes_per_repeat": {
            key: max(abs(row[f"change_{key}"]) for row in records)
            for key in ("mean_rel_l2", "std_rel_l2")
        },
        "results": summary,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_metric_response(args.output, args.split, case_rows, summary)
    plot_lag_correlations(args.output, args.split, case_rows)
    print(f"Wrote {len(records)} comparisons to {args.output}", flush=True)


if __name__ == "__main__":
    main()
