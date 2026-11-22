"""Measure PosteriorBench detection rates for controlled smooth-field changes."""

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
from scipy.stats import binomtest

from poisson_reference_correlations import write_csv
from poisson_smooth_fields import draw_fields, sampling_amplitude
from posteriorbench.evaluation.metrics import evaluate_distribution


METRICS = {
    "mean_rel_l2": ("Mean error", "#657789"),
    "std_rel_l2": ("Standard deviation error", "#b36725"),
    "mmd": ("MMD", "#8e2d53"),
    "swd": ("Sliced Wasserstein", "#247675"),
    "spectral_rel_geomean": ("Radial spectral error", "#4254a1"),
}
FAMILIES = ("shorter range", "longer range", "direction", "anticorrelation")
ENSEMBLE_SIZE = 100
NULL_QUANTILE = 0.95
MIN_DETECTION_RATE = 0.80
WEIGHTS = np.full(ENSEMBLE_SIZE, 1 / ENSEMBLE_SIZE)


def initialize_worker(spectra_path, config):
    global amplitudes, normalization
    with np.load(spectra_path) as spectra:
        amplitudes = {name: sampling_amplitude(spectra[name], config["marginal_std"])
                      for name in spectra.files}
    normalization = {key: config[key] for key in ("swd_normalization_mean", "swd_normalization_std")}


def score_trial(job):
    model, model_index, trial, seed = job
    rng = np.random.default_rng(np.random.SeedSequence([seed, model_index, trial]))
    anchor = draw_fields(amplitudes["baseline"], rng, ENSEMBLE_SIZE)
    candidate = draw_fields(amplitudes[model], rng, ENSEMBLE_SIZE)
    scores = evaluate_distribution(candidate, anchor, WEIGHTS, WEIGHTS, **normalization)
    if not np.isfinite(list(scores.values())).all():
        raise ValueError(f"Nonfinite score for {model} trial {trial}")
    return {"variant": model, "trial": trial, **scores}


def family_of(variant):
    if variant["pattern"] == "range":
        return "shorter range" if variant["strength"] < 0 else "longer range"
    return variant["pattern"]


def detection_rates(variants, scores, cutoffs):
    rows = []
    for variant in variants[1:]:
        comparisons = scores[variant["name"]]
        for metric, cutoff in cutoffs.items():
            hits = sum(row[metric] > cutoff for row in comparisons)
            interval = binomtest(hits, len(comparisons)).proportion_ci(method="wilson")
            rows.append({
                "family": family_of(variant), "variant": variant["name"],
                "strength": abs(variant["strength"]), "metric": metric,
                "detections": hits, "comparisons": len(comparisons),
                "rate": hits / len(comparisons), "ci95_low": interval.low, "ci95_high": interval.high,
            })
    return rows


def metric_thresholds(rates):
    output = []
    for family in FAMILIES:
        for metric in METRICS:
            rows = sorted((row for row in rates if row["family"] == family and row["metric"] == metric),
                          key=lambda row: row["strength"])
            best = max(rows, key=lambda row: row["rate"])
            first = next((index for index, row in enumerate(rows) if row["rate"] >= MIN_DETECTION_RATE), None)
            lower = rows[-1]["strength"] if first is None else (rows[first - 1]["strength"] if first else 0.)
            detected = rows[first] if first is not None else None
            output.append({
                "family": family, "metric": metric,
                "status": "reached" if detected else "not reached in tested range",
                "previous_strength_below_80pct": lower,
                "first_strength_at_or_above_80pct": detected["strength"] if detected else None,
                "first_detected_variant": detected["variant"] if detected else None,
                "rate_at_first_detection": detected["rate"] if detected else None,
                "highest_tested_strength": rows[-1]["strength"],
                "highest_detection_rate": best["rate"],
            })
    return output


def plot_detection(path, families, rates, title, xlabel, comparisons):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(9.8 if len(families) == 2 else 6.6, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(families), squeeze=False, sharey=True).ravel()
    for axis, family in zip(axes, families):
        for metric, (label, color) in METRICS.items():
            rows = sorted((row for row in rates if row["family"] == family and row["metric"] == metric),
                          key=lambda row: row["strength"])
            x = [row["strength"] for row in rows]
            axis.plot(x, [row["rate"] for row in rows], "o-", color=color, label=label)
            axis.fill_between(x, [row["ci95_low"] for row in rows], [row["ci95_high"] for row in rows],
                              color=color, alpha=0.12, edgecolor="none")
        axis.axhline(MIN_DETECTION_RATE, color="0.35", linestyle="--", linewidth=0.85)
        axis.axhline(1 - NULL_QUANTILE, color="0.6", linestyle=":", linewidth=0.85)
        if len(families) > 1:
            axis.set_title(family.capitalize(), loc="left")
        axis.set_ylim(0, 1.04)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Detection rate")
    axes[-1].legend(frameon=False, loc="center right", fontsize=8)
    figure.suptitle(f"{title}\n{comparisons} comparisons per strength · dashed line marks 80%")
    figure.supxlabel(xlabel)
    figure.savefig(path, dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--null-trials", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=841792)
    args = parser.parse_args()
    if min(args.workers, args.null_trials, args.trials) < 1 or args.seed < 0:
        parser.error("workers and trial counts must be positive, and the seed must be nonnegative")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    design = json.loads((args.design / "design.json").read_text())
    variants = design["variants"]
    args.output.mkdir(parents=True)
    scores = {}
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn"),
        initializer=initialize_worker, initargs=(args.design / "spectra.npz", design),
    ) as executor, (args.output / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "trial", *METRICS])
        writer.writeheader()
        for model_index, variant in enumerate(variants):
            name = variant["name"]
            trial_count = args.trials if model_index else args.null_trials
            jobs = ((name, model_index, trial, args.seed) for trial in range(trial_count))
            scores[name] = []
            print(f"Running {name} ({trial_count} comparisons)", flush=True)
            for count, row in enumerate(executor.map(score_trial, jobs, chunksize=4), start=1):
                writer.writerow(row)
                scores[name].append(row)
                if count % 100 == 0 or count == trial_count:
                    handle.flush()
                    print(f"  {count}/{trial_count} complete", flush=True)

    cutoffs = {key: float(np.quantile([row[key] for row in scores["baseline"]], NULL_QUANTILE))
               for key in METRICS}
    report = {
        "null_trials": args.null_trials, "trials_per_variant": args.trials, "seed": args.seed,
        "ensemble_size": ENSEMBLE_SIZE, "null_quantile": NULL_QUANTILE,
        "minimum_detection_rate": MIN_DETECTION_RATE, "cutoffs": cutoffs,
    }
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    rates = detection_rates(variants, scores, cutoffs)
    write_csv(args.output / "detection_rates.csv", rates)
    write_csv(args.output / "thresholds.csv", metric_thresholds(rates))
    plot_detection(args.output / "range_detection.png", ("shorter range", "longer range"), rates,
                   "Detection of changes in dependence range", "Shift in the 0.5 crossing (pixels)", args.trials)
    plot_detection(args.output / "direction_detection.png", ("direction",), rates,
                   "Detection of directional dependence", "Gap between the two 0.5 crossings (pixels)", args.trials)
    plot_detection(args.output / "anticorrelation_detection.png", ("anticorrelation",), rates,
                   "Detection of weaker long-range anticorrelation", "Rise in correlation at 50 pixels", args.trials)
    print(f"Wrote detection results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
