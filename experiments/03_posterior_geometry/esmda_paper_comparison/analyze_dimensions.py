"""Measure dimension of the same Poisson ES-MDA ensembles scored by the released evaluator."""

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_experiment import RADII, ROOT, SEED, analyze, get_dataset, gram_matrix, normalize, pca
from matplotlib import pyplot as plt

N_SUBSET = 38
REPEATS = 8
LABELS = {"reference": "Weighted reference", "esmda": "ES-MDA"}
COLORS = {"reference": "#243444", "esmda": "#b65031"}


def plot(output, curves, subsamples):
    fig, ax = plt.subplots(1, 2, figsize=(10.3, 3.9), layout="constrained")
    modes = np.arange(1, 31)
    for name in LABELS:
        color = COLORS[name]
        full, mass = map(np.stack, zip(*curves[name]))
        q1, mid, q3 = np.quantile(full[:, :len(modes)], [.25, .5, .75], axis=0)
        ax[0].plot(modes, mid, color=color, label=LABELS[name])
        ax[0].fill_between(modes, q1, q3, color=color, alpha=.14)
        ax[0].plot(modes, np.median(subsamples[name], axis=0)[:len(modes)],
                   color=color, ls="--", lw=1.3)

        q1, mid, q3 = np.quantile(mass, [.25, .5, .75], axis=0)
        ax[1].plot(RADII, mid, color=color, label=LABELS[name])
        ax[1].fill_between(RADII, q1, q3, color=color, alpha=.14)
    ax[0].axhline(.90, color="#777777", lw=1, ls=":")
    ax[0].set(title="How many PCA patterns explain the variance?", xlabel="PCA patterns",
              ylabel="Fraction of variance", ylim=(0, 1.02))
    ax[0].plot([], [], ls="--", color="#777777", label="38-map subsets")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set(title="Close-pair probability across radius, 100 maps", xlabel="Radius / reference RMS pair scale",
              ylabel="Fraction of pair weight", xscale="log", yscale="log", xlim=(.09, 4), ylim=(.0005, 1.1))
    ax[1].set_xticks([.1, .2, .5, 1, 2, 4], labels=["0.1", "0.2", "0.5", "1", "2", "4"])
    ax[1].minorticks_off()
    for axis in ax:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "dimensions.png", dpi=180)
    plt.close(fig)


def main():
    run = HERE / "run"
    output = HERE / "dimensions"
    output.mkdir(exist_ok=True)
    setup = json.loads((run / "run_settings.json").read_text(encoding="utf-8"))
    scores = json.loads((run / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    case_ids = set(setup["case_ids"])
    adapter = get_dataset("poisson")
    cases = [ref for ref in adapter.discover(Path(os.environ.get("POSTERIORBENCH_DATA", str(ROOT.parents[1] / "data"))))
             if ref.split == "test" and ref.stem in case_ids]
    assert {ref.stem for ref in cases} == case_ids == set(scores["per_case"])
    assert len(cases) == scores["num_cases"] == 90
    curves = {name: [] for name in LABELS}
    sub_curves = {name: [] for name in LABELS}
    full_rows, subset_rows = [], []
    for i, case in enumerate(cases, 1):
        ref = adapter.load_reference(case)
        fname = run / "predictions" / case.stem / "generated.h5"
        with h5py.File(fname) as file:
            if file.attrs["method"] != "esmda" or file.attrs["case_id"] != case.stem:
                raise ValueError(f"Mismatched prediction file: {fname}")
            solver = file["f"][:]
            weights = normalize(file["w"][:])
        if solver.shape != ref.samples.shape or solver.shape != (100, 128, 128):
            raise ValueError(f"Expected 100 matched maps for {case.stem}.")
        if not np.isfinite(solver).all():
            raise ValueError(f"Nonfinite maps in {fname}.")
        grams = {"reference": gram_matrix(ref.samples), "esmda": gram_matrix(solver)}
        weights = {"reference": normalize(ref.weights), "esmda": weights}
        scale = np.sqrt(2 * pca(grams["reference"], weights["reference"])[-1])
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 138, case.index]))
        for name, gram in grams.items():
            w = weights[name]
            full, (cum, mass, slopes, _) = analyze(gram, w, scale)
            full_rows.append(dict(case=case.stem, method=name, any_supported_window=np.isfinite(slopes).any(), **full))
            curves[name].append((cum, mass))
            subset_curves = []
            for rep in range(REPEATS):
                ids = rng.choice(len(w), N_SUBSET, replace=False)
                stats, (sub_cum, _, sub_slopes, _) = analyze(gram[np.ix_(ids, ids)], normalize(w[ids]), scale)
                subset_rows.append(dict(case=case.stem, method=name, repeat=rep,
                                        any_supported_window=np.isfinite(sub_slopes).any(), **stats))
                subset_curves.append(sub_cum)
            sub_curves[name].append(np.median(subset_curves, axis=0))
        if i % 15 == 0 or i == len(cases):
            print(f"Analyzed {i}/{len(cases)} test cases", flush=True)

    full = pd.DataFrame(full_rows)
    subsets = pd.DataFrame(subset_rows)
    full.to_csv(output / "per_case_100.csv", index=False)
    subsets.to_csv(output / "subsets_38.csv", index=False)
    summary = []
    for name, records in (("100", full), ("38", subsets)):
        for method in LABELS:
            selection = records[records.method == method]
            typical = selection.groupby("case")[["d90", "participation_ratio", "variance_per_pixel"]].median()
            summary.append(dict(
                maps=int(name), method=method, cases=len(typical), estimates=len(selection),
                median_d90=typical.d90.median(), median_pr=typical.participation_ratio.median(),
                median_variance_per_pixel=typical.variance_per_pixel.median(),
                supported_window_estimates=int(selection.any_supported_window.sum()),
                resolved_correlation_dimension=int(selection.d2.count()),
            ))
    report = pd.DataFrame(summary)
    report.to_csv(output / "summary.csv", index=False)
    comparisons = []
    for n, rows in ((100, full), (38, subsets)):
        index = ["case"] + (["repeat"] if n == 38 else [])
        pair = rows.pivot(index=index, columns="method", values=["variance_per_pixel", "participation_ratio"])
        for metric in ("variance_per_pixel", "participation_ratio"):
            ratios = pair[metric]["esmda"] / pair[metric]["reference"]
            per_case = ratios.groupby("case").median()
            comparisons.append(dict(maps=n, metric=metric, median_paired_ratio=per_case.median(),
                                    cases_esmda_greater=int((per_case > 1).sum())))
    pd.DataFrame(comparisons).to_csv(output / "paired_ratios.csv", index=False)
    plot(output, curves, sub_curves)
    print(report.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(3):
        main()
