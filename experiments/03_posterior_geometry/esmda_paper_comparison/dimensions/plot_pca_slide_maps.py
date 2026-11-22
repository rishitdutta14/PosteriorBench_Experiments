"""Draw posterior moments and PCA modes for the Poisson case in the deck."""

import csv
import os
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, PowerNorm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
sys.path.insert(0, str(PROJECT / "vendor" / "PosteriorBench"))
from posteriorbench.datasets import get_dataset
from posteriorbench.evaluation.metrics import normalize_weights

CASE = "poisson_12"
GROUPS = ("Weighted reference", "ES-MDA")
COLORS = ("#243444", "#b65031")
PANEL_LEFT = (0.028, 0.163, 0.298, 0.454, 0.589, 0.724)


def weighted_modes(maps, weights):
    maps = maps.astype(np.float64)
    mean = np.average(maps, axis=0, weights=weights)
    centered = (maps - mean).reshape(len(maps), -1)
    weighted = centered * np.sqrt(weights[:, None])
    variance = np.einsum('ij,ij->j', weighted, weighted).reshape(mean.shape)
    values, vectors = np.linalg.eigh(weighted @ weighted.T)
    values, vectors = values[::-1], vectors[:, ::-1]
    components = (vectors[:, :3].T @ weighted) / np.sqrt(values[:3, None])
    np.testing.assert_allclose(components @ components.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(variance.sum(), values.sum(), rtol=1e-12)
    shares = values / values.sum()
    for mode in components:
        peak = np.argmax(np.abs(mode))
        mode /= mode[peak]
    return mean, variance, components.reshape(3, *mean.shape), shares


def figure(left=PANEL_LEFT, width=.125, bottom=.25, group_centers=(.2255, .6515), group_top=.89):
    fig = plt.figure(figsize=(12, 3.35), facecolor="white")
    for center, label, color in zip(group_centers, GROUPS, COLORS):
        fig.text(center, group_top, label, color=color, ha="center", weight="bold", fontsize=17)
    axes = [fig.add_axes((x, bottom, width, width * 12 / 3.35)) for x in left]
    for axis in axes:
        axis.axis("off")
    return fig, axes


def colorbar(fig, norm, x, title, ticks, cmap="RdBu_r"):
    axis = fig.add_axes((x, .25, .010, .448))
    scale = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axis, ticks=ticks)
    axis.set_title(title, fontsize=8, pad=8)
    scale.ax.tick_params(labelsize=8, length=2)


def main():
    adapter = get_dataset("poisson")
    case = next(ref for ref in adapter.discover(Path(os.environ.get("POSTERIORBENCH_DATA", str(PROJECT / "data")))) if ref.stem == CASE and ref.split == "test")
    reference = adapter.load_reference(case)
    with h5py.File(HERE.parent / "run" / "predictions" / CASE / "generated.h5") as handle:
        assert handle.attrs["case_id"] == CASE and handle.attrs["method"] == "esmda"
        predicted, pw = handle["f"][:], normalize_weights(handle["w"][:])
    assert predicted.shape == reference.samples.shape == (100, 128, 128)
    summaries = [weighted_modes(fields, weights) for fields, weights in (
        (reference.samples, normalize_weights(reference.weights)), (predicted, pw),
    )]
    with (HERE / "per_case_100.csv").open(newline="", encoding="utf-8") as stream:
        reported = {row["method"]: row for row in csv.DictReader(stream) if row["case"] == CASE}
    for name, (_, variance, _, shares) in zip(("reference", "esmda"), summaries):
        assert np.searchsorted(np.cumsum(shares), .90) + 1 == int(reported[name]["d90"])
        print(name, "top component variance fractions", np.round(shares[:3], 5))
        print(name, "mean pixel variance", float(variance.mean()))

    limits = max(abs(mean).max() for mean, _, _, _ in summaries)
    bound = np.ceil(limits * 2) / 2
    mean_scale = Normalize(-bound, bound)
    variance_bound = np.ceil(max(var.max() for _, var, _, _ in summaries) * 20) / 20
    variance_scale = PowerNorm(gamma=.5, vmin=0, vmax=variance_bound)
    fig, axes = figure(left=(.047, .229, .453, .635), width=.164, bottom=.18,
                       group_centers=(.22, .626), group_top=.90)
    for i, (mean, variance, _, _) in enumerate(summaries):
        group = axes[i * 2:(i + 1) * 2]
        group[0].imshow(mean, cmap="RdBu_r", norm=mean_scale, origin="upper", interpolation="bilinear")
        group[1].imshow(variance, cmap="viridis", norm=variance_scale, origin="upper", interpolation="bilinear")
        for axis, label in zip(group, ("Mean source", "Posterior variance")):
            axis.set_title(label, fontsize=10, pad=6)
    colorbar(fig, mean_scale, .879, "Mean\nsource", [-bound, 0, bound])
    variance_ticks = [0, variance_bound / 25, variance_bound / 5, variance_bound]
    colorbar(fig, variance_scale, .948, "Posterior\nvariance", variance_ticks, cmap="viridis")
    fig.text(.5, .075, "Poisson case 12. Variance is the mean squared deviation from each ensemble’s mean.", ha="center", color="#485665", fontsize=10)
    fig.savefig(HERE / "pca_slide_mean_and_variance.png", dpi=180)
    plt.close(fig)

    fig, axes = figure()
    for i, (_, _, components, shares) in enumerate(summaries):
        for k, (axis, component) in enumerate(zip(axes[i * 3:(i + 1) * 3], components)):
            axis.imshow(component, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="bilinear")
            axis.set_title(f"PC {k + 1}   {shares[k]:.1%}", fontsize=10, pad=6)
    colorbar(fig, Normalize(-1, 1), .911, "Relative\nloading", [-1, 0, 1])
    fig.text(.5, .075, "Poisson test case 12. Labels give variance explained. Colors show normalized pattern shape.", ha="center", color="#485665", fontsize=10)
    fig.savefig(HERE / "pca_slide_leading_components.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=3, user_api="blas"):
        main()
