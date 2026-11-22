"""Render the Darcy pressure and local conservation results."""

import csv
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
import numpy as np
from threadpoolctl import threadpool_limits

from analyze_pressures import DATA, HERE, PATCH_SIDES, get_dataset, load_case
from darcy_grid import FORCING, divergence, face_coefficients, recover_pressure

COLORS = {"internal": "#923c28", "numerical": "#16707c", "reference": "#515e79"}
LABELS = {"internal": "ES-MDA internal", "numerical": "Darcy re-solve", "reference": "Stored reference"}


def read_csv(name):
    with (HERE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summary_figure(summary):
    pressure = read_csv("pressure_by_case.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), layout="constrained")
    fig.suptitle("Darcy pressure from the same ES-MDA permeability ensemble", fontsize=16, fontweight="bold")
    ax = axes[0]
    per_case = {
        name: {row["case_id"]: 1e3 * float(row["reference_w1"]) for row in pressure
               if row["ensemble"] == name and row["region"] == "held_out"}
        for name in ("internal", "numerical")
    }
    x = np.array(list(per_case["internal"].values()))
    y = np.array([per_case["numerical"][key] for key in per_case["internal"]])
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    margin = 0.08 * (high - low)
    limits = (low - margin, high + margin)
    ax.plot(limits, limits, linestyle="--", color="#686d74", lw=1)
    ax.scatter(x, y, s=18, color=COLORS["numerical"], alpha=0.65, edgecolor="none", zorder=2)
    ax.set(xlim=limits, ylim=limits, aspect="equal", xlabel="Internal pressure",
           ylabel="Numerically solved pressure", title="Distance from reference at held-out points")
    ax.text(0.96, 0.07, "Wasserstein-1  ·  ×10⁻³ pressure units\nOne dot per test case", va="bottom", ha="right",
            transform=ax.transAxes, fontsize=9)

    ax = axes[1]
    series = ("internal", "numerical", "reference")
    width = 0.24
    for j, name in enumerate(series):
        held_out = summary["pressure"][name]["held_out"]
        ax.bar(np.arange(2) + (j - 1) * width,
               1e3 * np.array([held_out["truth_mean_rmse"], held_out["truth_crps"]]),
               width=width, label=LABELS[name], color=COLORS[name], zorder=3)
    ax.set(xticks=[0, 1], xticklabels=["RMSE of mean", "CRPS"], ylabel="×10⁻³ pressure units",
           title="Agreement with hidden truth")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax = axes[2]
    curves = [
        ("internal_binary", COLORS["internal"], "o", "-", "Internal + rounded permeability"),
        ("internal_raw_clipped", "#a47321", "^", "--", "Internal + raw permeability"),
        ("reference", COLORS["reference"], "x", ":", "Stored reference"),
        ("numerical_binary", COLORS["numerical"], "o", "--", "Darcy re-solve"),
    ]
    for name, color, marker, linestyle, label in curves:
        trace = [summary["conservation"][name][str(side)] for side in PATCH_SIDES]
        ax.plot(range(len(PATCH_SIDES)), trace, marker=marker, markerfacecolor="white", color=color,
                linestyle=linestyle, linewidth=1.6, ms=5.4, label=label)
    ax.set(yscale="log", xticks=range(len(PATCH_SIDES)), xticklabels=PATCH_SIDES, title="Local flow conservation",
           xlabel="Patch side (grid cells)", ylabel="RMS imbalance / source in patch")
    ax.legend(loc="center right", frameon=True, fontsize=8.5)
    for ax in axes:
        ax.grid(axis="y", color="#d9dfe3", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "pressure_and_conservation.png", dpi=190)
    plt.close(fig)


def example_figure():
    paired = read_csv("paired_pressure_by_case.csv")
    means = np.array([float(row["mean_paired_rmse_all"]) for row in paired])
    case_id = paired[np.argmin(np.abs(means - np.median(means)))]["case_id"]
    dataset = get_dataset("darcy")
    ref = next(ref for ref in dataset.discover(DATA) if ref.stem == case_id)
    case = load_case(dataset, ref)

    fields = case["fields"]
    internal, numerical = fields["internal"], fields["numerical"]
    mean_ref = np.tensordot(case["weights"], fields["reference"], (0, 0))
    differences = numerical.astype(np.float64) - internal
    rms = np.sqrt(np.mean(differences**2, axis=(1, 2)))
    member = int(np.argmin(np.abs(rms - np.median(rms))))
    faces = face_coefficients(case["a"][member])
    residuals = [
        np.abs(divergence(recover_pressure(u), faces) - FORCING)
        for u in (internal[member], numerical[member])
    ]

    fig, ax = plt.subplots(2, 4, figsize=(12.8, 6.4), layout="constrained")
    fig.suptitle(f"{case_id} · test case near the median paired pressure gap", fontsize=15, fontweight="bold")
    panels = [
        ("Hidden truth", case["truth"]), ("Weighted reference mean", mean_ref),
        ("Internal pressure mean", internal.mean(axis=0)), ("Darcy pressure mean", numerical.mean(axis=0)),
    ]
    vmax = max(y.max() for _, y in panels)
    for panel, (label, y) in zip(ax[0], panels):
        pressure_image = panel.imshow(y * 1e3, cmap="cividis", origin="lower", vmin=0, vmax=vmax * 1e3)
        panel.set_title(label, fontsize=10)
    ys, xs = np.where(case["observed"])
    ax[0, 0].scatter(xs, ys, s=2, color="white", edgecolor="none", alpha=0.48)
    colorbar = fig.colorbar(pressure_image, ax=ax[0].tolist(), shrink=0.72, pad=0.012)
    colorbar.set_label("×10⁻³ pressure units", fontsize=9)

    material_cmap = ListedColormap(["#c6dce8", "#285372"])
    material = ax[1, 0].imshow(case["a"][member], cmap=material_cmap,
                               norm=BoundaryNorm([2, 7.5, 13], material_cmap.N), origin="lower")
    ax[1, 0].set_title(f"Permeability · draw {member + 1}", fontsize=10)
    fig.colorbar(material, ax=ax[1, 0], ticks=[3, 12], shrink=0.68)
    delta = differences[member] * 1e3
    max_delta = np.max(np.abs(delta))
    delta_image = ax[1, 1].imshow(delta, cmap="RdBu_r", vmin=-max_delta, vmax=max_delta, origin="lower")
    ax[1, 1].set_title("Darcy minus internal pressure", fontsize=10)
    colorbar = fig.colorbar(delta_image, ax=ax[1, 1], shrink=0.68)
    colorbar.set_label("×10⁻³ pressure units", fontsize=8)

    norm = LogNorm(vmin=1e-5, vmax=max(1.0, np.max(residuals[0])))
    for panel, label, r in zip(ax[1, 2:], ("Internal flow imbalance", "Darcy flow imbalance"), residuals):
        residual_image = panel.imshow(np.maximum(r, 1e-5), cmap="magma", norm=norm, origin="lower")
        panel.set_title(label, fontsize=10)
    colorbar = fig.colorbar(residual_image, ax=ax[1, 2:].tolist(), shrink=0.68)
    colorbar.set_label("Fractional imbalance", fontsize=8)
    for panel in ax.flat:
        panel.set_xticks([])
        panel.set_yticks([])
    fig.savefig(HERE / "representative_case.png", dpi=185)
    plt.close(fig)
    print(f"Rendered example for {case_id}, draw {member + 1}.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        summary_figure(json.loads((HERE / "summary.json").read_text(encoding="utf-8")))
        example_figure()
