"""Weighted PCA and finite-scale correlation slopes for Poisson posterior ensembles."""

import argparse
import os
import csv
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import idctn
from scipy.linalg import cholesky, cho_solve, solve_triangular
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path[:0] = [str(PROJECT / "vendor" / "PosteriorBench"), str(ROOT.parent / "02_sensor_placement")]
from run_ablation import Poisson, SIZE
from posteriorbench.datasets import get_dataset
from posteriorbench.evaluation import metrics

LABELS = {"reference": "Weighted reference", "esmda": "ES-MDA", "gaussian": "Gaussian conditioning"}
COLORS = {"reference": "#222936", "esmda": "#b65031", "gaussian": "#237b80"}
RADII = 2 ** np.arange(-6.0, 2.001, 0.05)
WINDOW = 20  # One octave on the grid above.
MIN_PAIRS = 50
MAX_MASS = 0.25
SLOPE_TOL = 0.15
SEED = 20261120
SUBSET_REPEATS = 8


def normalize(w):
    return metrics.normalize_weights(w)


def gram_matrix(fields):
    x = np.asarray(fields, dtype=np.float64).reshape(len(fields), -1)
    x = x - x.mean(axis=0)
    return x @ x.T / x.shape[1]


def pca(gram, w):
    """The sample-space Gram matrix has the nonzero eigenvalues of the field covariance."""
    gram_mean = gram @ w
    centered = gram - gram_mean[:, None] - gram_mean[None, :] + w @ gram_mean
    root_w = np.sqrt(w)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered * root_w[:, None] * root_w)[::-1], 0)
    variance = eigenvalues.sum()
    if variance == 0:
        return np.zeros(len(w)), 0, 0.0, 0.0
    fractions = eigenvalues / variance
    cumulative = np.cumsum(fractions)
    return cumulative, int(np.searchsorted(cumulative, 0.90) + 1), float(1 / (fractions @ fractions)), float(variance)


def correlation(gram, w, scale):
    i, j = np.triu_indices(len(w), 1)
    distances = np.sqrt(np.maximum(gram[i, i] + gram[j, j] - 2 * gram[i, j], 0))
    order = np.argsort(distances)
    pair_weights = (w[i] * w[j])[order]
    indices = np.searchsorted(distances[order], RADII * scale, side="left")
    weight = np.r_[0, np.cumsum(pair_weights)][indices]
    squared = np.r_[0, np.cumsum(pair_weights**2)][indices]
    mass = weight / pair_weights.sum()
    effective_pairs = np.divide(weight**2, squared, out=np.zeros_like(weight), where=squared > 0)
    log_mass = np.log(mass, out=np.full_like(mass, np.nan), where=mass > 0)
    slopes = np.full_like(RADII, np.nan)
    accepted = np.zeros_like(RADII, dtype=bool)
    log_radius = np.log(RADII)
    for start in range(len(RADII) - WINDOW):
        middle, end = start + WINDOW // 2, start + WINDOW
        if effective_pairs[start] < MIN_PAIRS or mass[end] > MAX_MASS:
            continue
        whole = (log_mass[end] - log_mass[start]) / (log_radius[end] - log_radius[start])
        first = (log_mass[middle] - log_mass[start]) / (log_radius[middle] - log_radius[start])
        second = (log_mass[end] - log_mass[middle]) / (log_radius[end] - log_radius[middle])
        slopes[middle] = whole
        accepted[middle] = whole > 0 and max(abs(first - whole), abs(second - whole)) <= SLOPE_TOL * whole
    # Choose the longest contiguous run of accepted window centers and its median point.
    runs = np.split(np.flatnonzero(accepted), np.flatnonzero(np.diff(np.flatnonzero(accepted)) > 1) + 1)
    run = max(runs, key=len)
    chosen = run[len(run) // 2] if len(run) >= 5 else None
    return mass, slopes, accepted, chosen


def correlation_summary(slopes, chosen, scale):
    if chosen is None:
        return dict(d2=None, window_low_rmse=None, window_high_rmse=None)
    return dict(d2=float(slopes[chosen]), window_low_rmse=float(scale * RADII[chosen - WINDOW // 2]),
                window_high_rmse=float(scale * RADII[chosen + WINDOW // 2]))


def analyze(gram, w, scale):
    cumulative, d90, pr, variance = pca(gram, w)
    mass, slopes, accepted, chosen = correlation(gram, w, scale)
    row = dict(sample_count=len(w), weight_ess=float(1 / (w @ w)), d90=d90, participation_ratio=pr,
               variance_per_pixel=variance, **correlation_summary(slopes, chosen, scale))
    return row, (cumulative, mass, slopes, accepted)


def dct_std(alpha, tau):
    k = np.arange(SIZE, dtype=float)**2
    frequency = np.pi**2 * (k[:, None] + k[None, :])
    alpha, tau = (np.asarray(par)[..., None, None] for par in (alpha, tau))
    std = SIZE * tau**(alpha - 1) * (frequency + tau**2)**(-alpha / 2)
    std[..., 0, 0] = 0.0
    return std


def mixture_std():
    nodes, weights = np.polynomial.legendre.leggauss(12)
    variance = dct_std(2.25 + .75 * nodes[:, None], 3 + nodes[None, :])**2
    return np.einsum("i,j,ijxy->xy", weights / 2, weights / 2, variance)**0.5


def dct_fields(coefficients):
    return idctn(coefficients.reshape(-1, SIZE, SIZE), norm="ortho", axes=(-2, -1))


def prepare_esmda(unscaled, observed, noise_variance, count, rng, *, observation_error=None):
    """Return the prior in DCT space and the linear update factors for each draw."""
    m = len(observed)
    prior_count = max(512, count)
    mixture = dct_std(rng.uniform(1.5, 3, prior_count), rng.uniform(2, 4, prior_count))
    source_coeff = (rng.standard_normal(mixture.shape) * mixture).reshape(prior_count, -1)
    response = source_coeff @ unscaled.T
    anom = response - response.mean(axis=0)
    empirical_cross = anom.T @ (source_coeff - source_coeff.mean(axis=0)) / (prior_count - 1)
    empirical_cov = anom.T @ anom / (prior_count - 1)
    noise_std = np.sqrt(noise_variance)
    jitter = 0.0
    if observation_error is not None:
        if observation_error["mode"] != "prior_std_fraction":
            raise ValueError("Expected per-sensor prior standard deviation scaling.")
        noise_std = np.maximum(observation_error["scale"] * response.std(axis=0, ddof=1),
                               observation_error["min_std"])
        jitter = observation_error["jitter"]
    innovation = observed + rng.normal(0, noise_std, (count, m)) - response[:count]
    system = empirical_cov + np.diag(np.broadcast_to(noise_std**2, (m,)))
    system[np.diag_indices_from(system)] += jitter * max(float(np.diag(system).mean()), 1.0)
    increments = np.linalg.solve(system, innovation.T).T
    updated_measurements = response[:count] + increments @ empirical_cov
    return source_coeff[:count], increments, empirical_cross, updated_measurements, noise_std


def baselines(model, coordinates, observed, weights_sigma2, count, rng, prior_std):
    # PosteriorBench's weight uses MSE, so this likelihood has variance m*sigma^2 per pixel.
    m = len(coordinates)
    noise_variance = m * weights_sigma2
    noise_std = np.sqrt(noise_variance)
    design = model.sensor_design(coordinates, prior_std)
    factor = cholesky(design @ design.T + noise_variance * np.eye(m), lower=True)
    flat_std = prior_std.ravel()
    cross = design * flat_std
    white = rng.standard_normal((count, SIZE**2))
    innovation = observed + rng.normal(0, noise_std, (count, m)) - white @ design.T
    gaussian_coeff = white * flat_std + cho_solve((factor, True), innovation.T).T @ cross

    # Analytic Gaussian population PR from traces of D - B.T @ B.
    b = solve_triangular(factor, cross, lower=True)
    removed_diag = np.sum(b**2, axis=0)
    diag = flat_std**2
    trace = np.sum(diag - removed_diag)
    trace_sq = diag @ diag - 2 * (diag @ removed_diag) + np.sum((b @ b.T)**2)
    oracle = dict(gaussian_population_pr=float(trace**2 / trace_sq),
                  gaussian_population_variance=float(trace / SIZE**2))

    # The unscaled sensor operator has no DC contribution for the zero-mean prior.
    unscaled = np.divide(design, flat_std, out=np.zeros_like(design), where=flat_std > 0)
    source_coeff, updates, empirical_cross, _, _ = prepare_esmda(
        unscaled, observed, noise_variance, count, rng
    )
    esmda_coeff = source_coeff + updates @ empirical_cross

    generated = {"esmda": dct_fields(esmda_coeff), "gaussian": dct_fields(gaussian_coeff)}
    residuals = {"esmda": esmda_coeff @ unscaled.T - observed,
                 "gaussian": gaussian_coeff @ unscaled.T - observed}
    return generated, residuals, oracle


def scores(pred, ref, ref_w, projections, field_scale):
    equal = np.full(len(pred), 1.0 / len(pred))
    mean_ref = np.average(ref, axis=0, weights=ref_w)
    std_ref = np.sqrt(np.average((ref - mean_ref)**2, axis=0, weights=ref_w))
    mean = pred.mean(axis=0)
    proj = pred.reshape(len(pred), -1) @ projections / field_scale
    proj_ref = ref.reshape(len(ref), -1) @ projections / field_scale
    return dict(
        mean_rel_l2=float(np.linalg.norm(mean - mean_ref) / (np.linalg.norm(mean_ref) + 1e-12)),
        std_rel_l2=float(np.linalg.norm(pred.std(axis=0) - std_ref) / (np.linalg.norm(std_ref) + 1e-12)),
        mmd=metrics.weighted_mmd(pred, ref, equal, ref_w),
        swd=float(np.mean([metrics.weighted_wasserstein_1d(x, y, equal, ref_w)
                           for x, y in zip(proj.T, proj_ref.T)])),
    )


def controls(seed):
    rng = np.random.default_rng(seed)
    embedding = np.linalg.qr(rng.standard_normal((1024, 8)))[0].T
    results = []
    for n in (100, 400):
        for trial in range(24):
            angle = rng.uniform(0, 2*np.pi, n)
            sphere = rng.normal(size=(n, 3))
            sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
            examples = {
                "line": (rng.uniform(-1, 1, (n, 1)), 1, 1),
                "circle": (np.column_stack((np.sin(angle), np.cos(angle))), 2, 1),
                "sphere_surface": (sphere, 3, 2),
                "gaussian_8": (rng.normal(size=(n, 8)), 8, 8),
            }
            for name, (latent, expected_pr, expected_d2) in examples.items():
                g = gram_matrix(latent @ embedding[:latent.shape[1]])
                for weighting, w in (("equal", np.ones(n)), ("uneven", rng.lognormal(0, .9, n))):
                    w = normalize(w)
                    _, _, _, v = pca(g, w)
                    scale = np.sqrt(2 * v)
                    row, _ = analyze(g, w, scale)
                    results.append(dict(control=name, trial=trial, weighting=weighting,
                                        population_pr=expected_pr, population_d2=expected_d2, **row))
                    if name == "circle" and trial == 0:
                        contracted, _ = analyze(g * .25**2, w, scale * .25)
                        np.testing.assert_allclose(contracted["variance_per_pixel"], v * .25**2)
                        np.testing.assert_allclose(contracted["participation_ratio"], row["participation_ratio"])
                        assert contracted["d90"] == row["d90"]
                        assert contracted["d2"] == row["d2"]
    return results


def save_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output, rows, subsets, curves, control_rows, pca_limit=60):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for name, series in curves.items():
        cumulative, mass, slopes, accepted = map(np.stack, zip(*series))
        for values, x, axis in ((cumulative[:, :pca_limit], np.arange(1, pca_limit + 1), ax[0]), (mass, RADII, ax[1])):
            lo, med, hi = np.quantile(values, [.25, .5, .75], axis=0)
            axis.plot(x, med, color=COLORS[name], label=LABELS[name])
            axis.fill_between(x, lo, hi, color=COLORS[name], alpha=.15)
        supported = np.isfinite(slopes)
        ax[2].plot(RADII, 100 * supported.mean(axis=0), color=COLORS[name], linestyle="--")
        ax[2].plot(RADII, 100 * accepted.mean(axis=0), color=COLORS[name])
    ax[0].set(xlabel="Number of PCA modes", ylabel="Fraction of variance", title="Linear variance spectrum", ylim=(0, 1.02))
    ax[0].axhline(.90, color="#777777", linestyle=":", linewidth=1)
    ax[0].legend(frameon=False)
    for axis in ax[1:]:
        axis.set_xscale("log")
        axis.set_xlim(.09, 4)
        axis.set_xticks([.1, .2, .5, 1, 2, 4], labels=["0.1", "0.2", "0.5", "1", "2", "4"])
        axis.minorticks_off()
        axis.set_xlabel("Radius / reference RMS pair scale")
    ax[1].set(ylabel="Weighted fraction of close pairs", title="Growth of pair probability")
    ax[1].set_yscale("log")
    ax[1].set_ylim(.0005, 1.08)
    ax[2].set(ylabel="Percentage of cases", title="Scale coverage and slope stability", ylim=(-2, 104))
    ax[2].plot([], [], color="#646464", linestyle="--", label="Supported one-octave window")
    ax[2].plot([], [], color="#646464", label="Stable within an octave")
    ax[2].legend(frameon=False, fontsize=8)
    if not any(np.isfinite(series[2]).any() for group in curves.values() for series in group):
        ax[2].text(.5, .48, "No ensemble supported\na full octave", ha="center", va="center",
                   transform=ax[2].transAxes, color="#646464")
    fig.savefig(output / "dimension_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.9), layout="constrained")
    for name in LABELS:
        subset_rows = [row for row in subsets if row["method"] == name]
        full = {row["case"]: row for row in rows if row["method"] == name}
        values, rates = [], []
        for n in (50, 75, 100):
            q = [row for row in subset_rows if row["sample_count"] == n] if n < 100 else list(full.values())
            values.append(np.quantile([row["participation_ratio"] / full[row["case"]]["participation_ratio"]
                                       for row in q], [.25, .5, .75]))
            rates.append(np.mean([row["d2"] is not None for row in q]) * 100)
        lo, med, hi = np.asarray(values).T
        ax[0].plot([50, 75, 100], med, "o-", color=COLORS[name], label=LABELS[name])
        ax[0].fill_between([50, 75, 100], lo, hi, color=COLORS[name], alpha=.15)
        ax[1].plot([50, 75, 100], rates, "o-", color=COLORS[name])
    ax[0].set(ylabel="Participation ratio / full-ensemble value", title="Linear estimate sensitivity")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set(ylabel="Percentage with a stable window", title="Nonlinear estimate availability", ylim=(-2, 104))
    for axis in ax:
        axis.set(xlabel="Ensemble size", xticks=[50, 75, 100])
    fig.savefig(output / "sample_size_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    names = ("line", "circle", "sphere_surface", "gaussian_8")
    for k, name in enumerate(names):
        selected = [r for r in control_rows if r["control"] == name and r["sample_count"] == 100]
        for offset, weighting, color in ((-.12, "equal", "#237b80"), (.12, "uneven", "#b65031")):
            draws = [r for r in selected if r["weighting"] == weighting]
            found = [r["d2"] for r in draws if r["d2"] is not None]
            if found:
                ax.scatter(np.full(len(found), k + offset), found, s=12, color=color, alpha=.45)
            ax.text(k + offset, .26 if offset < 0 else .62, f"{len(found)}/24", color=color, ha="center", fontsize=8)
        ax.scatter(k, selected[0]["population_d2"], s=65, marker="_", color="#222936")
    ax.scatter([], [], color="#237b80", label="Equal weights")
    ax.scatter([], [], color="#b65031", label="Uneven weights")
    ax.scatter([], [], color="#222936", marker="_", label="Known dimension")
    ax.set(xticks=range(4), xticklabels=["Line", "Circle", "Sphere surface", "8-D Gaussian"],
           ylabel="Estimated slope", ylim=(0, 8.5), title="Nonlinear controls with 100 points")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.savefig(output / "synthetic_controls.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = Path(os.environ.get("POSTERIORBENCH_DATA", str(PROJECT / "data")))
    metadata = json.loads((data / "PosteriorDataset_hf/Poisson/metadata.json").read_text())
    training = json.loads((data / "PDEFieldDataset_hf/Poisson/metadata.json").read_text())
    sigma2 = metadata["posterior_weighting"]["parameters"]["sigma_squared"]
    max_mse = metadata["selection"]["parameters"]["retained_sample_mse_upper_bound"]
    field_scale = training["stats"]["std"][0]
    adapter = get_dataset("poisson")
    case_refs = [ref for ref in adapter.discover(data) if ref.split == args.split][:args.max_cases]
    if not case_refs:
        parser.error("No cases selected.")
    model, prior_std = Poisson(), mixture_std()
    projection_rng = np.random.default_rng(metrics.DEFAULT_PROJECTION_SEED)
    projections = np.column_stack([metrics.sample_grf_projection(SIZE, SIZE, projection_rng).ravel()
                                   for _ in range(metrics.DEFAULT_N_PROJECTIONS)])
    control_rows = controls(SEED)
    save_csv(args.output / "synthetic_controls.csv", control_rows)
    print("Synthetic controls complete", flush=True)
    rows, subsets, oracle_rows, array_rows = [], [], [], []
    curves = {name: [] for name in LABELS}
    verification = {}
    for index, case_ref in enumerate(case_refs):
        case = adapter.load_case(case_ref)
        reference = adapter.load_reference(case_ref)
        ref = reference.samples.astype(np.float64)
        w = normalize(reference.weights)
        coords = case.sensor_coords
        observed = case.fields["phi"][coords[:, 0], coords[:, 1]].astype(np.float64)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, int(args.split == "test"), case_ref.index]))
        generated, residuals, oracle = baselines(model, coords, observed, sigma2, len(ref), rng, prior_std)
        key = f"{case_ref.split}:{case.case_id}"
        oracle_rows.append(dict(case=key, **oracle))
        if index == 0:
            actual = model.forward(case.fields["f"].astype(np.float64))
            verification["forward_max_abs_error"] = float(np.max(np.abs(actual - case.fields["phi"])))
            np.testing.assert_allclose(actual, case.fields["phi"], atol=2e-7, rtol=2e-5)
            actual_residual = model.forward(generated["esmda"][:3])[:, coords[:, 0], coords[:, 1]] - observed
            np.testing.assert_allclose(actual_residual, residuals["esmda"][:3], atol=1e-11, rtol=0)
            official = metrics.evaluate_distribution(
                generated["esmda"], ref, np.full(len(ref), 1 / len(ref)), w,
                swd_normalization_mean=training["stats"]["mean"][0], swd_normalization_std=field_scale
            )
            fast = scores(generated["esmda"], ref, w, projections, field_scale)
            for name, value in fast.items():
                np.testing.assert_allclose(value, official[name], rtol=2e-10, atol=1e-12)
            verification["metrics_max_abs_error"] = max(abs(official[k] - v) for k, v in fast.items())
        ref_gram = gram_matrix(ref)
        ref_variance = pca(ref_gram, w)[-1]
        radius_scale = np.sqrt(2 * ref_variance)
        for name, fields in {"reference": ref, **generated}.items():
            gram = ref_gram if name == "reference" else gram_matrix(fields)
            weights = w if name == "reference" else np.full(len(fields), 1 / len(fields))
            row, arrays = analyze(gram, weights, radius_scale)
            curves[name].append(arrays)
            array_rows.append((key, name, radius_scale, *arrays))
            existing = {} if name == "reference" else scores(fields, ref, w, projections, field_scale)
            if name != "reference":
                existing["fraction_inside_reference_gate"] = float(np.mean(np.mean(residuals[name]**2, axis=1) <= max_mse))
            rows.append(dict(case=key, method=name, **row, **existing))
            for n in (50, 75):
                for repeat in range(SUBSET_REPEATS):
                    idx = rng.choice(len(fields), size=n, replace=False)
                    subset_row, _ = analyze(gram[np.ix_(idx, idx)], normalize(weights[idx]), radius_scale)
                    subsets.append(dict(case=key, method=name, repeat=repeat, **subset_row))
        if (index + 1) % 5 == 0 or index + 1 == len(case_refs):
            print(f"{args.split}: {index + 1}/{len(case_refs)} cases", flush=True)

    save_csv(args.output / "per_case.csv", rows)
    save_csv(args.output / "subsamples.csv", subsets)
    save_csv(args.output / "gaussian_population.csv", oracle_rows)
    cases, methods, scales, cumulative, mass, slopes, accepted = zip(*array_rows)
    np.savez_compressed(args.output / "curves.npz", case=cases, method=methods, radius_scale=scales,
                        radii=RADII, cumulative=cumulative, mass=mass, slopes=slopes, accepted=accepted)
    plot_results(args.output, rows, subsets, curves, control_rows)
    pinned = json.loads((ROOT.parent / "pinned_inputs.json").read_text())
    provenance = dict(
        benchmark_commit=subprocess.check_output(["git", "-C", str(ROOT.parent / "PosteriorBench"),
                                                  "rev-parse", "HEAD"], text=True).strip(),
        dataset_revision=pinned["dataset_revision"], split=args.split, case_count=len(case_refs), seed=SEED,
        synthetic_trials=24, subsamples_per_size=SUBSET_REPEATS, likelihood_mse_sigma_squared=sigma2,
        reference_threshold_mse=max_mse, min_effective_pairs=MIN_PAIRS, max_pair_mass=MAX_MASS,
        half_window_slope_tolerance=SLOPE_TOL, min_contiguous_window_centers=5,
        packages={name: version(name) for name in ("numpy", "scipy", "matplotlib", "datasets", "threadpoolctl")},
        verification=verification,
    )
    (args.output / "settings.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=3, user_api="blas"):
        main()
