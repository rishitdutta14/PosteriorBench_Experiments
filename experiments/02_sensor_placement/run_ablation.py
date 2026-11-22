"""Paired sensor-layout pilot for the PosteriorBench Poisson inverse problem."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dctn, dstn, idctn, idstn
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1] / "vendor" / "PosteriorBench"
sys.path.insert(0, str(REPO))
from posteriorbench.evaluation import metrics

SIZE = 128
SENSORS = 128
ALPHA = 2.25
TAU = 3.0
NOISE = 1.33e-4
LAYOUTS = ("random", "even", "diagonal", "split_diagonal", "horizontal", "vertical")
TITLE = {
    "random": "Random", "even": "Evenly spread", "diagonal": "Main diagonal",
    "split_diagonal": "Split diagonal", "horizontal": "Horizontal strip",
    "vertical": "Vertical strip",
}


def rotations(coords):
    """Rotate image coordinates by 90 degrees counterclockwise."""
    return np.column_stack((SIZE - 1 - coords[:, 1], coords[:, 0]))


def make_layouts(rng):
    r, c = np.indices((SIZE, SIZE))
    even_rows, even_cols = np.meshgrid(np.arange(8), np.arange(16), indexing="ij")
    even = np.column_stack((even_rows.ravel() * 16, even_cols.ravel() * 8))
    even += rng.integers([4, 2], [12, 6], size=even.shape)
    if rng.integers(2):
        even = even[:, ::-1]
    locations = np.arange(SIZE)
    offset = rng.integers(5, 10)
    diagonal_sites = np.rint(np.linspace(3, SIZE - offset - 4, 64)).astype(int)
    split = np.column_stack((diagonal_sites, diagonal_sites + offset))
    strip_rows = np.array([-6, -2, 2, 6]) + rng.integers(40, 88)
    strip_cols = np.linspace(2, SIZE - 3, 32).round().astype(int)
    horizontal = np.column_stack((np.repeat(strip_rows, 32), np.tile(strip_cols, 4)))
    random = rng.choice(SIZE**2, SENSORS, replace=False)
    result = dict(
        random=np.column_stack((r.ravel()[random], c.ravel()[random])),
        even=even,
        diagonal=np.column_stack((locations, locations)),
        split_diagonal=np.concatenate((split, split[:, ::-1])),
        horizontal=horizontal,
        vertical=rotations(horizontal),
    )
    for name, coords in result.items():
        if len(np.unique(coords, axis=0)) != SENSORS or np.any((coords < 0) | (coords >= SIZE)):
            raise ValueError(f"Invalid sensor layout: {name}")
    return result


def make_heldout(layout_groups, rng, count=256):
    observed = {tuple(p) for layouts in layout_groups for points in layouts.values() for p in points}
    selected = set()
    for coord in rng.permutation(SIZE**2):
        point = np.array([divmod(int(coord), SIZE)])
        orbit = set()
        for _ in range(4):
            orbit.add(tuple(point[0]))
            point = rotations(point)
        if len(orbit) == 4 and not (orbit & (observed | selected)):
            selected.update(orbit)
        if len(selected) == count:
            return np.array(sorted(selected))
    raise ValueError("Could not find enough common held-out pixels")


class Poisson:
    """Released Poisson discretization, solved directly by DST instead of CG."""

    def __init__(self):
        nodes = np.linspace(0, 1, SIZE)
        centers = (np.arange(SIZE) + 0.5) / SIZE
        eye = np.eye(SIZE)
        to_node = CubicSpline(centers, eye, axis=0)(nodes[1:-1])
        self.to_node = to_node
        self.to_center = CubicSpline(nodes, eye, axis=0)(centers)[:, 1:-1]
        frequencies = np.arange(1, SIZE - 1)
        eigenvalues = 4 * (SIZE - 1)**2 * np.sin(np.pi * frequencies / (2 * (SIZE - 1)))**2
        self.eigenvalues = eigenvalues[:, None] + eigenvalues[None, :]

    def inverse_laplacian(self, values):
        return idstn(dstn(values, type=1, norm="ortho", axes=(-2, -1)) / self.eigenvalues,
                     type=1, norm="ortho", axes=(-2, -1))

    def forward(self, source):
        potential_nodes = self.inverse_laplacian(self.to_node @ source @ self.to_node.T)
        return self.to_center @ potential_nodes @ self.to_center.T

    def sensor_design(self, coords, coefficient_std):
        # Each row maps white DCT coefficients to the potential at one sensor.
        result = []
        for chunk in np.array_split(coords, max(1, (len(coords) + 31) // 32)):
            row, col = chunk.T
            node_weights = self.to_center[row, :, None] * self.to_center[col, None, :]
            source_weights = self.to_node.T @ self.inverse_laplacian(node_weights) @ self.to_node
            result.append((dctn(source_weights, norm="ortho", axes=(-2, -1))
                           * coefficient_std).reshape(len(chunk), -1))
        return np.concatenate(result)


def at_points(maps, coordinates):
    return maps[..., coordinates[:, 0], coordinates[:, 1]]


def rms(array):
    return float(np.sqrt(np.mean(array**2)))


def relative_l2(value, reference):
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-15))


def anti_mode(fields):
    coefficients = dctn(fields, norm="ortho", axes=(-2, -1))
    return (coefficients[..., 1, 0] - coefficients[..., 0, 1]) / np.sqrt(2)


def prior_std():
    k = np.arange(SIZE, dtype=float)**2
    std = SIZE * TAU**(ALPHA - 1) * (np.pi**2 * (k[:, None] + k[None, :]) + TAU**2)**(-ALPHA / 2)
    std[0, 0] = 0.0
    return std


class Conditioner:
    """One linear Gaussian update, with exact or empirical prior cross-covariance."""

    def __init__(self, sensor_covariance, source_cross, response_cross, observation_std=NOISE,
                 jitter=0.):
        self.observation_std = np.broadcast_to(observation_std, (len(sensor_covariance),))
        system = sensor_covariance + np.diag(self.observation_std**2)
        system[np.diag_indices_from(system)] += jitter * max(np.diag(system).mean(), 1.)
        self.factor = cho_factor(system, lower=True)
        self.source_cross = source_cross
        self.response_cross = response_cross

    def update(self, sources, responses, innovations):
        weights = cho_solve(self.factor, innovations.T).T
        source_change = (weights @ self.source_cross).reshape(-1, SIZE, SIZE)
        return sources + source_change, responses + weights @ self.response_cross

    def posterior_std(self, prior_var, cross_covariance):
        whitened = solve_triangular(self.factor[0], cross_covariance, lower=True)
        return np.sqrt(np.maximum(prior_var - np.sum(whitened**2, axis=0), 0))


def empirical_conditioner(prior_fields, prior_potentials, coords, heldout, observation_error=None):
    observations = at_points(prior_potentials, coords)
    predictions = at_points(prior_potentials, heldout)
    obs_anom = observations - observations.mean(axis=0)
    fields_flat = prior_fields.reshape(len(prior_fields), -1)
    scale = len(prior_fields) - 1
    covariance = obs_anom.T @ obs_anom / scale
    source_cross = obs_anom.T @ (fields_flat - fields_flat.mean(axis=0)) / scale
    response_cross = obs_anom.T @ (predictions - predictions.mean(axis=0)) / scale
    observation_std, jitter = NOISE, 0.
    if observation_error is not None:
        observation_std = np.maximum(float(observation_error["min_std"]),
                                     float(observation_error["scale"]) * np.sqrt(np.diag(covariance)))
        jitter = float(observation_error["jitter"])
    return Conditioner(covariance, source_cross, response_cross, observation_std, jitter)


def load_esmda_profile():
    from posteriorbench.artifacts import load_checkpoint_profile

    profile = load_checkpoint_profile(REPO / "configs/evaluations/esmda_poisson.yaml")
    observation_error = profile.guidance["observation_error"]
    if (profile.sampling["num_assimilations"] != 1
        or observation_error["mode"] != "prior_std_fraction"
        or profile.guidance["observation_operator"] != "sparse_points"):
        raise ValueError("Expected the released single-update Poisson ES-MDA profile")
    return profile


def ensemble_scores(predicted, reference, projections, scale):
    weights = np.full(len(predicted), 1 / len(predicted))
    flat = predicted.reshape(len(predicted), -1)
    ref_flat = reference.reshape(len(reference), -1)
    # With equal masses W1 is the mean distance between sorted projections.
    projected = np.sort(flat @ projections, axis=0)
    ref_projected = np.sort(ref_flat @ projections, axis=0)
    return {
        "mean_rel_l2": relative_l2(predicted.mean(axis=0), reference.mean(axis=0)),
        "std_rel_l2": relative_l2(predicted.std(axis=0), reference.std(axis=0)),
        "mmd": metrics.weighted_mmd(predicted, reference, weights, weights),
        "swd": float(np.mean(np.abs(projected - ref_projected)) / scale),
        "spectral_rel": metrics.spectral_relative_geomean(
            metrics.radially_averaged_power_spectrum(predicted, weights),
            metrics.radially_averaged_power_spectrum(reference, weights),
        ),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_layout(
    *, name, repeat, coords, heldout, model, std, prior_source_var, heldout_design,
    truths, truth_potentials, obs_noise, prior, prior_potential, reference_bases,
    reference_potentials, draw_noise, projections, field_scale, score_samples,
    observation_error=None,
):
    design = model.sensor_design(coords, std)
    source_cross = idctn(design.reshape(-1, SIZE, SIZE) * std, norm="ortho",
                         axes=(-2, -1)).reshape(SENSORS, -1)
    reference = Conditioner(design @ design.T, source_cross, design @ heldout_design.T)
    empirical = empirical_conditioner(prior, prior_potential, coords, heldout, observation_error)
    exact_field_std = reference.posterior_std(prior_source_var, source_cross)
    exact_response_std = reference.posterior_std(np.sum(heldout_design**2, axis=1),
                                                 reference.response_cross)
    anti_design = (design[:, SIZE] - design[:, 1]) / np.sqrt(2)
    # The design acts on white coefficients. The reported mode is in physical
    # source units, so its cross-covariance includes its prior standard deviation.
    exact_anti_std = reference.posterior_std(std[1, 0]**2, std[1, 0] * anti_design)
    metrics_rows = []
    for case in range(len(truths)):
        observation = at_points(truth_potentials[case], coords) + obs_noise[case]
        oracle_samples = []
        oracle_responses = []
        for fields, potentials, noise in zip(reference_bases, reference_potentials, draw_noise[1:]):
            samples, responses = reference.update(
                fields, at_points(potentials, heldout), observation + noise - at_points(potentials, coords)
            )
            oracle_samples.append(samples)
            oracle_responses.append(responses)
        sample_result, response_result = empirical.update(
            prior[:score_samples], at_points(prior_potential[:score_samples], heldout),
            observation + draw_noise[0] * (empirical.observation_std / NOISE)
            - at_points(prior_potential[:score_samples], coords),
        )
        if case == 0:
            np.testing.assert_allclose(at_points(model.forward(sample_result[:3]), heldout),
                                       response_result[:3], atol=2e-12, rtol=1e-9)
        mean_field, mean_response = reference.update(
            np.zeros((1, SIZE, SIZE)), np.zeros((1, len(heldout))), observation[None, :]
        )
        heldout_truth = at_points(truth_potentials[case], heldout)
        oracle_coverage = np.mean(np.abs(heldout_truth - mean_response[0])
                                  <= ndtri(0.95) * exact_response_std)
        for method, fields, responses in (
            ("ES-MDA", sample_result, response_result),
            ("Independent oracle draws", oracle_samples[1], oracle_responses[1]),
        ):
            low, high = np.quantile(responses, [0.05, 0.95], axis=0)
            metrics_rows.append(dict(
                layout=name, repeat=repeat, case=case, method=method,
                **ensemble_scores(fields, oracle_samples[0], projections, field_scale),
                analytic_mean_rel_l2=relative_l2(fields.mean(axis=0), mean_field[0]),
                analytic_std_rel_l2=relative_l2(fields.std(axis=0).ravel(), exact_field_std),
                heldout_mean_rmse=rms(responses.mean(axis=0) - mean_response[0]),
                heldout_std_rmse=rms(responses.std(axis=0) - exact_response_std),
                heldout_w1=float(np.mean(np.abs(np.sort(responses, axis=0)
                                               - np.sort(oracle_responses[0], axis=0)))),
                heldout_coverage_90=float(np.mean((heldout_truth >= low) & (heldout_truth <= high))),
                heldout_interval_width_90=float(np.mean(high - low)),
                oracle_coverage_90=float(oracle_coverage),
                oracle_interval_width_90=float(np.mean(2 * ndtri(0.95) * exact_response_std)),
                anti_std_ratio_to_prior=float(anti_mode(fields).std(ddof=1) / std[1, 0]),
            ))
    layout_row = dict(
        layout=name, repeat=repeat,
        source_std_ratio_to_prior=rms(exact_field_std) / np.sqrt(prior_source_var.mean()),
        heldout_std_ratio_to_prior=rms(exact_response_std) / rms(np.linalg.norm(heldout_design, axis=1)),
        anti_std_ratio_to_prior=float(exact_anti_std / std[1, 0]),
        anti_sensor_response_max=float(np.max(np.abs(anti_design))),
        esmda_median_observation_std=float(np.median(empirical.observation_std)),
    )
    return layout_row, metrics_rows


def summarize(layout_rows, case_rows):
    summary = []
    for layout in LAYOUTS:
        oracle_rows = [row for row in layout_rows if row["layout"] == layout]
        entry = {"layout": layout}
        for key in oracle_rows[0]:
            if key in {"layout", "repeat"}:
                continue
            entry[f"reference_{key}"] = float(np.mean([row[key] for row in oracle_rows]))
        for method, prefix in (("ES-MDA", "esmda"), ("Independent oracle draws", "oracle_draws")):
            rows = [row for row in case_rows if row["layout"] == layout and row["method"] == method]
            for key in rows[0]:
                if key in {"layout", "repeat", "case", "method"}:
                    continue
                entry[f"{prefix}_{key}"] = float(np.mean([row[key] for row in rows]))
        summary.append(entry)
    return summary


def plot_results(output, layouts, summary):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.5), layout="constrained")
    for ax, name in zip(axes.flat, LAYOUTS):
        row, col = layouts[name].T
        ax.scatter(col, row, s=8, c="#205c87", linewidths=0, alpha=0.84)
        ax.set(xlim=(-2, SIZE + 1), ylim=(SIZE + 1, -2), aspect="equal", title=TITLE[name])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Poisson sensor layouts · 128 potential readings each", fontsize=14)
    fig.savefig(output / "layouts.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), layout="constrained")
    labels = [TITLE[row["layout"]] for row in summary]
    positions = np.arange(len(summary))
    for ax, key, label in zip(axes,
        ("reference_source_std_ratio_to_prior", "reference_anti_std_ratio_to_prior",
         "esmda_heldout_coverage_90"),
        ("Source posterior RMS std / prior", "Antisymmetric mode std / prior",
         "Held-out 90% interval coverage"),
    ):
        series_label = "Exact posterior" if key.startswith("reference_") else "ES-MDA"
        ax.bar(positions, [row[key] for row in summary], color="#205c87", label=series_label)
        ax.set_xticks(positions, labels, rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    axes[1].plot(positions, [row["esmda_anti_std_ratio_to_prior"] for row in summary],
                 "o--", color="#c36b34", label="ES-MDA")
    axes[1].legend(loc="upper right")
    axes[2].plot(positions, [row["esmda_oracle_coverage_90"] for row in summary],
                 "o--", color="#c36b34", label="Exact posterior")
    axes[2].axhline(0.9, color="gray", linestyle=":", linewidth=1)
    axes[2].legend(loc="lower left")
    coverage_min = min(min(row["esmda_heldout_coverage_90"], row["esmda_oracle_coverage_90"])
                       for row in summary)
    axes[2].set_ylim(max(0, min(coverage_min - 0.06, 0.8)), 1.0)
    fig.savefig(output / "findings.png", dpi=175)
    plt.close(fig)


def plot_source_uncertainty(output, layouts, model, std, prior_source_var):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10.7, 7.1), layout="constrained")
    for ax, name in zip(axes.flat, LAYOUTS):
        coords = layouts[name]
        design = model.sensor_design(coords, std)
        cross = idctn(design.reshape(-1, SIZE, SIZE) * std, norm="ortho",
                      axes=(-2, -1)).reshape(SENSORS, -1)
        reference = Conditioner(design @ design.T, cross, np.empty((SENSORS, 0)))
        ratio = reference.posterior_std(prior_source_var, cross) / np.sqrt(prior_source_var)
        image = ax.imshow(ratio.reshape(SIZE, SIZE), vmin=0, vmax=1, cmap="viridis")
        ax.scatter(coords[:, 1], coords[:, 0], s=5, c="white", linewidths=0, alpha=0.65)
        ax.set(title=TITLE[name], xticks=[], yticks=[])
    fig.colorbar(image, ax=axes, label="Posterior standard deviation / prior", shrink=0.9)
    fig.suptitle("How much source uncertainty remains after observing potential?", fontsize=14)
    fig.savefig(output / "source_uncertainty.png", dpi=175)
    plt.close(fig)


def main(args):
    observation_error = None
    if args.solver_settings == "paper":
        profile = load_esmda_profile()
        if args.prior_members != profile.sampling["ensemble_size"]:
            raise ValueError(f"The released profile uses {profile.sampling['ensemble_size']} prior members")
        observation_error = profile.guidance["observation_error"]
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    groups = [make_layouts(rng) for _ in range(args.repeats)]
    heldout = make_heldout(groups, rng)
    np.savez(args.output / "coordinates.npz", heldout=heldout,
             **{name: np.stack([group[name] for group in groups]) for name in LAYOUTS})
    model = Poisson()
    std = prior_std()
    dct_basis = dctn(np.eye(SIZE), norm="ortho", axes=(0,))
    prior_source_var = ((dct_basis.T**2) @ std**2 @ (dct_basis**2)).ravel()
    field_scale = float(np.sqrt(prior_source_var.mean()))
    heldout_design = model.sensor_design(heldout, std)
    projector_rng = np.random.default_rng(metrics.DEFAULT_PROJECTION_SEED)
    projections = np.column_stack([metrics.sample_grf_projection(SIZE, SIZE, projector_rng).ravel()
                                   for _ in range(metrics.DEFAULT_N_PROJECTIONS)])
    def prior_sample(count):
        return idctn(rng.standard_normal((count, SIZE, SIZE)) * std, norm="ortho", axes=(-2, -1))

    layout_rows, case_rows = [], []
    for repeat, layouts in enumerate(groups):
        truths = prior_sample(args.cases)
        truth_potentials = model.forward(truths)
        prior = prior_sample(args.prior_members)
        prior_potential = model.forward(prior)
        references = [prior_sample(args.score_samples) for _ in range(2)]
        reference_potentials = [model.forward(fields) for fields in references]
        obs_noise = NOISE * rng.standard_normal((args.cases, SENSORS))
        draw_noise = NOISE * rng.standard_normal((3, args.score_samples, SENSORS))
        for name, coords in layouts.items():
            def rotate_if_vertical(fields):
                return np.rot90(fields, axes=(-2, -1)) if name == "vertical" else fields
            row, cases = evaluate_layout(
                name=name, repeat=repeat, coords=coords, heldout=heldout, model=model, std=std,
                prior_source_var=prior_source_var, heldout_design=heldout_design,
                truths=rotate_if_vertical(truths), truth_potentials=rotate_if_vertical(truth_potentials),
                obs_noise=obs_noise, prior=prior, prior_potential=prior_potential,
                reference_bases=[rotate_if_vertical(fields) for fields in references],
                reference_potentials=[rotate_if_vertical(fields) for fields in reference_potentials],
                draw_noise=draw_noise, projections=projections, field_scale=field_scale,
                score_samples=args.score_samples,
                observation_error=observation_error,
            )
            layout_rows.append(row)
            case_rows.extend(cases)
            print(f"Repeat {repeat + 1}/{len(groups)} · {TITLE[name]} complete", flush=True)
    summary = summarize(layout_rows, case_rows)
    write_csv(args.output / "layout_reference.csv", layout_rows)
    write_csv(args.output / "case_scores.csv", case_rows)
    write_csv(args.output / "summary.csv", summary)
    plot_results(args.output, groups[0], summary)
    plot_source_uncertainty(args.output, groups[0], model, std, prior_source_var)
    (args.output / "settings.json").write_text(json.dumps(
        dict(grid=SIZE, sensors=SENSORS, alpha=ALPHA, tau=TAU, noise_std=NOISE,
             seed=args.seed, layout_repeats=args.repeats, cases_per_repeat=args.cases,
             prior_members=args.prior_members, samples_per_score=args.score_samples,
             swd_projections=metrics.DEFAULT_N_PROJECTIONS, swd_global_prior_std=field_scale,
             heldout_per_case=len(heldout), esmda_solver_settings=args.solver_settings,
             esmda_profile_observation_error=observation_error),
        indent=2,
    ) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--solver-settings", choices=("matched", "paper"), default="matched")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", type=int, default=16, help="Hidden fields per layout repeat")
    parser.add_argument("--prior-members", type=int, default=512)
    parser.add_argument("--score-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / ("paper_esmda_settings" if args.solver_settings == "paper" else "results")
    if min(args.repeats, args.cases) < 1 or not 2 <= args.score_samples <= args.prior_members:
        parser.error("Use positive layout and case counts, and 2 <= score-samples <= prior-members")
    with threadpool_limits(limits=4, user_api="blas"):
        main(args)
