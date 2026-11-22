"""Compare the same four groups of sensors placed together or spread across the grid."""

import json
import os

import numpy as np
from scipy.fft import dctn, idctn
from threadpoolctl import threadpool_limits

from run_ablation import (
    ROOT, SIZE, SENSORS, NOISE, Conditioner, Poisson, at_points,
    empirical_conditioner, ensemble_scores, load_esmda_profile, make_heldout,
    make_layouts, metrics, prior_std, rms, write_csv,
)

NAMES = ("one_cluster", "four_clusters")
LABELS = {"one_cluster": "One cluster", "four_clusters": "Four clusters"}
OUTPUT = ROOT / "cluster_ablation"


def cluster_pair(rng):
    """Translate four identical 4 x 8 tiles while keeping their common center."""
    center = rng.integers(59, 70, size=2)
    separation = rng.integers(26, 34, size=2)
    row, col = np.indices((4, 8))
    tile = np.column_stack((2 * row.ravel() - 3, 2 * col.ravel() - 7))
    packed = np.array([4, 8])
    if rng.integers(2):
        tile = tile[:, ::-1]
        packed = packed[::-1]
    quadrants = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    masks = {}
    for name, offset in zip(NAMES, (packed, separation)):
        coords = (center + quadrants[:, None] * offset + tile).reshape(-1, 2)
        if len(np.unique(coords, axis=0)) != SENSORS or np.any((coords < 0) | (coords >= SIZE)):
            raise ValueError(f"Invalid sensors for {name}")
        np.testing.assert_array_equal(coords.mean(axis=0), center)
        groups = coords.reshape(4, 32, 2)
        np.testing.assert_array_equal(groups - groups[:, :1], np.broadcast_to(tile-tile[:1], groups.shape))
        masks[name] = coords
    return masks


def source_update(conditioner, fields, residual):
    empty = np.empty((len(fields), 0))
    return conditioner.update(fields, empty, residual)[0]


def draw(rng, std, count):
    white = rng.standard_normal((count, SIZE, SIZE))
    return idctn(white * std, axes=(-2, -1), norm="ortho")


def summarize(layout_rows, case_rows, pair_rows):
    summary = {}
    for name in NAMES:
        layouts = [row for row in layout_rows if row["layout"] == name]
        info = {key: float(np.mean([row[key] for row in layouts]))
                for key in layouts[0] if key not in ("layout", "repeat")}
        for method, prefix in (("ES-MDA", "esmda"), ("Independent reference draws", "reference_draws")):
            subset = [row for row in case_rows if row["layout"] == name and row["method"] == method]
            info.update({f"{prefix}_{key}": float(np.mean([row[key] for row in subset]))
                         for key in subset[0] if key not in ("repeat", "case", "layout", "method")})
        summary[name] = info
    paired = {key: float(np.mean([row[key] for row in pair_rows]))
              for key in pair_rows[0] if key not in ("repeat", "case")}
    return summary, paired


def make_figure(truth, layouts, preview):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_scale = max(np.abs(truth).max(), max(np.abs(f).max() for d in preview.values()
                                              for f in (d[0], d[1])))
    spread_scale = max(d[index].max() for d in preview.values() for index in (2, 3))
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.3), layout="constrained")
    titles = ("Hidden source + sensors", "Reference mean", "ES-MDA mean",
              "Reference spread", "ES-MDA spread")
    for row, name in enumerate(NAMES):
        maps = [truth, *preview[name]]
        for col, field in enumerate(maps):
            if col < 3:
                source_img = axes[row, col].imshow(field, cmap="RdBu_r",
                                                   vmin=-mean_scale, vmax=mean_scale)
            else:
                spread_img = axes[row, col].imshow(field, cmap="viridis", vmin=0, vmax=spread_scale)
            axes[row, col].set(xticks=[], yticks=[])
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=10)
        row_coords, col_coords = layouts[name].T
        axes[row, 0].scatter(col_coords, row_coords, c="#17181c", s=6, linewidths=0.)
        axes[row, 0].set_ylabel(LABELS[name])
    fig.colorbar(source_img, ax=axes[:, :3], label="Source strength", shrink=0.88)
    fig.colorbar(spread_img, ax=axes[:, 3:], label="Std across source draws", shrink=0.88)
    fig.suptitle("Poisson case 00 · Published ES-MDA settings\n"
                 "Same hidden source and 128 sensors. Dots mark potential readings.", fontsize=13)
    fig.savefig(OUTPUT / "cluster_comparison.png", dpi=175)
    plt.close(fig)


def main():
    settings = json.loads((ROOT / "paper_esmda_settings/settings.json").read_text(encoding="utf-8"))
    profile = load_esmda_profile()
    observation_error = profile.guidance["observation_error"]
    count = settings["samples_per_score"]
    n_cases = settings["cases_per_repeat"]
    repeats = settings["layout_repeats"]
    rng = np.random.default_rng(settings["seed"])
    original_masks = [make_layouts(rng) for _ in range(repeats)]
    make_heldout(original_masks, rng)  # Preserve the original case-generation sequence.
    placement_rng = np.random.default_rng(settings["seed"] + 57)
    layouts = [cluster_pair(placement_rng) for _ in range(repeats)]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT / "coordinates.npz",
             **{name: np.stack([repeat[name] for repeat in layouts]) for name in NAMES})
    model = Poisson()
    std = prior_std()
    dct = dctn(np.eye(SIZE), norm="ortho", axes=(0,))
    prior_source_var = ((dct.T**2) @ std**2 @ (dct**2)).ravel()
    prior_scale = float(np.sqrt(prior_source_var.mean()))
    projection_rng = np.random.default_rng(metrics.DEFAULT_PROJECTION_SEED)
    projections = np.column_stack([metrics.sample_grf_projection(SIZE, SIZE, projection_rng).ravel()
                                   for _ in range(metrics.DEFAULT_N_PROJECTIONS)])
    layout_rows, case_rows, pair_rows = [], [], []
    for repeat, masks in enumerate(layouts):
        truths = draw(rng, std, n_cases)
        prior = draw(rng, std, profile.sampling["ensemble_size"])
        reference_bases = [draw(rng, std, count), draw(rng, std, count)]
        measurement_noise = rng.normal(0., NOISE, (n_cases, SENSORS))
        white_noise = rng.standard_normal((3, count, SENSORS))
        truth_y, prior_y = model.forward(truths), model.forward(prior)
        reference_y = [model.forward(fields) for fields in reference_bases]
        maps_by_case = {}
        preview = {}

        for name, coords in masks.items():
            design = model.sensor_design(coords, std)
            cross = idctn(design.reshape(-1, SIZE, SIZE) * std, axes=(-2, -1),
                          norm="ortho").reshape(SENSORS, -1)
            ref = Conditioner(design @ design.T, cross, np.empty((SENSORS, 0)))
            ref_std = ref.posterior_std(prior_source_var, cross).reshape(SIZE, SIZE)
            es = empirical_conditioner(prior, prior_y, coords, np.empty((0, 2), dtype=int),
                                       observation_error)
            layout_rows.append(dict(repeat=repeat, layout=name,
                                    reference_rms_std_prior_ratio=rms(ref_std)/prior_scale,
                                    esmda_median_assumed_obs_std=float(np.median(es.observation_std))))
            observations = at_points(truth_y, coords) + measurement_noise
            prior_sensors = at_points(prior_y[:count], coords)
            base_sensors = [at_points(y, coords) for y in reference_y]
            for i, observed in enumerate(observations):
                y = observed[None, :]
                reference = [
                    source_update(ref, fields, y + NOISE * e - readings)
                    for fields, e, readings in zip(reference_bases, white_noise[1:], base_sensors)
                ]
                predicted = source_update(es, prior[:count],
                                          y + white_noise[0] * es.observation_std - prior_sensors)
                ref_mean = source_update(ref, np.zeros((1, SIZE, SIZE)), y)[0]
                solver_mean, solver_std = predicted.mean(axis=0), predicted.std(axis=0)
                maps_by_case[i, name] = (ref_mean, solver_mean, ref_std, solver_std)
                if repeat == 0 and i == 0:
                    preview[name] = maps_by_case[i, name]
                for method, fields in (("ES-MDA", predicted),
                                       ("Independent reference draws", reference[1])):
                    case_rows.append(dict(
                        repeat=repeat, case=i, layout=name, method=method,
                        **ensemble_scores(fields, reference[0], projections, prior_scale),
                        rms_std_prior_ratio=rms(fields.std(axis=0))/prior_scale,
                    ))
            print(f"Repeat {repeat + 1}/{repeats} · {LABELS[name]} complete", flush=True)

        for i in range(n_cases):
            packed, dispersed = (maps_by_case[i, name] for name in NAMES)
            pair_rows.append(dict(
                repeat=repeat, case=i,
                reference_mean_rms_change=rms(packed[0] - dispersed[0]),
                esmda_mean_rms_change=rms(packed[1] - dispersed[1]),
                reference_spread_rms_change=rms(packed[2] - dispersed[2]),
                esmda_spread_rms_change=rms(packed[3] - dispersed[3]),
            ))
        if repeat == 0:
            make_figure(truths[0], masks, preview)
    summary, paired = summarize(layout_rows, case_rows, pair_rows)
    write_csv(OUTPUT / "case_scores.csv", case_rows)
    write_csv(OUTPUT / "layout_references.csv", layout_rows)
    write_csv(OUTPUT / "paired_map_changes.csv", pair_rows)
    write_csv(OUTPUT / "summary.csv", [dict(layout=name, **summary[name]) for name in NAMES])
    report = dict(cases_per_layout=n_cases*repeats, repeats=repeats, sensors=SENSORS,
                  source_prior={"alpha": settings["alpha"], "tau": settings["tau"]},
                  observation_std=NOISE, solver_observation_error=observation_error,
                  ensemble_size=profile.sampling["ensemble_size"], samples_per_score=count,
                  paired_map_changes_rms=paired)
    (OUTPUT / "settings.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"layouts": summary, "paired_map_changes": paired}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4, user_api="blas"):
        main()
