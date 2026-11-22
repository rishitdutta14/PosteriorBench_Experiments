"""Compare the saved LTMI light maps and material layers with weighted references."""

import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from datasets import load_from_disk
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
from matplotlib import pyplot as plt

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
sys.path.insert(0, str(RUN))
from reproduce import DATA, PRIOR
from posteriorbench.datasets import get_dataset
from posteriorbench.evaluation.metrics import normalize_weights, weighted_mmd, weighted_wasserstein_1d
from posteriorbench.methods.esmda import _area_average_batch, _read_prior_ensemble
from posteriorbench.training_source import open_training_field_source

LAYER_NAMES = ("average", "layer_1", "layer_2", "half_difference")
LAYER_LABELS = ("Layer average", "Layer 1", "Layer 2", "Half difference")
LIGHT_NAMES = ("full", "coarse", "within_block", "interior_detail")
LIGHT_LABELS = ("Full image", "Block averages", "All detail", "Interior detail")


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def moments(fields, weights):
    mean = np.average(fields, weights=weights, axis=0)
    variance = np.average((fields - mean) ** 2, weights=weights, axis=0)
    return mean, np.sqrt(variance)


def compare(pred, ref, pw, rw):
    pm, ps = moments(pred, pw)
    rm, rs = moments(ref, rw)
    pred_draw_rms = np.sqrt(np.mean(pred ** 2, axis=(1, 2)))
    ref_draw_rms = np.sqrt(np.mean(ref ** 2, axis=(1, 2)))
    return {
        "mean_rmse": rms(pm - rm),
        "mean_relative_error": rms(pm - rm) / rms(rm),
        "std_relative_error": rms(ps - rs) / rms(rs),
        "pred_uncertainty_rms": rms(ps),
        "ref_uncertainty_rms": rms(rs),
        "uncertainty_ratio": rms(ps) / rms(rs),
        "mmd": weighted_mmd(pred, ref, pw, rw),
        "draw_rms_wasserstein": weighted_wasserstein_1d(pred_draw_rms, ref_draw_rms, pw, rw),
        "pred_signal_rms": float(np.sqrt(pw @ (pred_draw_rms ** 2))),
        "ref_signal_rms": float(np.sqrt(rw @ (ref_draw_rms ** 2))),
    }


def light_parts(fields, observation_shape):
    coarse = _area_average_batch(fields, observation_shape)
    by = fields.shape[1] // coarse.shape[1]
    bx = fields.shape[2] // coarse.shape[2]
    fine = fields - coarse.repeat(by, axis=1).repeat(bx, axis=2)
    np.testing.assert_allclose(_area_average_batch(fine, observation_shape), 0, atol=1e-12)
    return dict(zip(LIGHT_NAMES, (fields, coarse, fine, fine[:, by:-by, bx:-bx])))


def layer_parts(fields):
    first, second = fields[:, 0], fields[:, 1]
    return dict(zip(LAYER_NAMES, ((first + second) * 0.5, first, second, (first - second) * 0.5)))


def layer_correlation(fields, weights):
    centered = fields - np.average(fields, weights=weights, axis=0)
    left, right = centered[:, 0], centered[:, 1]
    covariance = float(weights @ np.mean(left * right, axis=(1, 2)))
    scale = np.sqrt((weights @ np.mean(left ** 2, axis=(1, 2))) * (weights @ np.mean(right ** 2, axis=(1, 2))))
    return covariance / float(scale)


def read_case(case_id, row):
    with h5py.File(RUN / "predictions" / case_id / "generated.h5") as handle:
        assert handle.attrs["case_id"] == case_id
        pred_a = np.stack([handle[key][:] for key in ("sigma_t1", "sigma_t2")], axis=1).astype(np.float64)
        pred_u = handle["u"][:].astype(np.float64)
        pw = normalize_weights(handle["w"][:], len(pred_a))
        seed = int(handle.attrs["seed"])
    ref_a = np.asarray(row["posterior_samples_a"], dtype=np.float64)
    ref_u = np.asarray(row["posterior_samples_u"], dtype=np.float64)[:, 0]
    rw = normalize_weights(row["posterior_weights"], len(ref_a))
    obs = np.asarray(row["metadata"]["obs_u_channel0"], dtype=np.float64)
    assert pred_a.shape == ref_a.shape == (100, 2, 64, 64)
    assert pred_u.shape == ref_u.shape == (100, 64, 64) and obs.shape == (16, 16)
    assert str(int(row["case_id"]) + 1) == case_id.rsplit("_", 1)[1]
    return pred_a, ref_a, pred_u, ref_u, pw, rw, obs, seed


def case_metrics(case_id, row, prior_source, ensemble_size):
    pred_a, ref_a, pred_u, ref_u, pw, rw, obs, seed = read_case(case_id, row)
    # Recover the initial fields at the exact saved member indices using the released seed rule.
    indices = np.random.default_rng(seed).choice(prior_source.num_samples, ensemble_size, replace=False)[:len(pw)]
    prior_u = _read_prior_ensemble(prior_source, ("u",), indices, 64)[:, 0].astype(np.float64)
    pred_light = light_parts(pred_u, obs.shape)
    ref_light = light_parts(ref_u, obs.shape)
    prior_light = light_parts(prior_u, obs.shape)
    output = {"case_id": case_id, "reference_effective_sample_size": float(1 / (rw @ rw))}
    for name in LIGHT_NAMES:
        for model, fields in (("esmda", pred_light), ("prior", prior_light)):
            output.update({f"light_{model}_{name}_{key}": val for key, val in compare(fields[name], ref_light[name], pw, rw).items()})
    for name, fields, weights in (("esmda", pred_light, pw), ("prior", prior_light, pw), ("reference", ref_light, rw)):
        output[f"light_{name}_coarse_observation_rmse"] = rms(np.average(fields["coarse"], weights=weights, axis=0) - obs)
    pred_mat, ref_mat = layer_parts(pred_a), layer_parts(ref_a)
    for name in LAYER_NAMES:
        values = compare(pred_mat[name], ref_mat[name], pw, rw)
        if name != "half_difference":
            values["pred_negative_pixel_fraction"] = float(pw @ np.mean(pred_mat[name] < 0, axis=(1, 2)))
            values["ref_negative_pixel_fraction"] = float(rw @ np.mean(ref_mat[name] < 0, axis=(1, 2)))
        output.update({f"material_{name}_{key}": val for key, val in values.items()})

    output["material_esmda_cross_layer_correlation"] = layer_correlation(pred_a, pw)
    output["material_reference_cross_layer_correlation"] = layer_correlation(ref_a, rw)
    common = output["material_average_mean_rmse"] ** 2
    contrast = output["material_half_difference_mean_rmse"] ** 2
    layer_mse = np.mean([output[f"material_{name}_mean_rmse"] ** 2 for name in ("layer_1", "layer_2")])
    np.testing.assert_allclose(common + contrast, layer_mse, rtol=1e-12)
    output["material_hidden_mean_mse_fraction"] = contrast / (common + contrast)
    for model in ("pred", "ref"):
        common_variance = output[f"material_average_{model}_uncertainty_rms"] ** 2
        contrast_variance = output[f"material_half_difference_{model}_uncertainty_rms"] ** 2
        output[f"material_{model}_hidden_variance_fraction"] = contrast_variance / (common_variance + contrast_variance)
    return output


def write_tables(records):
    with (OUT / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    for column in records[0]:
        if column != "case_id":
            values = [row[column] for row in records]
            assert np.isfinite(values).all(), column
            q25, median, q75 = np.percentile(values, [25, 50, 75])
            summary[column] = dict(mean=float(np.mean(values)), median=float(median), q25=float(q25), q75=float(q75))
    common = np.sum([row["material_average_mean_rmse"] ** 2 for row in records])
    contrast = np.sum([row["material_half_difference_mean_rmse"] ** 2 for row in records])
    result = dict(cases=len(records), draws_per_ensemble=100, pooled_hidden_mean_mse_fraction=float(contrast / (common + contrast)), metrics=summary)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def representative(records, column):
    median = np.median([row[column] for row in records])
    return min(range(len(records)), key=lambda index: abs(records[index][column] - median))


def plot_light(records, case_id, row):
    _, _, pred_u, ref_u, pw, rw, obs, _ = read_case(case_id, row)
    pm, ps = moments(light_parts(pred_u, obs.shape)["interior_detail"], pw)
    rm, rs = moments(light_parts(ref_u, obs.shape)["interior_detail"], rw)
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.1), layout="constrained", width_ratios=[1, 1, 1.7])
    mean_limit = max(np.abs(pm).max(), np.abs(rm).max())
    std_limit = max(ps.max(), rs.max())
    for j, (mean, std, label) in enumerate(((rm, rs, "Weighted reference"), (pm, ps, "ES-MDA"))):
        mean_img = ax[0, j].imshow(mean, cmap="RdBu_r", vmin=-mean_limit, vmax=mean_limit)
        std_img = ax[1, j].imshow(std, cmap="viridis", vmin=0, vmax=std_limit)
        for i, suffix in enumerate(("mean", "standard deviation")):
            ax[i, j].set_title(f"{label}\n{suffix}", fontsize=10)
            ax[i, j].axis("off")
    fig.colorbar(mean_img, ax=ax[0, :2], shrink=.8, label="Light response")
    fig.colorbar(std_img, ax=ax[1, :2], shrink=.8, label="Light response")
    for axis, key, ylabel in ((ax[0, 2], "mean_rmse", "Posterior mean RMSE"), (ax[1, 2], "uncertainty_ratio", "Uncertainty RMS / reference")):
        for offset, model, color, label in ((-.16, "prior", "#9A6851", "Initial members"), (.16, "esmda", "#28768D", "After ES-MDA")):
            data = [[r[f"light_{model}_{name}_{key}"] for r in records] for name in LIGHT_NAMES]
            box = axis.boxplot(data, positions=np.arange(len(LIGHT_NAMES)) + offset, widths=.25, showfliers=False, patch_artist=True, manage_ticks=False)
            for patch in box["boxes"]:
                patch.set_facecolor(color)
            axis.plot([], [], color=color, linewidth=6, label=label)
        axis.set_xticks(np.arange(len(LIGHT_NAMES)), LIGHT_LABELS, fontsize=8)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.25)
    ax[0, 2].legend(frameon=False, fontsize=8)
    ax[1, 2].axhline(1, linestyle="--", color="0.4", lw=1)
    fig.suptitle(f"LTMI light response · all 90 cases and interior detail from case {case_id.rsplit('_', 1)[1]}")
    fig.savefig(OUT / "light_response.png", dpi=190)
    plt.close(fig)


def plot_layers(records, case_id, row):
    pred_a, ref_a, _, _, pw, rw, _, _ = read_case(case_id, row)
    pred, ref = layer_parts(pred_a), layer_parts(ref_a)
    fig, ax = plt.subplots(3, 4, figsize=(11.2, 7.6), layout="constrained")
    discrepancies = [moments(pred[name], pw)[0] - moments(ref[name], rw)[0] for name in LAYER_NAMES]
    limit = max(np.abs(diff).max() for diff in discrepancies)
    for i, (name, label, diff) in enumerate(zip(LAYER_NAMES, LAYER_LABELS, discrepancies)):
        ref_mean = moments(ref[name], rw)[0]
        cmap = "RdBu_r" if name == "half_difference" else "viridis"
        bounds = (-2.5, 2.5) if name == "half_difference" else (0.0, 5.0)
        im = ax[0, i].imshow(ref_mean, cmap=cmap, vmin=bounds[0], vmax=bounds[1])
        err = ax[1, i].imshow(diff, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax[0, i].set_title(label, fontsize=11)
        for image_ax in ax[:2, i]:
            image_ax.set_xticks([])
            image_ax.set_yticks([])
        axis = ax[2, i]
        data = [row[f"material_{name}_uncertainty_ratio"] for row in records]
        axis.hist(data, bins=np.linspace(0.85, 1.45, 21), color="#28768D")
        axis.axvline(1, linestyle="--", color="0.35", lw=1)
        axis.set_xlabel("Uncertainty RMS / reference", fontsize=8)
        axis.set_xlim(.85, 1.45)
    ymax = max(axis.get_ylim()[1] for axis in ax[2])
    for axis in ax[2]:
        axis.set_ylim(0, ymax)
    ax[0, 0].set_ylabel("Reference mean")
    ax[1, 0].set_ylabel("ES-MDA mean error")
    ax[2, 0].set_ylabel("Cases")
    fig.colorbar(ax[0, 0].images[0], ax=ax[0, :3], label="Extinction", shrink=.8)
    fig.colorbar(im, ax=ax[0, 3], shrink=.8)
    fig.colorbar(err, ax=ax[1, :], label="Extinction error", shrink=.8)
    fig.suptitle(f"LTMI material layers · all 90 cases and example {case_id.rsplit('_', 1)[1]}")
    fig.savefig(OUT / "material_layers.png", dpi=190)
    plt.close(fig)


def main():
    settings = json.loads((RUN / "run_settings.json").read_text())
    cases = [ref for ref in get_dataset("light_transport").discover(DATA) if ref.split == "test"]
    rows = load_from_disk(Path(cases[0].source) / "test").with_format("numpy")
    prior = open_training_field_source(PRIOR, dataset_name="light_transport")
    assert len(rows) == len(cases) == 90 and [ref.stem for ref in cases] == settings["case_ids"]
    assert prior.num_samples == settings["prior_rows"] == 100_000
    records = []
    for number, ref in enumerate(cases, start=1):
        records.append(case_metrics(ref.stem, rows[ref.index], prior, settings["sampling"]["ensemble_size"]))
        if number % 10 == 0:
            print(f"Scored {number}/{len(cases)} cases", flush=True)
    summary = write_tables(records)
    official = json.loads((RUN / "evaluation" / "metrics.json").read_text())["summary"]["sigma_t"]
    for key, official_key in (("mean_relative_error", "mean_rel_l2"), ("std_relative_error", "std_rel_l2"), ("mmd", "mmd")):
        np.testing.assert_allclose(summary["metrics"][f"material_average_{key}"]["mean"], official[official_key]["mean"], atol=1e-7)
    light_case = cases[representative(records, "light_esmda_interior_detail_std_relative_error")]
    layer_case = cases[representative(records, "material_hidden_mean_mse_fraction")]
    plot_light(records, light_case.stem, rows[light_case.index])
    plot_layers(records, layer_case.stem, rows[layer_case.index])


if __name__ == "__main__":
    with threadpool_limits(limits=4, user_api="blas"):
        main()
