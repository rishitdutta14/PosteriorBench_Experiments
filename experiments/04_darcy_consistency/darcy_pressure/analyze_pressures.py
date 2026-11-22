"""Compare ES-MDA pressure predictions with the Darcy reference and hidden truth."""

import csv
import os
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

from darcy_grid import (
    FORCING, SIZE, TO_CENTERS, divergence, face_coefficients,
    patch_rms, recover_pressure, rms, solve,
)
from solve_pressures import BASELINE, OUTPUT, SAMPLES_PER_CASE

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = HERE.parent
DATA = Path(os.environ.get("POSTERIORBENCH_DATA", str(PROJECT / "data")))
REPO = PROJECT / "vendor" / "PosteriorBench"
PATCH_SIDES = (1, 4, 16, 32)
FIELDS = ("internal", "numerical", "reference")
FLUX_PAIRS = ("internal_binary", "numerical_binary", "internal_raw_clipped", "reference", "hidden_truth")
NUMERICAL_WEIGHT = np.full(SAMPLES_PER_CASE, 1.0 / SAMPLES_PER_CASE)
sys.path.insert(0, str(REPO))
from posteriorbench.datasets import get_dataset
from posteriorbench.evaluation.metrics import sample_grf_projection, weighted_wasserstein_1d


def load_case(dataset, ref):
    row = dataset._load_hf_row(ref)
    truth = np.asarray(row["u_ref"], dtype=np.float32).squeeze()
    truth_a = np.asarray(row["a_ref"], dtype=np.float32).squeeze()
    sensors = np.asarray(row["metadata"]["sensor_coords"], dtype=np.int64)
    reference_u = np.asarray(row["posterior_samples_u"], dtype=np.float32).squeeze()
    reference_a = np.asarray(row["posterior_samples_a"], dtype=np.float32).squeeze()
    weights = np.asarray(row["posterior_weights"], dtype=np.float64)
    if sensors.shape != (128, 2) or len(np.unique(sensors, axis=0)) != len(sensors):
        raise ValueError(f"Invalid observed locations for {ref.stem}")
    if weights.shape != (SAMPLES_PER_CASE,) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(f"Invalid reference weights for {ref.stem}")
    weights /= weights.sum()

    source = BASELINE / "predictions" / ref.stem / "generated.h5"
    with h5py.File(source, "r") as saved:
        internal_u = saved["u"][:]
        a = saved["a"][:]
        a_raw = saved["a_raw"][:]
    with h5py.File(OUTPUT / f"{ref.stem}.h5", "r") as saved:
        if saved.attrs["permeability_sha256"] != hashlib.sha256(a.tobytes()).hexdigest():
            raise ValueError(f"Numerical pressures do not match the baseline inputs for {ref.stem}")
        numerical_u = saved["u_num"][:]

    ensembles = (internal_u, numerical_u, reference_u, a, a_raw, reference_a)
    shape = (SAMPLES_PER_CASE, SIZE, SIZE)
    if any(array.shape != shape or not np.isfinite(array).all() for array in ensembles):
        raise ValueError(f"Invalid ensembles for {ref.stem}")
    if any(array.shape != (SIZE, SIZE) or not np.isfinite(array).all() for array in (truth, truth_a)):
        raise ValueError(f"Invalid hidden fields for {ref.stem}")
    observed = np.zeros((SIZE, SIZE), dtype=bool)
    observed[sensors[:, 0], sensors[:, 1]] = True
    return dict(
        truth=truth, truth_a=truth_a, observed=observed, a=a, a_raw=a_raw,
        reference_a=reference_a, weights=weights,
        fields=dict(internal=internal_u, numerical=numerical_u, reference=reference_u),
    )


def distribution_summary(samples, weights, truth):
    """Pointwise moments, central interval and CRPS for a weighted empirical law."""
    values = np.ascontiguousarray(samples.reshape(len(samples), -1).T, dtype=np.float64)
    order = np.argsort(values, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    mass = weights[order]
    cdf = np.cumsum(mass, axis=1)
    mean = np.sum(values * mass, axis=1)
    std = np.sqrt(np.sum(mass * (values - mean[:, None]) ** 2, axis=1))

    def inverse_cdf(q):
        return values[np.arange(len(values)), np.argmax(cdf >= q, axis=1)]

    low, high = inverse_cdf(0.05), inverse_cdf(0.95)
    target = truth.ravel().astype(np.float64)
    crps = (
        np.sum(mass * np.abs(values - target[:, None]), axis=1)
        - np.sum(np.diff(values, axis=1) * cdf[:, :-1] * (1 - cdf[:, :-1]), axis=1)
    )
    if np.min(crps) < -1e-12:
        raise ValueError("Negative CRPS outside rounding tolerance")
    return {
        key: value.reshape(SIZE, SIZE)
        for key, value in dict(
            mean=mean, std=std, coverage=(target >= low) & (target <= high),
            width=high - low, crps=np.maximum(crps, 0.0),
        ).items()
    }


def pointwise_w1(x, y, wx, wy):
    """Integrate the absolute difference of the two empirical CDFs at each pixel."""
    pooled = np.ascontiguousarray(np.concatenate((x, y), axis=0).reshape(len(x) + len(y), -1).T)
    order = np.argsort(pooled, axis=1)
    signed_mass = np.concatenate((wx, -wy))
    levels = np.take_along_axis(pooled, order, axis=1).astype(np.float64)
    difference = np.cumsum(signed_mass[order][:, :-1], axis=1)
    return np.sum(np.diff(levels, axis=1) * np.abs(difference), axis=1).reshape(SIZE, SIZE)


def pressure_metrics(case_id, case, directions, pressure_scale):
    truth, observed = case["truth"], case["observed"]
    regions = {"all": np.ones_like(observed), "observed": observed, "held_out": ~observed}
    weights = {"reference": case["weights"], "numerical": NUMERICAL_WEIGHT, "internal": NUMERICAL_WEIGHT}
    ensembles = case["fields"]
    summaries = {
        name: distribution_summary(samples, weights[name], truth)
        for name, samples in ensembles.items()
    }
    ref_summary = summaries["reference"]
    projected_ref = (ensembles["reference"].reshape(SAMPLES_PER_CASE, -1) @ directions.T) / pressure_scale
    rows = []
    for name, fields in summaries.items():
        if name == "reference":
            w1 = np.zeros_like(truth)
            swd = 0.0
        else:
            w1 = pointwise_w1(ensembles[name], ensembles["reference"], weights[name], weights["reference"])
            projected = (ensembles[name].reshape(SAMPLES_PER_CASE, -1) @ directions.T) / pressure_scale
            swd = float(np.mean([
                weighted_wasserstein_1d(x, y, weights[name], weights["reference"])
                for x, y in zip(projected.T, projected_ref.T)
            ]))
        for region, mask in regions.items():
            rows.append(dict(
                case_id=case_id, ensemble=name, region=region,
                reference_mean_relative_l2=rms((fields["mean"] - ref_summary["mean"])[mask]) / rms(ref_summary["mean"][mask]),
                reference_std_relative_l2=rms((fields["std"] - ref_summary["std"])[mask]) / rms(ref_summary["std"][mask]),
                reference_w1=float(w1[mask].mean()),
                reference_swd_pressure_zscore=swd,
                truth_mean_rmse=rms((fields["mean"] - truth)[mask]),
                truth_crps=float(fields["crps"][mask].mean()),
                truth_90_coverage=float(fields["coverage"][mask].mean()),
                interval_90_width=float(fields["width"][mask].mean()),
                mean_pointwise_std=float(fields["std"][mask].mean()),
            ))
    difference = ensembles["numerical"].astype(np.float64) - ensembles["internal"]
    paired = dict(case_id=case_id)
    for region, mask in regions.items():
        member_rms = np.sqrt(np.mean(difference[:, mask] ** 2, axis=1))
        paired[f"mean_paired_rmse_{region}"] = float(member_rms.mean())
        paired[f"median_paired_rmse_{region}"] = float(np.median(member_rms))
    return rows, paired


def conservation_metrics(case_id, case):
    """Use the original face flux on reconstructed nodal pressures for all pairs."""
    fields = case["fields"]
    interior = {name: recover_pressure(values) for name, values in fields.items()}
    names = {
        "internal_binary": (case["a"], interior["internal"], NUMERICAL_WEIGHT),
        # Isolate the map projection while retaining the released nodal clipping.
        "internal_raw_clipped": (case["a_raw"], interior["internal"], NUMERICAL_WEIGHT),
        "numerical_binary": (case["a"], interior["numerical"], NUMERICAL_WEIGHT),
        "reference": (case["reference_a"], interior["reference"], case["weights"]),
        "hidden_truth": (case["truth_a"][None], recover_pressure(case["truth"])[None], np.ones(1)),
    }
    rows = []
    solved_reference = []
    control_indices = {0, SAMPLES_PER_CASE - 1, int(np.argmax(case["weights"]))}
    for name in FLUX_PAIRS:
        permeability, pressures, weights = names[name]
        scores = np.empty((len(permeability), len(PATCH_SIDES)))
        for i, (a, u) in enumerate(zip(permeability, pressures)):
            faces = face_coefficients(a)
            residual = divergence(u, faces) - FORCING
            scores[i] = list(patch_rms(residual, PATCH_SIDES).values())
            if name == "reference" and i in control_indices:
                solved_reference.append(rms(solve(faces) - fields["reference"][i]))
            if name == "hidden_truth":
                truth_solve_rmse = rms(solve(faces) - case["truth"])
        for side, values in zip(PATCH_SIDES, scores.T):
            rows.append(dict(
                case_id=case_id, pressure_and_permeability=name,
                patch_side=side, mean_member_rms_fraction=float(weights @ values),
            ))

    checks = dict(
        case_id=case_id, truth_resolve_rmse=truth_solve_rmse,
        num_refs_resolved=len(solved_reference),
        max_reference_resolve_rmse=max(solved_reference),
    )
    for name, values in fields.items():
        projected = TO_CENTERS @ interior[name] @ TO_CENTERS.T
        checks[f"{name}_boundary_projection_rmse"] = rms(projected - values)
    if max(checks["truth_resolve_rmse"], checks["max_reference_resolve_rmse"]) > 1e-8:
        raise ValueError(f"Stored pressure does not match the released Darcy solver for {case_id}")
    return rows, checks


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_pressure(rows):
    return {
        name: {
            region: {
                metric: float(np.mean([row[metric] for row in rows if row["ensemble"] == name and row["region"] == region]))
                for metric in rows[0] if metric not in ("case_id", "ensemble", "region")
            }
            for region in ("all", "observed", "held_out")
        }
        for name in FIELDS
    }


def paired_bootstrap(rows):
    """Intervals for mean changes by resampling whole test cases."""
    old = {row["case_id"]: row for row in rows if row["ensemble"] == "internal" and row["region"] == "held_out"}
    new = {row["case_id"]: row for row in rows if row["ensemble"] == "numerical" and row["region"] == "held_out"}
    rng = np.random.default_rng(0)
    samples = rng.integers(len(old), size=(10000, len(old)))
    output = {}
    for key in ("reference_w1", "truth_mean_rmse", "truth_crps", "truth_90_coverage"):
        changes = np.array([new[case][key] - old[case][key] for case in old])
        output[key] = {
            "mean_change_numerical_minus_internal": float(changes.mean()),
            "case_bootstrap_percentile_95": np.quantile(changes[samples].mean(axis=1), [0.025, 0.975]).tolist(),
            "num_cases_lower_for_numerical": int((changes < 0).sum()),
        }
    return output


def main():
    baseline = json.loads((BASELINE / "run_settings.json").read_text(encoding="utf-8"))
    dataset = get_dataset("darcy")
    refs = [ref for ref in dataset.discover(DATA) if ref.split == "test"]
    if [ref.stem for ref in refs] != baseline["case_ids"]:
        raise ValueError("Test split does not match the baseline")
    metadata = json.loads((DATA / "PDEFieldDataset_hf/Darcy/metadata.json").read_text())
    pressure_scale = metadata["stats"]["std"][metadata["field_names"].index("u")]
    rng = np.random.default_rng(0)
    directions = np.stack([sample_grf_projection(SIZE, SIZE, rng).ravel() for _ in range(128)])

    pressure, conservation, paired, controls = [], [], [], []
    for i, ref in enumerate(refs, 1):
        case = load_case(dataset, ref)
        predictive_rows, paired_row = pressure_metrics(ref.stem, case, directions, pressure_scale)
        flow_rows, control_row = conservation_metrics(ref.stem, case)
        pressure.extend(predictive_rows)
        conservation.extend(flow_rows)
        paired.append(paired_row)
        controls.append(control_row)
        if i == 1 or i % 5 == 0:
            print(f"Analyzed {i}/{len(refs)} cases ({ref.stem}).", flush=True)

    for filename, rows in (
        ("pressure_by_case.csv", pressure), ("conservation_by_case.csv", conservation),
        ("paired_pressure_by_case.csv", paired), ("solver_controls_by_case.csv", controls),
    ):
        write_csv(HERE / filename, rows)

    summary = dict(
        cases=len(refs), draws_per_case=SAMPLES_PER_CASE, pressure_swd_normalizer=pressure_scale,
        pressure=summarize_pressure(pressure), held_out_case_bootstrap=paired_bootstrap(pressure),
        conservation={
            name: {
                str(side): float(np.mean([
                    row["mean_member_rms_fraction"] for row in conservation
                    if row["pressure_and_permeability"] == name and row["patch_side"] == side
                ]))
                for side in PATCH_SIDES
            }
            for name in FLUX_PAIRS
        },
        paired_pressure={
            key: float(np.mean([row[key] for row in paired])) for key in paired[0] if key != "case_id"
        },
        solver_controls=dict(
            num_hidden_truth_resolved=len(controls),
            num_reference_members_resolved=sum(row["num_refs_resolved"] for row in controls),
            max_truth_resolve_rmse=max(row["truth_resolve_rmse"] for row in controls),
            max_reference_resolve_rmse=max(row["max_reference_resolve_rmse"] for row in controls),
            mean_pressure_representation_rmse={
                name: float(np.mean([row[f"{name}_boundary_projection_rmse"] for row in controls]))
                for name in FIELDS
            },
        ),
    )
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Saved pressure, conservation, and solver-control results.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
