"""Reproduce the released Darcy ES-MDA run and keep paired internal fields."""

import argparse
import os
import csv
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = HERE.parent
REPO = PROJECT / "vendor" / "PosteriorBench"
DATA = Path(os.environ.get("POSTERIORBENCH_DATA", str(PROJECT / "data")))
PRIOR = DATA / "PDEFieldDataset_hf" / "Darcy"
PROFILE = REPO / "configs" / "evaluations" / "esmda_darcy.yaml"
NUM_SAMPLES = 100
BATCH_SIZE = 10
SHAPE = (NUM_SAMPLES, 128, 128)
PAPER = {
    "mean_rel_l2": 0.4519,
    "std_rel_l2": 1.7547,
    "mmd": 0.5812,
    "swd": 48.4836,
    "spectral_rel_geomean": 0.3481,
}

sys.path.insert(0, str(REPO))
from posteriorbench.artifacts import load_checkpoint_profile
from posteriorbench.datasets import get_dataset
from posteriorbench.io import write_generated_posterior, write_json
from posteriorbench.methods.esmda import ESMDAAdapter


class DarcyESMDARecorder(ESMDAAdapter):
    """Read the joint state at the end of the released assimilation."""

    def _assimilate(self, state, observed_vector, case, rng):
        updated = super()._assimilate(state, observed_vector, case, rng)
        a_channel = self.profile.channels.index("a")
        u_channel = self.profile.channels.index("u")
        self.raw_permeability = updated[:NUM_SAMPLES, a_channel].copy()
        self.internal_pressure = updated[:NUM_SAMPLES, u_channel].copy()
        return updated


def get_cases(dataset):
    cases = [ref for ref in dataset.discover(DATA) if ref.split == "test"]
    if len(cases) != 90:
        raise ValueError(f"Expected 90 Darcy test cases, got {len(cases)}")
    return cases


def validate_saved_case(path, seed):
    with h5py.File(path, "r") as handle:
        if int(handle.attrs["seed"]) != seed:
            raise ValueError(f"Unexpected seed in {path}")
        for field in ("a", "a_raw", "u"):
            if handle[field].shape != SHAPE or handle[field].dtype != np.float32:
                raise ValueError(f"Unexpected {field} shape or type in {path}")
        np.testing.assert_array_equal(
            handle["a"][:], np.where(handle["a_raw"][:] > 7.5, 12.0, 3.0)
        )
        if not np.isfinite(handle["u"][:]).all():
            raise ValueError(f"Nonfinite internal pressure in {path}")


def generate(output, dataset, cases):
    profile = load_checkpoint_profile(PROFILE)
    adapter = DarcyESMDARecorder(PRIOR, profile, dataset, "cpu")
    adapter.load()
    if adapter.num_prior_samples != 50_000:
        raise ValueError(f"Expected full release of 50,000 prior rows, got {adapter.num_prior_samples}")

    manifest = json.loads((PROJECT / "results/04_darcy_consistency/generation/data_manifest.json").read_text(encoding="utf-8"))
    settings = {
        "benchmark_commit": "ab416f5b322717f5668de01889ae0141dbc00af6",
        "data_revision": manifest["revision"],
        "case_split": "test",
        "case_ids": [ref.stem for ref in cases],
        "prior_rows": adapter.num_prior_samples,
        "training_source": str(PRIOR),
        "device": "cpu",
        "numpy_version": np.__version__,
        "samples_per_case": NUM_SAMPLES,
        "batch_size": BATCH_SIZE,
        "seed_rule": "0 plus the index in the filtered test case list, as in released CLI",
        "sampling": profile.sampling,
        "inflation_factors": adapter._alphas(),
        "observation_error": profile.guidance["observation_error"],
        "observation_operator": profile.guidance["observation_operator"],
        "postprocess": {"a": [3.0, 12.0], "threshold": 7.5},
    }
    path = output / "run_settings.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Output directory belongs to a run with different settings")
    write_json(path, settings)
    print(f"Loaded {adapter.num_prior_samples:,} released prior pairs.", flush=True)

    for seed, ref in enumerate(cases):
        path = output / "predictions" / ref.stem / "generated.h5"
        if path.exists():
            validate_saved_case(path, seed)
            continue
        case = dataset.load_case(ref)
        if case.sensor_coords is None or case.sensor_coords.shape != (128, 2):
            raise ValueError(f"Unexpected sensor coordinates for {case.case_id}")
        samples = adapter.generate(case, NUM_SAMPLES, BATCH_SIZE, seed)
        for field in (samples["a"], adapter.raw_permeability, adapter.internal_pressure):
            if field.shape != SHAPE or not np.isfinite(field).all():
                raise ValueError(f"Unexpected generated fields for {case.case_id}")
        if seed == 0:
            baseline = ESMDAAdapter(PRIOR, profile, dataset, "cpu")
            baseline.load()
            original = baseline.generate(case, NUM_SAMPLES, BATCH_SIZE, seed)
            np.testing.assert_array_equal(samples["a"], original["a"])
            print("Capture check passed: first case exactly matches the uninstrumented adapter.", flush=True)

        partial = path.with_suffix(".h5.part")
        write_generated_posterior(partial, samples, case, "esmda", PRIOR, PROFILE, seed)
        with h5py.File(partial, "a") as handle:
            for name, values, description in (
                ("u", adapter.internal_pressure, "Internal pressure after the second update"),
                ("a_raw", adapter.raw_permeability, "Permeability before binary postprocessing"),
            ):
                field = handle.create_dataset(
                    name, data=values, compression="gzip", chunks=(1, 128, 128)
                )
                field.attrs["description"] = description
                field.attrs["pairing"] = "Same member order as a"
        validate_saved_case(partial, seed)
        partial.replace(path)
        if seed == 0 or (seed + 1) % 10 == 0 or seed + 1 == len(cases):
            print(f"Saved {seed + 1}/{len(cases)} paired Darcy ensembles.", flush=True)


def evaluate(output, cases):
    evaluator_args = [
        "posteriorbench.evaluate",
        "--dataset", "darcy", "--case-source", str(DATA),
        "--predictions", str(output / "predictions"),
        "--training-source", str(PRIOR), "--output", str(output / "evaluation"),
    ]
    for ref in cases:
        evaluator_args.extend(("--case-id", ref.stem))
    # Run the unmodified evaluator inside this process to retain the BLAS thread limit.
    from posteriorbench.evaluate import main as evaluate_main

    original_argv = sys.argv
    try:
        sys.argv = evaluator_args
        evaluate_main()
    finally:
        sys.argv = original_argv
    metrics = json.loads((output / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    if metrics["num_cases"] != len(cases):
        raise ValueError("Evaluator case count disagrees with the generation worklist")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "paper_table_2", "local_mean", "local_std", "local_minus_paper"])
        for name, paper_value in PAPER.items():
            result = metrics["summary"]["a"][name]
            writer.writerow([name, paper_value, result["mean"], result["std"], result["mean"] - paper_value])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "run")
    parser.add_argument("--stage", choices=("all", "generate", "evaluate"), default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    dataset = get_dataset("darcy")
    cases = get_cases(dataset)
    if args.stage in ("all", "generate"):
        generate(output, dataset, cases)
    if args.stage in ("all", "evaluate"):
        evaluate(output, cases)


if __name__ == "__main__":
    with threadpool_limits(limits=4, user_api="blas"):
        main()
