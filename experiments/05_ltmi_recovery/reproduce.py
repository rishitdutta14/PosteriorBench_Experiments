"""Run the released LTMI ES-MDA adapter and evaluator on all 90 test cases."""

import argparse
import os
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPO = PROJECT / "vendor" / "PosteriorBench"
DATA = Path(os.environ.get("POSTERIORBENCH_DATA", str(PROJECT / "data")))
PRIOR = DATA / "PDEFieldDataset_hf" / "LTMI"
PROFILE = REPO / "configs" / "evaluations" / "esmda_light_transport.yaml"
NUM_OUTPUTS = 100
BATCH_SIZE = 10

sys.path.insert(0, str(REPO))
from posteriorbench.artifacts import load_checkpoint_profile
from posteriorbench.datasets import get_dataset
from posteriorbench.evaluate import main as evaluate
from posteriorbench.io import write_generated_posterior, write_json
from posteriorbench.methods.esmda import ESMDAAdapter


class ESMDAWithLightResponse(ESMDAAdapter):
    """Keep the internal response for the same draws the released adapter returns."""

    def _assimilate(self, state, observed_vector, case, rng):
        updated = super()._assimilate(state, observed_vector, case, rng)
        observed_channel = self.profile.channels.index(case.observed_field)
        self.internal_light = updated[:NUM_OUTPUTS, observed_channel].copy()
        return updated


def generate(args):
    dataset = get_dataset("light_transport")
    cases = [ref for ref in dataset.discover(DATA) if ref.split == "test"]
    if len(cases) != 90:
        raise ValueError(f"Expected 90 LTMI test cases, got {len(cases)}")
    case_ids = [ref.stem for ref in cases]
    settings_path = HERE / "run_settings.json"
    if args.stage in ("all", "generate"):
        profile = load_checkpoint_profile(PROFILE)
        adapter = ESMDAWithLightResponse(PRIOR, profile, dataset, "cpu")
        adapter.load()
        if adapter.num_prior_samples != 100_000:
            raise ValueError(f"Expected 100,000 prior fields, got {adapter.num_prior_samples}")
        manifest = json.loads((PROJECT / "results/05_ltmi_recovery/data_manifest.json").read_text(encoding="utf-8"))
        settings = {
            "benchmark_commit": "ab416f5b322717f5668de01889ae0141dbc00af6",
            "dataset_revision": manifest["revision"],
            "case_split": "test",
            "case_ids": case_ids,
            "sampling": profile.sampling,
            "observation_error": profile.guidance["observation_error"],
            "observation_operator": profile.guidance["observation_operator"],
            "prior_rows": adapter.num_prior_samples,
            "samples_per_case": NUM_OUTPUTS,
            "saved_internal_field": "u from the final joint state for each scored material pair",
            "batch_size": BATCH_SIZE,
            "seed_rule": "0 plus the index in the filtered test case list (released CLI convention)",
            "device": "cpu",
        }
        if settings_path.exists():
            recorded = json.loads(settings_path.read_text(encoding="utf-8"))
            if recorded != settings:
                raise ValueError("Existing predictions have different recorded run settings")
        else:
            write_json(settings_path, settings)

        for seed, ref in enumerate(cases):
            case = dataset.load_case(ref)
            path = HERE / "predictions" / case.case_id / "generated.h5"
            if path.is_file():
                continue
            started = time.perf_counter()
            samples = adapter.generate(case, NUM_OUTPUTS, BATCH_SIZE, seed)
            if set(samples) != set(case.target_fields):
                raise ValueError(f"Unexpected ES-MDA fields for {case.case_id}")
            samples[case.observed_field] = adapter.internal_light
            if any(
                values.shape != (NUM_OUTPUTS, 64, 64) or not np.isfinite(values).all()
                for values in samples.values()
            ):
                raise ValueError(f"Unexpected ES-MDA samples for {case.case_id}")
            partial = path.with_suffix(".h5.part")
            write_generated_posterior(partial, samples, case, "esmda", PRIOR, PROFILE, seed)
            partial.replace(path)
            print(f"Generated {seed + 1}/{len(cases)} {case.case_id} in {time.perf_counter() - started:.1f}s", flush=True)

    if args.stage in ("all", "evaluate"):
        sys.argv = [
            "posteriorbench.evaluate", "--dataset", "light_transport",
            "--case-source", str(DATA), "--predictions", str(HERE / "predictions"),
            "--output", str(HERE / "evaluation"), "--training-source", str(PRIOR),
        ]
        for case_id in case_ids:
            sys.argv.extend(("--case-id", case_id))
        evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "generate", "evaluate"), default="all")
    with threadpool_limits(limits=4, user_api="blas"):
        generate(parser.parse_args())
