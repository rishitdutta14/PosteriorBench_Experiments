"""Run the released Poisson ES-MDA adapter and scorer on the held-out cases."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
REPO = PROJECT / "vendor" / "PosteriorBench"
DATA = Path(os.environ.get("POSTERIORBENCH_DATA", str(PROJECT / "data")))
PROFILE = REPO / "configs/evaluations/esmda_poisson.yaml"
SHARD = DATA / "PDEFieldDataset_hf/Poisson/data-00000-of-00014.arrow"
SHARD_HASH = "1cc3079b780150e67f626dd949f9797d831199791c1bd95e26e6d43d0135187f"
sys.path.insert(0, str(REPO))
from datasets import Dataset
from posteriorbench.artifacts import load_checkpoint_profile
from posteriorbench.datasets import get_dataset
from posteriorbench.evaluate import main as evaluate
from posteriorbench.io import write_generated_posterior, write_json
from posteriorbench.methods.esmda import ESMDAAdapter


def materialize_first_shard():
    """Reuse the existing verified HF Arrow shard in a valid save_to_disk layout."""
    with SHARD.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != SHARD_HASH:
            raise ValueError("Training shard checksum mismatch")
    target = HERE / "training_shard"
    dataset = Dataset.from_file(str(SHARD))
    if not (target / "state.json").exists():
        dataset.save_to_disk(str(target))
    metadata = json.loads((SHARD.parent / "metadata.json").read_text())
    metadata["num_samples"] = len(dataset)
    metadata_path = target / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-source", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output", type=Path, default=HERE / "run")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.training_source.resolve() if args.training_source else materialize_first_shard()
    dataset = get_dataset("poisson")
    refs = [(i, ref) for i, ref in enumerate(dataset.discover(DATA)) if ref.split == "test"][:args.max_cases]
    if not refs:
        raise RuntimeError("No test cases.")
    profile = load_checkpoint_profile(PROFILE)
    method = ESMDAAdapter(checkpoint, profile, dataset, "cpu")
    method.load()
    case_ids = [ref.stem for _, ref in refs]
    meta = dict(
        benchmark_commit="ab416f5b322717f5668de01889ae0141dbc00af6",
        data_revision="f45719309a0ea6500b86e6e9973c18b3ffff8650",
        profile=str(PROFILE), observation_error=profile.guidance["observation_error"],
        training_source=str(checkpoint), training_rows=method.num_prior_samples,
        training_release_rows=50000, seed="released default seed (0) plus full discovery index",
        case_ids=case_ids, samples_per_case=100, device="cpu",
    )
    settings_path = output / "run_settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != meta:
        raise ValueError("This output directory belongs to different run settings.")
    write_json(settings_path, meta)
    print(f"Loaded {method.num_prior_samples} prior fields. Generating {len(refs)} test cases.", flush=True)
    for i, (seed, ref) in enumerate(refs, 1):
        output_path = output / "predictions" / ref.stem / "generated.h5"
        if not output_path.exists():
            case = dataset.load_case(ref)
            samples = method.generate(case, num_samples=100, batch_size=10, seed=seed)
            if samples["f"].shape != (100, 128, 128) or not np.isfinite(samples["f"]).all():
                raise ValueError(f"Unexpected samples for {case.case_id}")
            tmp = output_path.with_suffix(".h5.part")
            write_generated_posterior(tmp, samples, case, "esmda", checkpoint, PROFILE, seed)
            tmp.replace(output_path)
        if i % 10 == 0 or i == len(refs):
            print(f"Generated {i}/{len(refs)}", flush=True)

    sys.argv = ["posteriorbench.evaluate", "--dataset", "poisson", "--case-source", str(DATA),
                "--predictions", str(output / "predictions"), "--output", str(output / "evaluation"),
                "--training-source", str(checkpoint)]
    for case in case_ids:
        sys.argv.extend(["--case-id", case])
    evaluate()


if __name__ == "__main__":
    with threadpool_limits(3):
        main()
