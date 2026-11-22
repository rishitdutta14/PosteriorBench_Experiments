"""Forward-solve every published-setting ES-MDA permeability map."""

import argparse
import os
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from darcy_grid import SIZE, face_coefficients, solve

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "darcy_esmda" / "run"
OUTPUT = HERE / "numerical_pressures"
SAMPLES_PER_CASE = 100
SCHEMA = "darcy.pressure.released-stencil.v1"


def solve_case(case_id):
    input_path = BASELINE / "predictions" / case_id / "generated.h5"
    with h5py.File(input_path, "r") as source:
        permeability = source["a"][:]
    expected_shape = (SAMPLES_PER_CASE, SIZE, SIZE)
    if permeability.shape != expected_shape or not np.isin(permeability, [3.0, 12.0]).all():
        raise ValueError(f"Invalid rounded permeability for {case_id}")
    digest = hashlib.sha256(permeability.tobytes()).hexdigest()

    output_path = OUTPUT / f"{case_id}.h5"
    if output_path.exists():
        with h5py.File(output_path, "r") as saved:
            matching = (
                saved.attrs["schema"] == SCHEMA
                and saved.attrs["permeability_sha256"] == digest
                and saved["u_num"].shape == expected_shape
                and np.isfinite(saved["u_num"][:]).all()
            )
        if not matching:
            raise ValueError(f"Existing numerical maps do not match the input for {case_id}")
        return case_id

    fields = np.empty(expected_shape, dtype=np.float32)
    for i, a in enumerate(permeability):
        fields[i] = solve(face_coefficients(a))
    partial = output_path.with_suffix(".h5.part")
    with h5py.File(partial, "w") as saved:
        saved.attrs.update(
            schema=SCHEMA, case_id=case_id, permeability_sha256=digest,
            source_permeability="a, after the benchmark binary projection",
        )
        saved.create_dataset("u_num", data=fields, compression="lzf", shuffle=True, chunks=(1, SIZE, SIZE))
    partial.replace(output_path)
    return case_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    settings = json.loads((BASELINE / "run_settings.json").read_text(encoding="utf-8"))
    case_ids = settings["case_ids"]
    if len(case_ids) != 90 or settings["samples_per_case"] != SAMPLES_PER_CASE:
        raise ValueError("Expected the full official-setting test run")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, case_id in enumerate(pool.map(solve_case, case_ids), 1):
            if i == 1 or i % 10 == 0:
                print(f"Solved or verified {i}/{len(case_ids)} cases ({case_id}).", flush=True)


if __name__ == "__main__":
    main()
