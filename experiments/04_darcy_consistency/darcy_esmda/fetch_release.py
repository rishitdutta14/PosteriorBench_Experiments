"""Download and verify both datasets in the pinned Darcy release."""

import os
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DATA = Path(os.environ.get("POSTERIORBENCH_DATA", str(PROJECT / "data")))
REPO_ID = "jcy20/PosteriorBench"
REVISION = "f45719309a0ea6500b86e6e9973c18b3ffff8650"
API = f"https://huggingface.co/api/datasets/{REPO_ID}"
REPO = f"https://huggingface.co/datasets/{REPO_ID}"
FOLDERS = ("PosteriorDataset_hf/Darcy", "PDEFieldDataset_hf/Darcy")
RANGE_SIZE = 32 * 1024**2


def check_digest(path, item):
    with path.open("rb") as stream:
        if "lfs" in item:
            return hashlib.file_digest(stream, "sha256").hexdigest() == item["lfs"]["oid"]
        sha1 = hashlib.sha1(f"blob {item['size']}\0".encode())
        return hashlib.file_digest(stream, lambda: sha1).hexdigest() == item["oid"]


def fetch(item):
    relative = item["path"]
    path = DATA / relative
    size = item["size"]
    if path.is_file() and path.stat().st_size == size and check_digest(path, item):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    url = f"{REPO}/resolve/{REVISION}/{relative}"
    with requests.Session() as session, partial.open("ab", buffering=0) as stream:
        if stream.tell() > size:
            raise ValueError(f"Oversized partial download: {partial}")
        failures = 0
        while stream.tell() < size:
            start = stream.tell()
            end = min(start + RANGE_SIZE, size) - 1
            headers = {"Range": f"bytes={start}-{end}"} if "lfs" in item else {}
            params = {"download": "true", "darcy_range": f"{start}_{end}"}
            if failures:
                params["retry"] = time.time_ns()
            try:
                with session.get(url, params=params, headers=headers, timeout=(15, 25), stream=True) as response:
                    response.raise_for_status()
                    if "lfs" in item and response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise OSError("Unexpected response range")
                    if "lfs" not in item and start:
                        stream.truncate(0)
                        stream.seek(0)
                    for block in response.iter_content(1024**2):
                        stream.write(block)
                    if stream.tell() != end + 1:
                        raise OSError("Incomplete response")
                failures = 0
            except (requests.RequestException, OSError) as error:
                failures = failures + 1 if stream.tell() == start else 0
                print(f"Retry {relative} at {stream.tell():,} bytes ({type(error).__name__})", flush=True)
                if failures >= 7:
                    raise
                time.sleep(min(failures, 5))
    if not check_digest(partial, item):
        raise OSError(f"Checksum mismatch: {partial}")
    partial.replace(path)
    print(f"Downloaded and verified {relative}", flush=True)


def main():
    files = []
    for folder in FOLDERS:
        response = requests.get(f"{API}/tree/{REVISION}/{folder}", params={"recursive": "true"}, timeout=30)
        response.raise_for_status()
        files.extend(item for item in response.json() if item["type"] == "file")
    (HERE / "data_manifest.json").write_text(
        json.dumps({"revision": REVISION, "files": files}, indent=2) + "\n", encoding="utf-8"
    )
    files.sort(key=lambda item: (item["size"] > 10**6, not item["path"].startswith(FOLDERS[0]), item["path"]))
    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(fetch, files))
    print(f"Verified all {len(files)} released Darcy files.", flush=True)


if __name__ == "__main__":
    main()
