"""Fetch and verify the pinned public LTMI prior and posterior datasets."""

import os
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial as bind
from itertools import batched
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = Path(os.environ.get("POSTERIORBENCH_DATA", str(PROJECT / "data")))
REVISION = "f45719309a0ea6500b86e6e9973c18b3ffff8650"
REPO = "https://huggingface.co/datasets/jcy20/PosteriorBench"
API = "https://huggingface.co/api/datasets/jcy20/PosteriorBench"
FOLDERS = ("PDEFieldDataset_hf/LTMI", "PosteriorDataset_hf/LTMI")
CHUNK = 8 * 2**20


def digest(path, item):
    with path.open("rb") as stream:
        if "lfs" in item:
            return hashlib.file_digest(stream, "sha256").hexdigest()
        git_blob = hashlib.sha1(f"blob {item['size']}\0".encode())
        return hashlib.file_digest(stream, lambda: git_blob).hexdigest()


def fetch_range(item, start):
    size = item["size"]
    end = min(start + CHUNK, size) - 1
    url = f"{REPO}/resolve/{REVISION}/{item['path']}"
    headers = {"Range": f"bytes={start}-{end}"} if "lfs" in item else {}
    for attempt in range(5):
        params = {"download": "true", "ltmi_range": f"{start}-{end}"}
        if attempt:
            params["retry"] = time.time_ns()
        try:
            response = requests.get(url, params=params, headers=headers, timeout=(12, 35))
            response.raise_for_status()
            if headers and response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                raise OSError(f"Unexpected response range for {item['path']} at {start}")
            if len(response.content) != end - start + 1:
                raise OSError(f"Incomplete response range for {item['path']} at {start}")
            return response.content
        except (requests.RequestException, OSError):
            if attempt == 4:
                raise
            time.sleep(attempt + 1)


def fetch(item, download_pool):
    path = DATA / item["path"]
    size = item["size"]
    expected = item["lfs"]["oid"] if "lfs" in item else item["oid"]
    if path.is_file() and path.stat().st_size == size and digest(path, item) == expected:
        print(f"Verified existing {item['path']}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("ab", buffering=0) as stream:
        if stream.tell() > size:
            raise OSError(f"Partial download exceeds the released size: {partial}")
        offsets = range(stream.tell(), size, CHUNK)
        for batch in batched(offsets, 8):
            for data in download_pool.map(bind(fetch_range, item), batch):
                stream.write(data)
    if digest(partial, item) != expected:
        raise OSError(f"Checksum mismatch for {partial}")
    partial.replace(path)
    print(f"Downloaded and verified {item['path']}", flush=True)


def main():
    files = []
    for folder in FOLDERS:
        response = requests.get(f"{API}/tree/{REVISION}/{folder}", params={"recursive": "true"}, timeout=20)
        response.raise_for_status()
        files.extend(item for item in response.json() if item["type"] == "file")
    (HERE / "data_manifest.json").write_text(
        json.dumps({"revision": REVISION, "files": files}, indent=2) + "\n", encoding="utf-8"
    )
    # Make the evaluation cases available while the larger prior pool downloads.
    files.sort(key=lambda item: (item["size"] > 10**6, item["path"].startswith(FOLDERS[0])))
    with ThreadPoolExecutor(18) as download_pool, ThreadPoolExecutor(8) as file_pool:
        list(file_pool.map(bind(fetch, download_pool=download_pool), files))


if __name__ == "__main__":
    main()
