#!/usr/bin/env python3
"""Download the SimTK IMU dataset (study20-latest.tar.gz, ~5.5GB)."""

import os
import tarfile
import requests
from tqdm import tqdm

_URL = "https://datashare.simtk.org/apps/browse/download/sendRelease.php"
_FILENAME = "study20-latest.tar.gz"
_PARAMS = {
    "groupid": "2164",
    "userid": "0",
    "studyid": "20",
    "token": "$2y$10$LmcsiN6v44CdqRbgKUR7ze4OVHr9/gSCep4RZABfBMXAumMTr4o6S",
}


def download_and_extract_simtk_dataset():
    """Download, extract, and clean up the SimTK IMU dataset (~5.5GB)."""
    with requests.get(_URL, params=_PARAMS, stream=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(_FILENAME, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Extracting {_FILENAME}...")
    with tarfile.open(_FILENAME, "r:gz") as tar:
        tar.extractall(filter="data")
    os.rename("files", "data")
    os.remove(_FILENAME)


if __name__ == "__main__":
    download_and_extract_simtk_dataset()
