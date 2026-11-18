"""Download Basketball-Reference season data from the public Kaggle mirror.

basketball-reference.com blocks automated clients (Cloudflare challenge), so we
use the Kaggle dataset "NBA Stats (1947-present)" by Sumitro Datta (CC0), which
is scraped from Basketball-Reference and updated every season. Its player ids
are the Basketball-Reference slugs (e.g. 'jamesle01').

Validated against Basketball-Reference values (e.g. LeBron 2008-09: BPM 13.2,
VORP 11.8) and against our nba_api game logs (games match 99.9%).

The dataset keeps changing as seasons are added, so the cached copy freezes the
version used in this project. Delete the files to refresh them.

Outputs:
    data/raw/bbref/Advanced.csv             one row per player-season-team
    data/raw/bbref/Draft Pick History.csv   one row per draft pick

Run from the project root:
    python src/fetch_bbref.py
"""

import time
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "bbref"

DATASET = "sumitrodatta/nba-aba-baa-stats"
FILES = ["Advanced.csv", "Draft Pick History.csv"]
URL = "https://www.kaggle.com/api/v1/datasets/download/{dataset}/{file}"

TIMEOUT_S = 120
MAX_TRIES = 4


def download(file: str) -> bytes:
    """Download one file from the Kaggle dataset, retrying on failure."""
    url = URL.format(dataset=DATASET, file=quote(file))
    for attempt in range(1, MAX_TRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT_S)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            if attempt == MAX_TRIES:
                raise
            wait = 5 * 2 ** (attempt - 1)  # 5s, 10s, 20s
            print(f"  attempt {attempt} failed ({exc!r}); retrying in {wait}s")
            time.sleep(wait)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for file in FILES:
        path = RAW_DIR / file
        if path.exists():
            print(f"{file}: cached")
            continue
        path.write_bytes(download(file))
        print(f"{file}: downloaded ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
