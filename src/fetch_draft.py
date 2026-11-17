"""Download NBA draft history and build the 2000-2022 draft classes table.

Source: stats.nba.com DraftHistory endpoint (via nba_api). One request returns
every draft since 1947, so we cache that raw response once and filter locally.

Outputs:
    data/raw/draft_history.csv        full raw response (never re-downloaded)
    data/processed/draft_classes.csv  2000-2022 picks, cleaned

Run from the project root:
    python src/fetch_draft.py
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import drafthistory

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "draft_history.csv"
OUT_PATH = ROOT / "data" / "processed" / "draft_classes.csv"

FIRST_DRAFT = 2000
LAST_DRAFT = 2022  # last class with a completed year 4

TIMEOUT_S = 60
MAX_TRIES = 4


def download_draft_history() -> pd.DataFrame:
    """Call the DraftHistory endpoint, retrying with backoff on failure."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            resp = drafthistory.DraftHistory(league_id="00", timeout=TIMEOUT_S)
            return resp.get_data_frames()[0]
        except Exception as exc:  # stats.nba.com fails with many error types
            if attempt == MAX_TRIES:
                raise
            wait = 5 * 2 ** (attempt - 1)  # 5s, 10s, 20s
            print(f"Attempt {attempt} failed ({exc!r}); retrying in {wait}s")
            time.sleep(wait)


def load_raw() -> pd.DataFrame:
    """Return the raw draft history, downloading it only if not cached."""
    if RAW_PATH.exists():
        print(f"Using cached {RAW_PATH.relative_to(ROOT)}")
    else:
        print("Downloading draft history from stats.nba.com...")
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        download_draft_history().to_csv(RAW_PATH, index=False)
    return pd.read_csv(RAW_PATH)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the 2000-2022 classes and the columns we need, in snake_case."""
    df = raw.rename(
        columns={
            "PERSON_ID": "player_id",
            "PLAYER_NAME": "player_name",
            "SEASON": "draft_year",
            "ROUND_NUMBER": "round",
            "ROUND_PICK": "round_pick",
            "OVERALL_PICK": "pick",
            "TEAM_ABBREVIATION": "team",
            "ORGANIZATION": "pre_draft_team",
            "ORGANIZATION_TYPE": "pre_draft_type",
        }
    )
    df = df[df["draft_year"].between(FIRST_DRAFT, LAST_DRAFT)]
    cols = [
        "player_id", "player_name", "draft_year", "round", "round_pick",
        "pick", "team", "pre_draft_team", "pre_draft_type",
    ]
    return df[cols].sort_values(["draft_year", "pick"]).reset_index(drop=True)


def check(df: pd.DataFrame) -> None:
    """Sanity checks: fail loudly if the data doesn't look like we expect."""
    assert df["player_id"].is_unique, "a player appears in more than one draft"
    assert not df.duplicated(["draft_year", "pick"]).any(), "duplicate pick"
    years = set(df["draft_year"])
    assert years == set(range(FIRST_DRAFT, LAST_DRAFT + 1)), "missing years"


def main() -> None:
    draft = clean(load_raw())
    check(draft)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    draft.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(draft)} picks to {OUT_PATH.relative_to(ROOT)}")
    print("\nPicks per class:")
    print(draft.groupby("draft_year").size().to_string())


if __name__ == "__main__":
    main()
