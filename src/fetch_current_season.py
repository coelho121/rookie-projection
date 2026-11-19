"""Download what's needed to project the current rookie class.

The current season (2026-27) is still being played, so its game logs change
every day. Each download is cached with its date in the file name: running the
script twice on the same day reuses the file; on a new day it downloads a fresh
snapshot. Older snapshots are kept, so any projection can be reproduced.

Player info (birth date, position) for the current draft class comes from the
nba_api CommonPlayerInfo endpoint, one cached file per player (never
re-downloaded): Basketball-Reference's season table doesn't have the newest
rookies yet.

Outputs:
    data/raw/gamelogs_current/player_2026-27_asof_YYYY-MM-DD.csv
    data/raw/gamelogs_current/team_2026-27_asof_YYYY-MM-DD.csv
    data/raw/player_info/<player_id>.csv

Run from the project root:
    python src/fetch_current_season.py
"""

import time
from datetime import date
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, leaguegamelog

ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = ROOT / "data" / "raw" / "gamelogs_current"
INFO_DIR = ROOT / "data" / "raw" / "player_info"
DRAFT_RAW = ROOT / "data" / "raw" / "draft_history.csv"

CURRENT_SEASON = "2026-27"
CURRENT_CLASS = 2026
MODES = {"player": "P", "team": "T"}

TIMEOUT_S = 60
MAX_TRIES = 4
PAUSE_S = 2


def with_retries(call):
    """Run an nba_api call, retrying with backoff on failure."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            return call()
        except Exception as exc:  # stats.nba.com fails with many error types
            if attempt == MAX_TRIES:
                raise
            wait = 5 * 2 ** (attempt - 1)
            print(f"  attempt {attempt} failed ({exc!r}); retrying in {wait}s")
            time.sleep(wait)


def snapshot_path(mode: str, as_of: str) -> Path:
    return CURRENT_DIR / f"{mode}_{CURRENT_SEASON}_asof_{as_of}.csv"


def latest_snapshot(mode: str) -> Path:
    """Most recent cached snapshot for a mode (used by the build step)."""
    files = sorted(CURRENT_DIR.glob(f"{mode}_{CURRENT_SEASON}_asof_*.csv"))
    if not files:
        raise FileNotFoundError("No current-season snapshot: run src/fetch_current_season.py")
    return files[-1]


def fetch_game_logs(as_of: str) -> None:
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    for mode, code in MODES.items():
        path = snapshot_path(mode, as_of)
        if path.exists():
            print(f"{mode} {CURRENT_SEASON} as of {as_of}: cached")
            continue
        df = with_retries(lambda: leaguegamelog.LeagueGameLog(
            season=CURRENT_SEASON, season_type_all_star="Regular Season",
            player_or_team_abbreviation=code, timeout=TIMEOUT_S).get_data_frames()[0])
        df.to_csv(path, index=False)
        print(f"{mode} {CURRENT_SEASON} as of {as_of}: {len(df)} rows "
              f"(games up to {df['GAME_DATE'].max()})")
        time.sleep(PAUSE_S)


def fetch_player_info() -> None:
    """Birth date and position of every player drafted in the current class."""
    INFO_DIR.mkdir(parents=True, exist_ok=True)
    draft = pd.read_csv(DRAFT_RAW)
    ids = draft.loc[draft["SEASON"] == CURRENT_CLASS, "PERSON_ID"].tolist()
    new = [pid for pid in ids if not (INFO_DIR / f"{pid}.csv").exists()]
    print(f"Player info: {len(ids) - len(new)} cached, {len(new)} to download")
    for pid in new:
        info = with_retries(lambda: commonplayerinfo.CommonPlayerInfo(
            player_id=pid, timeout=TIMEOUT_S).get_data_frames()[0])
        info.to_csv(INFO_DIR / f"{pid}.csv", index=False)
        time.sleep(PAUSE_S)


def main() -> None:
    fetch_game_logs(date.today().isoformat())
    fetch_player_info()


if __name__ == "__main__":
    main()
