"""Download regular-season game logs (player and team) for every season we need.

Source: stats.nba.com LeagueGameLog endpoint (via nba_api). One request returns
every game of one season, so we need ~2 requests per season instead of one
request per player.

Seasons: 1999-00 to 2025-26 (last completed season). 1999-00 is only needed
for the previous-season team rating of 2000-01 rookies; 2025-26 covers late
debuts from the 2022 class. The 2026-27 season is still in progress and is
not downloaded.

Outputs (one file per season and mode, never re-downloaded):
    data/raw/gamelogs/player_2000-01.csv
    data/raw/gamelogs/team_2000-01.csv
    ...

Run from the project root:
    python src/fetch_gamelogs.py
"""

import time
from pathlib import Path

from nba_api.stats.endpoints import leaguegamelog

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "gamelogs"

FIRST_SEASON = 1999  # start year of the 1999-00 season (previous-season team rating)
LAST_SEASON = 2025   # start year of the 2025-26 season
MODES = {"player": "P", "team": "T"}

TIMEOUT_S = 60
MAX_TRIES = 4
PAUSE_S = 2  # pause between requests, to be polite to stats.nba.com


def season_label(start_year: int) -> str:
    """2000 -> '2000-01', the season format nba_api expects."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def raw_path(mode: str, season: str) -> Path:
    return RAW_DIR / f"{mode}_{season}.csv"


def download(season: str, mode_code: str):
    """Call LeagueGameLog for one season, retrying with backoff on failure."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            resp = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation=mode_code,
                timeout=TIMEOUT_S,
            )
            return resp.get_data_frames()[0]
        except Exception as exc:  # stats.nba.com fails with many error types
            if attempt == MAX_TRIES:
                raise
            wait = 5 * 2 ** (attempt - 1)  # 5s, 10s, 20s
            print(f"  attempt {attempt} failed ({exc!r}); retrying in {wait}s")
            time.sleep(wait)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        season = season_label(year)
        for mode, code in MODES.items():
            path = raw_path(mode, season)
            if path.exists():
                print(f"{mode} {season}: cached")
                continue
            df = download(season, code)
            df.to_csv(path, index=False)
            print(f"{mode} {season}: {len(df)} rows")
            time.sleep(PAUSE_S)


if __name__ == "__main__":
    main()
