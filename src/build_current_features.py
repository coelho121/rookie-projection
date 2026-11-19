"""Features for the current rookie class, from the latest season snapshot.

Same definitions as the historical features (same window rule, same formulas,
same era adjustment), with two differences:

- Shrinkage pulls toward the HISTORICAL average rookie (priors computed from
  the historical classes with the same window length), not toward this
  season's rookies.
- Age and position come from nba_api player info (birth date, position),
  because Basketball-Reference's tables don't have the newest rookies yet.
  Age is taken on Feb 1 of the season, the same definition as BBRef.

While the season is young, windows are shorter than requested: the output
records `window_team_games` so cards can say how much evidence they rest on.

Inputs:
    data/raw/gamelogs_current/*_asof_*.csv   (src/fetch_current_season.py)
    data/raw/player_info/*.csv               (src/fetch_current_season.py)
    data/raw/draft_history.csv               (src/fetch_draft.py)
    data/raw/gamelogs/team_2025-26.csv       previous season, for team ratings
    data/processed/rookie_debuts*.csv, rookie_window_games*.csv  (historical priors)

Output:
    data/processed/current_features[_wN].csv  one row per debuted current rookie

Run from the project root:
    python src/build_current_features.py --window 15
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

import build_features as bf
import build_rookie_window as bw
from fetch_current_season import CURRENT_CLASS, CURRENT_SEASON, INFO_DIR, latest_snapshot

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"

# nba_api positions are coarser than BBRef's 1 (PG) to 5 (C): map to midpoints
NBA_POSITION_NUMBER = {"Guard": 1.5, "Guard-Forward": 2.5, "Forward-Guard": 2.5,
                       "Forward": 3.5, "Forward-Center": 4.5, "Center-Forward": 4.5,
                       "Center": 5.0}


def current_draft_class() -> pd.DataFrame:
    """The current draft class, in the same shape as draft_classes.csv."""
    raw = pd.read_csv(ROOT / "data" / "raw" / "draft_history.csv")
    raw = raw[raw["SEASON"] == CURRENT_CLASS]
    return raw.rename(columns={
        "PERSON_ID": "player_id", "PLAYER_NAME": "player_name", "SEASON": "draft_year",
        "ROUND_NUMBER": "round", "ROUND_PICK": "round_pick", "OVERALL_PICK": "pick",
        "TEAM_ABBREVIATION": "team", "ORGANIZATION": "pre_draft_team",
        "ORGANIZATION_TYPE": "pre_draft_type",
    })[["player_id", "player_name", "draft_year", "round", "round_pick", "pick",
        "team", "pre_draft_team", "pre_draft_type"]]


def player_info(ids) -> pd.DataFrame:
    """Age on Feb 1 of the season and a numeric position, per player."""
    feb1 = date(int(CURRENT_SEASON[:4]) + 1, 2, 1)
    rows = []
    for pid in ids:
        info = pd.read_csv(INFO_DIR / f"{pid}.csv").iloc[0]
        birth = pd.to_datetime(info["BIRTHDATE"]).date()
        age = feb1.year - birth.year - ((feb1.month, feb1.day) < (birth.month, birth.day))
        rows.append({"player_id": pid, "age_at_debut": age,
                     "position": NBA_POSITION_NUMBER.get(info["POSITION"])})
    return pd.DataFrame(rows).set_index("player_id")


def historical_priors(window: int) -> dict:
    """Average-rookie value of each rate stat, from the historical classes."""
    suffix = bw.window_suffix(window)
    debuts = pd.read_csv(PROC_DIR / f"rookie_debuts{suffix}.csv")
    debuts = debuts[debuts["debuted"]].set_index("player_id")
    games = pd.read_csv(PROC_DIR / f"rookie_window_games{suffix}.csv", dtype={"game_id": str})
    league = bf.num_den(bf.team_rows_as_players(), ["season"])
    return bf.rate_features(games, league, debuts["debut_season"]).attrs["priors"]


def main(window: int = bw.WINDOW) -> None:
    player_path, team_path = latest_snapshot("player"), latest_snapshot("team")
    as_of = player_path.stem.split("_asof_")[1]
    print(f"Snapshot as of {as_of}")

    draft = current_draft_class()
    table, games = bw.build_window(draft, bw.load_logs("player", [player_path]),
                                   bw.load_logs("team", [team_path]), window)
    debuts = table[table["debuted"]].copy()
    debuts["debut_date"] = pd.to_datetime(debuts["debut_date"])
    debuts["window_end_date"] = pd.to_datetime(debuts["window_end_date"])
    debuts = debuts.set_index("player_id", drop=False)

    # Previous season too, for the previous-season team rating
    prev_season = f"{int(CURRENT_SEASON[:4]) - 1}-{CURRENT_SEASON[2:4]}"
    teams = bf.team_rows_as_players([bf.RAW_DIR / f"team_{prev_season}.csv", team_path])
    league = bf.num_den(teams, ["season"])

    feats = bf.rate_features(games, league, debuts["debut_season"],
                             priors=historical_priors(window))
    played = games.groupby("player_id").agg(games_played=("game_id", "size"),
                                            minutes=("min", "sum"))
    feats = feats.join(played)
    feats["mpg"] = feats["minutes"] / feats["games_played"]
    feats["games_played_share"] = feats["games_played"] / debuts["window_team_games"]
    feats["window_team_games"] = debuts["window_team_games"]
    feats["pick"] = debuts["pick"]
    feats["round"] = debuts["round"]
    feats["draft_year"] = debuts["draft_year"]
    feats["player_name"] = debuts["player_name"]
    feats["debut_season"] = debuts["debut_season"]
    feats["debut_team"] = debuts["debut_team"]
    feats["debut_lag"] = debuts["debut_season"].str[:4].astype(int) - debuts["draft_year"]
    feats["team_win_pct"] = bf.team_win_pct(debuts, teams)
    feats = feats.join(bf.team_strength(debuts, teams))
    feats = feats.join(player_info(feats.index))
    feats["as_of"] = as_of

    feats = feats.reset_index().rename(columns={"index": "player_id"})
    assert feats[list(bf.RATE_STATS)].notna().all().all(), "NaN after shrinkage"
    out = PROC_DIR / f"current_features{bw.window_suffix(window)}.csv"
    feats.to_csv(out, index=False)
    print(f"Window: up to {window} team games (shortest so far: "
          f"{int(feats['window_team_games'].min())}, longest: {int(feats['window_team_games'].max())})")
    print(f"Current rookies who have debuted: {len(feats)} of {len(draft)} drafted -> {out.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, default=bw.WINDOW,
                        help="window length in team games (default: 25)")
    main(parser.parse_args().window)
