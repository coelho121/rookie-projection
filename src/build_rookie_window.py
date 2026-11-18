"""Find each drafted player's NBA debut and cut his early-game window.

Window definition (see notes.md): the 25 games of the player's team starting
from his debut game, inside the debut season. If the season ends first, the
window is shorter, and `window_team_games` records how long it was.

Inputs:
    data/processed/draft_classes.csv   (src/fetch_draft.py)
    data/raw/gamelogs/*.csv            (src/fetch_gamelogs.py)

Outputs:
    data/processed/rookie_debuts.csv        one row per drafted player
    data/processed/rookie_window_games.csv  one row per player-game in the
                                            window, with team and opponent
                                            totals (needed for usage, rebound %)

Run from the project root:
    python src/build_rookie_window.py              # the main 25-game window
    python src/build_rookie_window.py --window 10  # robustness check: shorter
                                                   # windows write *_w10.csv
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "gamelogs"
PROC_DIR = ROOT / "data" / "processed"

WINDOW = 25  # team games, counted from the player's debut


def window_suffix(window: int) -> str:
    """File suffix for a window length: '' for the main 25 games, '_w10' etc."""
    return "" if window == WINDOW else f"_w{window}"

# Box-score columns kept for the player, his team and the opponent
STATS = [
    "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB",
    "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
]


def load_logs(mode: str) -> pd.DataFrame:
    """Stack every season file of one mode ('player' or 'team')."""
    frames = []
    for path in sorted(RAW_DIR.glob(f"{mode}_*.csv")):
        # GAME_ID has leading zeros ('0020000004'): read it as text
        df = pd.read_csv(path, dtype={"GAME_ID": str})
        season = path.stem.split("_")[1]  # 'player_2000-01' -> '2000-01'
        df["season"] = season
        df["season_start"] = int(season[:4])
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No game logs found: run src/fetch_gamelogs.py")
    logs = pd.concat(frames, ignore_index=True)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    return logs


def find_debuts(draft: pd.DataFrame, player_logs: pd.DataFrame) -> pd.DataFrame:
    """First regular-season game of each drafted player (if he ever played)."""
    logs = player_logs.merge(
        draft[["player_id", "draft_year"]],
        left_on="PLAYER_ID", right_on="player_id",
    )
    # A drafted player can't have NBA games before his draft; guard anyway
    logs = logs[logs["season_start"] >= logs["draft_year"]]
    first = (
        logs.sort_values(["GAME_DATE", "GAME_ID"])
        .groupby("player_id", as_index=False)
        .first()
    )
    return first.rename(
        columns={
            "GAME_DATE": "debut_date",
            "GAME_ID": "debut_game_id",
            "season": "debut_season",
            "TEAM_ID": "debut_team_id",
            "TEAM_ABBREVIATION": "debut_team",
        }
    )[["player_id", "debut_date", "debut_game_id", "debut_season",
       "debut_team_id", "debut_team"]]


def add_window_end(debuts: pd.DataFrame, team_logs: pd.DataFrame,
                   window_games: int = WINDOW) -> pd.DataFrame:
    """For each debut, find the date of the last team game in the window."""
    schedules = {
        key: g.sort_values("GAME_DATE")
        for key, g in team_logs.groupby(["season", "TEAM_ID"])
    }
    ends, lengths = [], []
    for row in debuts.itertuples():
        sched = schedules[(row.debut_season, row.debut_team_id)]
        window = sched[sched["GAME_DATE"] >= row.debut_date].head(window_games)
        ends.append(window["GAME_DATE"].max())
        lengths.append(len(window))
    return debuts.assign(window_end_date=ends, window_team_games=lengths)


def cut_window_games(debuts: pd.DataFrame, player_logs: pd.DataFrame) -> pd.DataFrame:
    """Player games between debut and window end, inside the debut season.

    The window is defined by dates (not game ids) so that a player traded
    during the window keeps his games with the new team.
    """
    games = player_logs.merge(
        debuts[["player_id", "debut_season", "debut_date", "window_end_date"]],
        left_on="PLAYER_ID", right_on="player_id",
    )
    in_window = (
        (games["season"] == games["debut_season"])
        & games["GAME_DATE"].between(games["debut_date"], games["window_end_date"])
    )
    return games[in_window]


def add_team_and_opponent(games: pd.DataFrame, team_logs: pd.DataFrame) -> pd.DataFrame:
    """Attach the team's and the opponent's box-score totals to each game."""
    totals = team_logs[["GAME_ID", "TEAM_ID"] + STATS]
    team = totals.rename(columns={s: f"team_{s}" for s in STATS})
    # Each game has two team rows; the opponent is the row with the other TEAM_ID
    opp = totals.rename(columns={"TEAM_ID": "OPP_TEAM_ID"})
    opp = opp.rename(columns={s: f"opp_{s}" for s in STATS})
    out = games.merge(team, on=["GAME_ID", "TEAM_ID"], how="left")
    out = out.merge(opp, on="GAME_ID", how="left")
    return out[out["OPP_TEAM_ID"] != out["TEAM_ID"]]


def main(window: int = WINDOW) -> None:
    draft = pd.read_csv(PROC_DIR / "draft_classes.csv")
    player_logs = load_logs("player")
    team_logs = load_logs("team")

    debuts = add_window_end(find_debuts(draft, player_logs), team_logs, window)
    games = add_team_and_opponent(cut_window_games(debuts, player_logs), team_logs)

    # Clean up the game table: snake_case, only the columns we need
    games = games[
        ["player_id", "season", "GAME_ID", "GAME_DATE", "TEAM_ID",
         "TEAM_ABBREVIATION", "OPP_TEAM_ID", "MATCHUP", "WL", "PLUS_MINUS"]
        + STATS
        + [f"team_{s}" for s in STATS]
        + [f"opp_{s}" for s in STATS]
    ]
    games.columns = [c.lower() for c in games.columns]
    games = games.rename(columns={"team_abbreviation": "team"})
    games = games.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # Sanity checks
    assert not games.duplicated(["player_id", "game_id"]).any(), "duplicate game"
    assert games["team_pts"].notna().all(), "missing team totals"
    assert games["opp_pts"].notna().all(), "missing opponent totals"
    assert (debuts["window_team_games"] <= window).all()

    # One row per drafted player, with debut info and a window summary
    summary = games.groupby("player_id").agg(
        window_games_played=("game_id", "size"),
        window_minutes=("min", "sum"),
    )
    table = (
        draft.merge(debuts.drop(columns="debut_team_id"), on="player_id", how="left")
        .merge(summary, on="player_id", how="left")
    )
    table["debuted"] = table["debut_date"].notna()

    suffix = window_suffix(window)
    table.to_csv(PROC_DIR / f"rookie_debuts{suffix}.csv", index=False)
    games.to_csv(PROC_DIR / f"rookie_window_games{suffix}.csv", index=False)

    n_debut = table["debuted"].sum()
    print(f"Window: {window} team games")
    print(f"Drafted players: {len(table)}")
    print(f"Debuted: {n_debut}  |  never debuted: {len(table) - n_debut}")
    print(f"Window games saved: {len(games)}")
    print(f"Short windows (< {window} team games): "
          f"{(table['window_team_games'] < window).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, default=WINDOW,
                        help="window length in team games (default: 25)")
    main(parser.parse_args().window)
