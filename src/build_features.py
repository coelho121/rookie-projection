"""Turn each rookie's 25-game window into one row of features.

For every rate stat:
    1. Sum the numerator and denominator over the window, then divide
       (never average per-game rates: a 1-shot game would weigh like a
       20-shot game).
    2. Era adjustment: subtract the league value for that season
       (e.g. TS% +3 = 3 points above that season's league TS%).
    3. Shrinkage toward the average rookie, by sample size:
           shrunk = (value * n + rookie_avg * k) / (n + k)
       n is the stat's own sample (shot attempts for shooting stats, minutes
       for the rest) and k is the number of "fictional average-rookie"
       attempts/minutes added. Small samples end up near the average rookie,
       big samples near their real value.

League values use the same formulas applied to team rows: a team is a
"player" who plays all its team's minutes (so league usage comes out at
exactly 20%, one fifth of the team).

The rookie average is pooled over all historical rookies. It uses no target
information, so it doesn't leak the outcome.

Inputs:
    data/processed/rookie_debuts.csv         (src/build_rookie_window.py)
    data/processed/rookie_window_games.csv   (src/build_rookie_window.py)
    data/processed/rookie_target.csv         (src/build_target.py)
    data/raw/gamelogs/team_*.csv             (src/fetch_gamelogs.py)

Outputs:
    data/processed/features.csv     one row per debuted player (no target)
    data/processed/model_table.csv  in-scope players: features + target

Run from the project root:
    python src/build_features.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "gamelogs"
PROC_DIR = ROOT / "data" / "processed"


def plays(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    """Possessions a player 'uses': shots, trips to the line, turnovers."""
    return (df[f"{prefix}fga"] + 0.44 * df[f"{prefix}fta"] + df[f"{prefix}tov"])


def on_court(df: pd.DataFrame) -> pd.Series:
    """Share of the team's game the player was on the floor, times 5."""
    return df["min"] / (df["team_min"] / 5)


# Each rate stat = sum(num) / sum(den) over the window.
#   base: what counts as sample size n ('den' = the stat's own attempts,
#         'min' = minutes played)
#   k:    shrinkage strength, in units of the base. Bigger k = the stat is
#         noisier and needs more sample before we trust it. These are
#         judgment values: shooting % stabilises slowly, rebounding and
#         blocks fast, plus/minus very slowly.
RATE_STATS = {
    # Scoring and shooting
    "ts_pct": dict(
        num=lambda d: d["pts"], den=lambda d: 2 * (d["fga"] + 0.44 * d["fta"]),
        base="den", k=200),
    "fg3_pct": dict(
        num=lambda d: d["fg3m"], den=lambda d: d["fg3a"], base="den", k=200),
    "fg3a_rate": dict(
        num=lambda d: d["fg3a"], den=lambda d: d["fga"], base="den", k=50),
    "fta_rate": dict(
        num=lambda d: d["fta"], den=lambda d: d["fga"], base="den", k=100),
    "pts36": dict(
        num=lambda d: 36 * d["pts"], den=lambda d: d["min"], base="min", k=200),
    # Usage and playmaking
    "usg_pct": dict(
        num=lambda d: plays(d), den=lambda d: on_court(d) * plays(d, "team_"),
        base="min", k=200),
    "ast36": dict(
        num=lambda d: 36 * d["ast"], den=lambda d: d["min"], base="min", k=250),
    "tov_pct": dict(
        num=lambda d: d["tov"], den=lambda d: plays(d), base="den", k=150),
    # Rebounding: share of the available rebounds while on the floor
    "orb_pct": dict(
        num=lambda d: d["oreb"],
        den=lambda d: on_court(d) * (d["team_oreb"] + d["opp_dreb"]),
        base="min", k=200),
    "drb_pct": dict(
        num=lambda d: d["dreb"],
        den=lambda d: on_court(d) * (d["team_dreb"] + d["opp_oreb"]),
        base="min", k=150),
    # Defence (box score) and fouls
    "stl36": dict(
        num=lambda d: 36 * d["stl"], den=lambda d: d["min"], base="min", k=500),
    "blk36": dict(
        num=lambda d: 36 * d["blk"], den=lambda d: d["min"], base="min", k=250),
    "pf36": dict(
        num=lambda d: 36 * d["pf"], den=lambda d: d["min"], base="min", k=200),
    # Impact
    "pm36": dict(
        num=lambda d: 36 * d["plus_minus"], den=lambda d: d["min"],
        base="min", k=800),
}

BOX = ["min", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
       "reb", "ast", "stl", "blk", "tov", "pf", "pts"]


def num_den(games: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sum each stat's numerator and denominator per group (player or season)."""
    parts = {"min": games["min"]}
    for name, s in RATE_STATS.items():
        parts[f"{name}__num"] = s["num"](games)
        parts[f"{name}__den"] = s["den"](games)
    return pd.DataFrame(parts).join(games[keys]).groupby(keys).sum()


def team_rows_as_players() -> pd.DataFrame:
    """Team game logs shaped like player rows (team = a 240-minute player)."""
    frames = []
    for path in sorted(RAW_DIR.glob("team_*.csv")):
        df = pd.read_csv(path, dtype={"GAME_ID": str})
        df["season"] = path.stem.split("_")[1]
        frames.append(df)
    t = pd.concat(frames, ignore_index=True)
    t.columns = [c.lower() for c in t.columns]
    t["game_date"] = pd.to_datetime(t["game_date"])
    t = t[["season", "game_id", "game_date", "team_id", "team_abbreviation",
           "wl", "plus_minus"] + BOX]
    team = t.rename(columns={s: f"team_{s}" for s in BOX})
    opp = t[["game_id", "team_id"] + BOX].rename(
        columns={"team_id": "opp_team_id", **{s: f"opp_{s}" for s in BOX}})
    rows = t.merge(team[["game_id", "team_id"] + [f"team_{s}" for s in BOX]],
                   on=["game_id", "team_id"])
    rows = rows.merge(opp, on="game_id")
    return rows[rows["opp_team_id"] != rows["team_id"]]


def rate_features(games: pd.DataFrame, league: pd.DataFrame,
                  season_of: pd.Series) -> pd.DataFrame:
    """Era-adjusted, shrunk rate stats, one row per player."""
    player = num_den(games, ["player_id"])
    out = pd.DataFrame(index=player.index)
    for name, s in RATE_STATS.items():
        num, den = player[f"{name}__num"], player[f"{name}__den"]
        value = num / den.where(den > 0)  # NaN when there's no sample
        league_value = (league[f"{name}__num"] / league[f"{name}__den"])
        relative = value - season_of.map(league_value)
        n = den if s["base"] == "den" else player["min"]
        rookie_avg = (relative * n).sum() / n[relative.notna()].sum()
        out[name] = (relative.fillna(0) * n + rookie_avg * s["k"]) / (n + s["k"])
        out[f"{name}_raw"] = value  # unadjusted, for exploration only
    return out


def team_win_pct(debuts: pd.DataFrame, teams: pd.DataFrame) -> pd.Series:
    """Win % of the player's debut team during his window."""
    t = teams[["season", "team_abbreviation", "game_date", "wl"]]
    m = debuts[["player_id", "debut_season", "debut_team", "debut_date",
                "window_end_date"]].reset_index(drop=True).merge(
        t, left_on=["debut_season", "debut_team"],
        right_on=["season", "team_abbreviation"])
    m = m[(m["game_date"] >= m["debut_date"]) & (m["game_date"] <= m["window_end_date"])]
    return m.assign(win=m["wl"] == "W").groupby("player_id")["win"].mean()


def main() -> None:
    debuts = pd.read_csv(PROC_DIR / "rookie_debuts.csv",
                         parse_dates=["debut_date", "window_end_date"])
    debuts = debuts[debuts["debuted"]].set_index("player_id", drop=False)
    games = pd.read_csv(PROC_DIR / "rookie_window_games.csv",
                        dtype={"game_id": str})
    target = pd.read_csv(PROC_DIR / "rookie_target.csv")

    teams = team_rows_as_players()
    league = num_den(teams, ["season"])

    feats = rate_features(games, league, debuts["debut_season"])

    # Sample size and coach's trust
    played = games.groupby("player_id").agg(
        games_played=("game_id", "size"), minutes=("min", "sum"))
    feats = feats.join(played)
    feats["mpg"] = feats["minutes"] / feats["games_played"]
    feats["games_played_share"] = (
        feats["games_played"] / debuts["window_team_games"])

    # Context
    feats["pick"] = debuts["pick"]
    feats["round"] = debuts["round"]
    feats["debut_lag"] = (
        debuts["debut_season"].str[:4].astype(int) - debuts["draft_year"])
    feats["team_win_pct"] = team_win_pct(debuts, teams)

    feats = feats.reset_index()
    assert feats["player_id"].is_unique
    assert feats[list(RATE_STATS)].notna().all().all(), "NaN after shrinkage"
    feats.to_csv(PROC_DIR / "features.csv", index=False)

    # Final modelling table: in-scope players only, features + target
    target_cols = ["player_id", "player_name", "draft_year", "debut_season",
                   "bbref_id", "age_at_debut", "vorp_y3", "vorp_y4",
                   "mp_y3", "mp_y4", "bpm_y3", "bpm_y4", "target_vorp"]
    table = target.loc[target["in_scope"], target_cols].merge(
        feats, on="player_id", how="left", validate="one_to_one")
    table.to_csv(PROC_DIR / "model_table.csv", index=False)

    print(f"Features for {len(feats)} players -> features.csv")
    print(f"Model table: {len(table)} players x {table.shape[1]} columns")


if __name__ == "__main__":
    main()
