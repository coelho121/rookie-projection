"""Build the target (VORP in years 3-4) and age at debut for every rookie.

Steps:
    1. Link NBA player ids to Basketball-Reference ids through the draft
    2. Collapse Basketball-Reference to one row per player-season
    3. Age at debut, and VORP / minutes / BPM in years 3 and 4 after debut
    4. Flag who is in the modelling scope

Rules (see notes.md):
    - Year 3 = debut season + 2, year 4 = debut season + 3
    - A season with no row (out of the league, didn't play) counts as VORP 0
    - Negative VORP is kept
    - In scope: debuted, age <= 25 at debut, year 4 already completed

Inputs:
    data/processed/rookie_debuts.csv        (src/build_rookie_window.py)
    data/raw/bbref/Advanced.csv             (src/fetch_bbref.py)
    data/raw/bbref/Draft Pick History.csv   (src/fetch_bbref.py)

Output:
    data/processed/rookie_target.csv  one row per debuted player, with an
                                      `in_scope` flag (out-of-scope players are
                                      kept for the robustness check without
                                      the age cut)

Run from the project root:
    python src/build_target.py
"""

import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BBREF_DIR = ROOT / "data" / "raw" / "bbref"
PROC_DIR = ROOT / "data" / "processed"

MAX_AGE = 25              # oldest age at debut kept in scope
LAST_COMPLETE_SEASON = 2026  # season end year of 2025-26

# Picks that neither the name nor the pick number can link (nickname plus the
# 2001-02 pick numbering offset). Key: (draft_year, NBA player name).
MANUAL_IDS = {
    (2002, "Flip Murray"): "murraro01",
}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def normalize(name: str) -> str:
    """'Jimmy Butler III' -> 'jimmy butler'; 'Ömer Aşık' -> 'omer asik'."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace(".", "").replace("'", "").replace("-", " ")
    words = [w for w in text.split() if w not in SUFFIXES]
    return " ".join(words)


def link_bbref_ids(debuts: pd.DataFrame, bb_draft: pd.DataFrame) -> pd.Series:
    """Return the Basketball-Reference id of each drafted player.

    Pass 1: same draft year and same normalized name.
    Pass 2: same draft year and pick number, among players still unlinked.
            (Names differ by nickname, e.g. 'Sun Sun' vs 'Sun Yue'.)
    Pass 3: manual fixes.
    Pick numbers can't be the first key: in 2001 and 2002 the NBA renumbered
    picks after Minnesota's forfeited first-rounder and Basketball-Reference
    didn't, so from pick 29 on they are off by one.
    """
    bb = bb_draft[(bb_draft["lg"] == "NBA")].rename(
        columns={"season": "draft_year", "player_id": "bbref_id"}
    )
    bb["name_key"] = bb["player"].map(normalize)
    nba = debuts[["player_id", "player_name", "draft_year", "pick"]].copy()
    nba["name_key"] = nba["player_name"].map(normalize)

    # Pass 1: name
    ids = nba.merge(
        bb[["draft_year", "name_key", "bbref_id"]],
        on=["draft_year", "name_key"], how="left",
    ).set_index("player_id")["bbref_id"]

    # Pass 2: pick number, only between players unlinked on both sides
    left = nba[nba["player_id"].isin(ids[ids.isna()].index)]
    free = bb[~bb["bbref_id"].isin(ids.dropna())]
    by_pick = left.merge(
        free[["draft_year", "overall_pick", "bbref_id"]],
        left_on=["draft_year", "pick"], right_on=["draft_year", "overall_pick"],
    ).set_index("player_id")["bbref_id"]
    ids = ids.fillna(by_pick)

    # Pass 3: manual fixes
    for (year, name), bbref_id in MANUAL_IDS.items():
        pid = nba.loc[(nba["draft_year"] == year) & (nba["player_name"] == name),
                      "player_id"]
        ids.loc[pid] = bbref_id

    missing = nba.loc[nba["player_id"].isin(ids[ids.isna()].index), "player_name"]
    assert missing.empty, f"unlinked players: {missing.tolist()}"
    assert ids.is_unique, "two NBA players linked to the same BBRef id"
    return ids


def one_row_per_season(adv: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per player-season.

    A player traded mid-season has one row per team plus a total row
    ('2TM', '3TM'...). Keep only the total, otherwise his VORP is counted twice.
    """
    adv = adv[adv["lg"] == "NBA"]
    n_rows = adv.groupby(["season", "player_id"])["team"].transform("size")
    is_total = adv["team"].str.fullmatch(r"\d+TM")
    out = adv[(n_rows == 1) | is_total]
    assert not out.duplicated(["season", "player_id"]).any()
    return out[["season", "player_id", "age", "mp", "bpm", "vorp"]]


def main() -> None:
    debuts = pd.read_csv(PROC_DIR / "rookie_debuts.csv")
    debuts = debuts[debuts["debuted"]].copy()
    adv = pd.read_csv(BBREF_DIR / "Advanced.csv")
    bb_draft = pd.read_csv(BBREF_DIR / "Draft Pick History.csv")

    # 1. Link ids
    ids = link_bbref_ids(debuts, bb_draft)
    debuts["bbref_id"] = debuts["player_id"].map(ids)

    # 2. One row per player-season ('season' = end year: 2001 is 2000-01)
    seasons = one_row_per_season(adv).set_index(["player_id", "season"])

    # 3. Age at debut, and years 3-4
    debuts["debut_end_year"] = debuts["debut_season"].str[:4].astype(int) + 1
    debut_keys = list(zip(debuts["bbref_id"], debuts["debut_end_year"]))
    debuts["age_at_debut"] = seasons["age"].reindex(debut_keys).to_numpy()

    for n, offset in [(3, 2), (4, 3)]:
        keys = list(zip(debuts["bbref_id"], debuts["debut_end_year"] + offset))
        rows = seasons.reindex(keys)
        # No row = didn't play that season -> VORP 0 and 0 minutes
        debuts[f"vorp_y{n}"] = rows["vorp"].fillna(0).to_numpy()
        debuts[f"mp_y{n}"] = rows["mp"].fillna(0).to_numpy()
        debuts[f"bpm_y{n}"] = rows["bpm"].to_numpy()  # stays NaN if no games
    debuts["target_vorp"] = debuts["vorp_y3"] + debuts["vorp_y4"]

    # 4. Scope flags
    debuts["year4_complete"] = debuts["debut_end_year"] + 3 <= LAST_COMPLETE_SEASON
    debuts["age_ok"] = debuts["age_at_debut"] <= MAX_AGE
    debuts["in_scope"] = debuts["year4_complete"] & debuts["age_ok"]

    missing_age = debuts["age_at_debut"].isna().sum()
    assert missing_age == 0, f"{missing_age} players have no BBRef row at debut"

    debuts = debuts.drop(columns=["debuted"])
    debuts.to_csv(PROC_DIR / "rookie_target.csv", index=False)

    print(f"Debuted players: {len(debuts)}")
    print(f"  year 4 not completed yet: {(~debuts['year4_complete']).sum()}")
    print(f"  older than {MAX_AGE} at debut: {(~debuts['age_ok']).sum()}")
    print(f"In scope: {debuts['in_scope'].sum()}")
    print("\nTarget (in scope):")
    print(debuts.loc[debuts["in_scope"], "target_vorp"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
