"""Turn predictions into communicable projections (step 6).

Tiers
    Cut-offs on the target (VORP summed over years 3 and 4), set from the
    real distribution of the development set. The target mixes quality and
    volume, so "Starter" includes future All-Stars whose years 3-4 were
    still developing (e.g. Kawhi Leonard 7.6 is just above the All-Star cut).

Tier probabilities ("neighbours in predicted value")
    For a player predicted at X, take the k historical players whose
    out-of-fold prediction was closest to X and count which tier they really
    reached. A single prediction is an average; this shows the spread of
    real outcomes behind it.

Style comparables ("neighbours in feature space")
    k-NN on the standardised window profile (role, size, usage, skills):
    "who did he look like in his first 25 games?".
"""

import numpy as np
import pandas as pd

TIERS = [
    ("Little or no value", -np.inf, 0.5),
    ("Bench", 0.5, 2.0),
    ("Rotation", 2.0, 4.0),
    ("Starter", 4.0, 6.0),
    ("All-Star level", 6.0, np.inf),
]
TIER_NAMES = [name for name, _, _ in TIERS]

STYLE_FEATURES = ["position", "age_at_debut", "mpg", "usg_pct", "pts36", "ast36",
                  "orb_pct", "drb_pct", "blk36", "stl36", "fg3a_rate", "fta_rate",
                  "ts_pct"]


def tier_of(values) -> pd.Series:
    """Tier name for each target value."""
    edges = [low for _, low, _ in TIERS] + [np.inf]
    return pd.Series(pd.cut(np.asarray(values, dtype=float), edges, right=False,
                            labels=TIER_NAMES), index=getattr(values, "index", None))


def tier_probabilities(preds, ref_preds: pd.Series, ref_actual: pd.Series,
                       k: int | None = None, bandwidth: float | None = None,
                       min_neighbors: int = 0,
                       exclude_self: bool = False) -> pd.DataFrame:
    """Tier shares among the reference players with similar predictions.

    Two ways to define "similar" (pass exactly one):
      k:         the k closest predictions, each counting the same
      bandwidth: every reference player counts, with a weight that falls
                 with the distance in prediction (Gaussian kernel, in VORP
                 units). Unlike k-NN, players at the very top are compared
                 mostly with others predicted near them, instead of all
                 sharing the same top-k group.
    min_neighbors (kernel only): the min_neighbors closest reference players
                 always count in full, so the rare top predictions aren't
                 based on 2 or 3 players.

    preds:        predictions to explain (Series)
    ref_preds:    out-of-fold predictions of the historical players
    ref_actual:   their real target values
    exclude_self: True when `preds` are the reference players themselves
                  (leave-one-out, so nobody is his own neighbour)

    Returns one column per tier plus `n_eff`, the effective number of
    players the percentages are based on.
    """
    if (k is None) == (bandwidth is None):
        raise ValueError("pass exactly one of k or bandwidth")
    onehot = (pd.get_dummies(tier_of(ref_actual))
              .reindex(columns=TIER_NAMES, fill_value=0).to_numpy(dtype=float))
    ref = ref_preds.to_numpy()
    rows = []
    for idx, p in preds.items():
        dist = np.abs(ref - p)
        if k is not None:
            if exclude_self:
                dist[ref_preds.index.get_loc(idx)] = np.inf
            weights = np.zeros(len(ref))
            weights[np.argsort(dist, kind="stable")[:k]] = 1.0
        else:
            if exclude_self:
                dist[ref_preds.index.get_loc(idx)] = np.inf
            weights = np.exp(-0.5 * (dist / bandwidth) ** 2)
            if min_neighbors:
                # the closest min_neighbors players always count in full. Widening
                # the bell curve instead would, for the most extreme prediction,
                # reach down into the crowded middle (all his neighbours are below)
                nearest = np.argsort(dist, kind="stable")[:min_neighbors]
                weights[nearest] = 1.0
        shares = weights @ onehot / weights.sum()
        n_eff = weights.sum() ** 2 / (weights ** 2).sum()
        rows.append(pd.Series([*shares, n_eff], index=[*TIER_NAMES, "n_eff"], name=idx))
    return pd.DataFrame(rows)


def brier_score(probs: pd.DataFrame, actual) -> float:
    """Multi-class Brier score: mean squared distance between the predicted
    tier probabilities and the real tier (one-hot). Lower is better."""
    onehot = pd.get_dummies(tier_of(actual)).reindex(columns=TIER_NAMES, fill_value=0)
    diff = probs[TIER_NAMES].to_numpy() - onehot.to_numpy(dtype=float)
    return float((diff ** 2).sum(axis=1).mean())


# "Empty minutes" warning (notebooks/07_empty_minutes.ipynb). Thresholds were
# fixed before looking at the errors: starter minutes, weakest ~quarter of teams.
HEAVY_MPG = 28
WEAK_TEAM_NET = -5
PROFILES = ["Heavy minutes, weak team", "Heavy minutes, other team"]


def minutes_profile(df: pd.DataFrame) -> pd.Series:
    """Which minutes profile each player has ('' if none)."""
    heavy = df["mpg"] >= HEAVY_MPG
    weak = df["team_net_window"] <= WEAK_TEAM_NET
    return pd.Series(np.select([heavy & weak, heavy], PROFILES, default=""),
                     index=df.index)


def profile_history(ref: pd.DataFrame, ref_preds: pd.Series, target: str = "target_vorp",
                    n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """How far off the model was, historically, for each minutes profile.

    Average error (actual - out-of-fold prediction) with a bootstrap 90% range.
    """
    rng = np.random.default_rng(seed)
    profile = minutes_profile(ref)
    rows = {}
    for name in PROFILES:
        err = (ref[target] - ref_preds)[profile == name].to_numpy()
        boot = rng.choice(err, size=(n_boot, len(err)), replace=True).mean(axis=1)
        rows[name] = {"players": len(err), "avg_error": err.mean(),
                      "low": np.percentile(boot, 5), "high": np.percentile(boot, 95)}
    return pd.DataFrame(rows).T


def profile_warning(profile: str, history: pd.DataFrame) -> str:
    """One line of context for a player card ('' when no profile applies)."""
    if not profile:
        return ""
    h = history.loc[profile]
    direction = "too optimistic" if h["avg_error"] < 0 else "too cautious"
    return (f"Warning - {profile.lower()}: historically the model was {direction} "
            f"for this profile by {abs(h['avg_error']):.1f} VORP on average "
            f"(90% range {h['low']:+.1f} to {h['high']:+.1f}, {int(h['players'])} players)")


def style_comps(players: pd.DataFrame, ref: pd.DataFrame, k: int = 5,
                features=STYLE_FEATURES) -> pd.DataFrame:
    """The k reference players with the most similar window profile.

    Features are standardised with the reference set's mean and spread, so
    each counts the same. A player is never his own comparable.
    Returns one row per (player, rank) with the comp's name and distance.
    """
    mean, std = ref[features].mean(), ref[features].std()
    ref_z = ((ref[features] - mean) / std).to_numpy()
    rows = []
    for idx, row in players.iterrows():
        z = ((row[features] - mean) / std).to_numpy(dtype=float)
        dist = np.sqrt(((ref_z - z) ** 2).sum(axis=1))
        dist[ref["player_id"].to_numpy() == row["player_id"]] = np.inf
        for rank, j in enumerate(np.argsort(dist)[:k], start=1):
            comp = ref.iloc[j]
            rows.append({"player": row["player_name"], "rank": rank,
                         "comp": comp["player_name"], "comp_pick": int(comp["pick"]),
                         "comp_draft_year": int(comp["draft_year"]),
                         "comp_target": comp.get("target_vorp", np.nan),
                         "distance": dist[j]})
    return pd.DataFrame(rows)
