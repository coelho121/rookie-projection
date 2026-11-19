"""Player cards for every rookie: the historical classes and the current one.

A card = what the system says after the first games: what the pick alone
expects, the model's central case, floor and ceiling, tier probabilities,
style comparables and the minutes warning (plus, for past players, what
really happened).

Historical cards (classes 2000-2022, 25-game window) are honest "as of then"
predictions: every player is predicted by a model that never saw him.
    - development players (target known before 2017): out-of-fold, repeated CV
    - everyone else: walk-forward by draft class (model trained only on players
      whose year 4 had ended before that class's rookie season)
Tier probabilities and the warning use the development set as reference, as
in notebooks 05 and 08. Style comparables come from all eras (they describe a
profile, not an outcome).

Current-class cards use the final model (trained on 2000-2022) with the
requested window, as in notebooks/09_current_rookies.ipynb.

Outputs:
    reports/cards_history.md        readable, by draft class
    reports/cards_2026_class.md     readable, by projection
    data/processed/cards_history.csv, cards_2026_class.csv

Run from the project root (after the build scripts and build_current_features.py):
    python src/build_cards.py --window 15     # window for the current class
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import modelling as m
import models as md
import projection as pj
from build_rookie_window import WINDOW, window_suffix
from fetch_current_season import CURRENT_CLASS

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
QUANTILES = [0.1, 0.5, 0.9]
KERNEL = dict(bandwidth=0.5, min_neighbors=50)


# ------------------------------------------------------------------ predictions
def predict_history(df: pd.DataFrame, dev: pd.DataFrame, make_model, features) -> pd.Series:
    """Out-of-fold for the development set, walk-forward by class for the rest.

    Returns predictions indexed by player_id.
    """
    _, oof = m.repeated_cv(make_model, features, dev)
    parts = [pd.Series(oof.to_numpy(), index=dev["player_id"].to_numpy())]
    others = df[~df["player_id"].isin(dev["player_id"])]
    for draft_class in sorted(others["draft_year"].unique()):
        train = df[m.known_before(df, draft_class)]
        rows = others[others["draft_year"] == draft_class]
        if len(train) < 200:
            raise ValueError(f"only {len(train)} players known before class {draft_class}")
        model = make_model().fit(train[features], train[m.TARGET])
        parts.append(pd.Series(model.predict(rows[features]), index=rows["player_id"].to_numpy()))
    return pd.concat(parts)


def quantile_preds(predict) -> pd.DataFrame:
    """p10 / p50 / p90 with any prediction routine, sorted so they never cross."""
    cols = [predict(lambda q=q: md.boosting(loss="quantile", quantile=q)) for q in QUANTILES]
    raw = pd.concat(cols, axis=1)
    return pd.DataFrame(np.sort(raw.to_numpy(), axis=1), index=raw.index,
                        columns=["p10", "p50", "p90"])


# ------------------------------------------------------------------ text
def card_markdown(r: pd.Series, comps: list[str], warning: str, history_card: bool) -> str:
    lines = [f"#### {r['player_name']} - #{int(r['pick'])} pick, {int(r['draft_year'])}"
             + (f", {r['debut_team']}" if isinstance(r.get("debut_team"), str) else "")]
    lines.append(f"- First games: {r['mpg']:.1f} mpg in {int(r['games_played'])} of "
                 f"{int(r['window_team_games'])} team games, age {int(r['age_at_debut'])}")
    lines.append(f"- Pick alone expects **{r['base_pred']:.1f}** VORP (years 3+4) | "
                 f"model **{r['model_pred']:.1f}** | floor {r['p10']:.1f}, "
                 f"median {r['p50']:.1f}, ceiling {r['p90']:.1f}")
    lines.append("- Tiers: " + " · ".join(f"{t} {r[t]:.0%}" for t in pj.TIER_NAMES)
                 + f" (~{r['n_eff']:.0f} similar players)")
    lines.append("- Looks like: " + ", ".join(comps))
    if warning:
        lines.append(f"- ⚠️ {warning}")
    if r["age_at_debut"] > 25:
        lines.append("- Note: debuted older than 25, outside the model's population")
    if history_card:
        lines.append(f"- **What happened:** {r[m.TARGET]:.1f} VORP -> {r['tier']}")
    return "\n".join(lines)


def comps_by_player(players: pd.DataFrame, ref: pd.DataFrame, k: int = 3) -> dict:
    c = pj.style_comps(players, ref, k=k)
    c["label"] = c["comp"] + " (#" + c["comp_pick"].astype(str) + ", " + c["comp_draft_year"].astype(str) + ")"
    return c.groupby("player_id")["label"].apply(list).to_dict()


def table_columns(cards: pd.DataFrame, history: bool) -> pd.DataFrame:
    cols = ["player_id", "player_name", "draft_year", "pick", "age_at_debut", "games_played",
            "window_team_games", "mpg", "base_pred", "model_pred", "p10", "p50", "p90",
            *pj.TIER_NAMES, "n_eff", "comps", "warning"]
    if history:
        cols += [m.TARGET, "tier", "prediction_type"]
    return cards[cols]


# ------------------------------------------------------------------ historical
def history_cards() -> pd.DataFrame:
    df = md.add_pick_terms(m.load_table())
    dev = m.dev_set(df)
    predict = lambda make, feats=md.ALL: predict_history(df, dev, make, feats)

    model_pred = predict(md.ensemble)
    base_pred = predict(md.baseline, ["pick"])
    quant = quantile_preds(predict)

    cards = df.set_index("player_id", drop=False).copy()
    cards["model_pred"], cards["base_pred"] = model_pred, base_pred
    cards = cards.join(quant)
    in_dev = cards["player_id"].isin(dev["player_id"])
    cards["prediction_type"] = np.where(in_dev, "out-of-fold (development)", "walk-forward")

    ref = cards.loc[in_dev, "model_pred"]
    ref_actual = cards.loc[in_dev, m.TARGET]
    probs = pd.concat([
        pj.tier_probabilities(ref, ref, ref_actual, exclude_self=True, **KERNEL),
        pj.tier_probabilities(cards.loc[~in_dev, "model_pred"], ref, ref_actual, **KERNEL),
    ])
    cards = cards.join(probs)
    cards["tier"] = pj.tier_of(cards[m.TARGET])

    history = pj.profile_history(dev, pd.Series(ref.reindex(dev["player_id"]).to_numpy(), index=dev.index))
    profile = pj.minutes_profile(cards)
    cards["warning"] = [pj.profile_warning(p, history, shown=pj.SHOWN_PROFILES) for p in profile]
    cards["window_team_games"] = cards["games_played"] / cards["games_played_share"]
    comps = comps_by_player(cards.reset_index(drop=True), df)
    cards["comps"] = cards["player_id"].map(lambda pid: comps.get(pid, []))
    return cards


# ------------------------------------------------------------------ current
def current_cards(window: int) -> pd.DataFrame:
    suffix = window_suffix(window)
    train = md.add_pick_terms(m.load_table("model_table" + suffix))
    cur = md.add_pick_terms(pd.read_csv(PROC_DIR / f"current_features{suffix}.csv"))
    cur = cur.set_index("player_id", drop=False)

    final = md.ensemble().fit(train[md.ALL], train[m.TARGET])
    cur["model_pred"] = final.predict(cur[md.ALL])
    cur["base_pred"] = md.baseline().fit(train[["pick"]], train[m.TARGET]).predict(cur[["pick"]])
    quant = quantile_preds(lambda make: pd.Series(
        make().fit(train[md.ALL], train[m.TARGET]).predict(cur[md.ALL]), index=cur.index))
    cur = cur.join(quant)

    _, train_oof = m.repeated_cv(md.ensemble, md.ALL, train)
    cur = cur.join(pj.tier_probabilities(cur["model_pred"], train_oof, train[m.TARGET], **KERNEL))
    history = pj.profile_history(train, train_oof)
    cur["warning"] = [pj.profile_warning(p, history, shown=pj.SHOWN_PROFILES)
                      for p in pj.minutes_profile(cur)]
    comps = comps_by_player(cur.reset_index(drop=True), train)
    cur["comps"] = cur["player_id"].map(lambda pid: comps.get(pid, []))
    return cur


# ------------------------------------------------------------------ main
def main(window: int) -> None:
    REPORTS.mkdir(exist_ok=True)

    hist = history_cards()
    table_columns(hist, history=True).assign(comps=lambda d: d["comps"].str.join("; ")).to_csv(
        PROC_DIR / "cards_history.csv", index=False)
    parts = ["# Player cards - every rookie, 2000-2022\n",
             "What the system would have said after each player's first 25 games, using "
             "only players it had never seen (out-of-fold for classes up to ~2013, "
             "walk-forward by class after that), and what really happened. VORP is summed "
             "over years 3 and 4. Style comparables come from all eras.\n"]
    for draft_class, g in hist.sort_values(["draft_year", "pick"]).groupby("draft_year"):
        parts.append(f"\n## {draft_class} draft class\n")
        parts += [card_markdown(r, r["comps"], r["warning"], history_card=True) + "\n"
                  for _, r in g.iterrows()]
    (REPORTS / "cards_history.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Historical cards: {len(hist)} players -> reports/cards_history.md")

    cur = current_cards(window)
    as_of = cur["as_of"].iloc[0]
    name = f"cards_{CURRENT_CLASS}_class"
    table_columns(cur, history=False).assign(comps=lambda d: d["comps"].str.join("; "),
                                             as_of=as_of).to_csv(PROC_DIR / f"{name}.csv", index=False)
    parts = [f"# Player cards - {CURRENT_CLASS} draft class (as of {as_of})\n",
             f"Projections for years 3-4 after each rookie's first games so far (up to "
             f"{window} team games), from the final model trained on every class from 2000-2022. "
             "Ordered by the model's projection. Rookies with very few games rest on little evidence; "
             "recent classes have tended to be projected a bit optimistically.\n"]
    parts += [card_markdown(r, r["comps"], r["warning"], history_card=False) + "\n"
              for _, r in cur.sort_values("model_pred", ascending=False).iterrows()]
    (REPORTS / f"{name}.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Current-class cards: {len(cur)} players (as of {as_of}) -> reports/{name}.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, default=WINDOW,
                        help="window for the current class (default: 25)")
    main(parser.parse_args().window)
