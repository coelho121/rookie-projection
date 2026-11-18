"""Shared evaluation rules for every modelling notebook.

Two settings (see notes.md):

Development (cross-validation)
    Players whose target was already known before the first test rookie
    season (year 4 ended by 2016-17): classes 2000-2013. Used for every
    choice: model type, tuning, features. 5 folds grouped by draft class.

Test (walk-forward), used once at the end
    For each test class 2017-2022, a model is trained only on players whose
    year 4 had ended before that class's rookie season, then predicts that
    class. No information from the future reaches any prediction.

Models are passed as factories (functions that return a fresh, unfitted
scikit-learn estimator), so every fold/class gets a clean model.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
MODEL_TABLE = ROOT / "data" / "processed" / "model_table.csv"

TARGET = "target_vorp"
TEST_CLASSES = list(range(2017, 2023))
N_FOLDS = 5
SEGMENTS = ["Picks 1-10", "Picks 11-30", "2nd round"]


def load_table() -> pd.DataFrame:
    """Model table plus helper columns used by the evaluation rules."""
    df = pd.read_csv(MODEL_TABLE)
    # End year of the player's year-4 season: when his target became known
    df["target_known_year"] = df["debut_season"].str[:4].astype(int) + 4
    df["segment"] = np.select(
        [df["pick"] <= 10, df["round"] == 1], SEGMENTS[:2], default=SEGMENTS[2])
    return df


def known_before(df: pd.DataFrame, draft_class: int) -> pd.Series:
    """Players usable to predict `draft_class`: drafted earlier and with a
    target already known when that class plays its rookie season
    (class 2017 -> seasons ending in 2017 or earlier)."""
    return (df["draft_year"] < draft_class) & (df["target_known_year"] <= draft_class)


def dev_set(df: pd.DataFrame) -> pd.DataFrame:
    """Development set: what was known before the first test class."""
    return df[known_before(df, TEST_CLASSES[0])].reset_index(drop=True)


def metrics(y_true, y_pred) -> dict:
    """RMSE (main), MAE and Spearman rank correlation."""
    y_true, y_pred = pd.Series(np.asarray(y_true)), pd.Series(np.asarray(y_pred))
    err = y_pred - y_true
    # A constant prediction (e.g. "mean only") has no ranking: Spearman is undefined
    ranked = y_pred.nunique() > 1
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(err.abs().mean()),
        "spearman": float(y_true.corr(y_pred, method="spearman")) if ranked else np.nan,
    }


def cross_validate(make_model: Callable, features: list[str],
                   dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Grouped 5-fold CV on the development set.

    Returns per-fold metrics and the out-of-fold predictions (each player
    predicted by a model that never saw him).
    """
    folds = GroupKFold(n_splits=N_FOLDS)
    oof = pd.Series(np.nan, index=dev.index)
    rows = []
    for k, (tr, va) in enumerate(folds.split(dev, groups=dev["draft_year"])):
        model = make_model().fit(dev.loc[tr, features], dev.loc[tr, TARGET])
        pred = model.predict(dev.loc[va, features])
        oof.iloc[va] = pred
        classes = sorted(dev.loc[va, "draft_year"].unique())
        rows.append({"fold": k + 1, "classes": classes, "n": len(va),
                     **metrics(dev.loc[va, TARGET], pred)})
    return pd.DataFrame(rows), oof


def walk_forward(make_model: Callable, features: list[str],
                 df: pd.DataFrame) -> pd.DataFrame:
    """Final test: one model per test class, trained only on the past.

    Returns the test players with a `pred` column. Call this once, at the end.
    """
    out = []
    for draft_class in TEST_CLASSES:
        train = df[known_before(df, draft_class)]
        test = df[df["draft_year"] == draft_class]
        model = make_model().fit(train[features], train[TARGET])
        out.append(test.assign(pred=model.predict(test[features]),
                               n_train=len(train)))
    return pd.concat(out)


def summarize(fold_metrics: pd.DataFrame) -> pd.Series:
    """Mean and standard deviation of each metric across folds."""
    m = fold_metrics[["rmse", "mae", "spearman"]]
    return pd.concat([m.mean().add_suffix("_mean"), m.std().add_suffix("_std")])


def by_segment(df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """Metrics split by pick segment."""
    rows = {seg: metrics(df.loc[df["segment"] == seg, TARGET],
                         pred[df["segment"] == seg]) for seg in SEGMENTS}
    out = pd.DataFrame(rows).T
    out["n"] = df["segment"].value_counts().reindex(SEGMENTS)
    return out
