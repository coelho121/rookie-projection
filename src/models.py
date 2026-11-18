"""Feature groups and the final model definitions (chosen in 02_modelling).

Settings were tuned with grouped CV on the development set; see notes.md.
Every factory returns a fresh, unfitted scikit-learn model.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler

# Feature groups
PICK = ["log_pick", "log_pick2"]
CONTEXT = ["age_at_debut", "debut_lag", "team_win_pct"]
OPPORTUNITY = ["mpg", "games_played_share", "minutes"]
PRODUCTION = ["ts_pct", "fg3_pct", "fg3a_rate", "fta_rate", "pts36", "usg_pct",
              "ast36", "tov_pct", "orb_pct", "drb_pct", "stl36", "blk36", "pf36", "pm36"]
ALL = PICK + CONTEXT + OPPORTUNITY + PRODUCTION
GROUPS = {"Pick": PICK, "Context": CONTEXT, "Opportunity": OPPORTUNITY,
          "Production": PRODUCTION}

# Candidate group added after interpretation (notebooks/04_team_features.ipynb)
TEAM = ["team_net_prev", "team_net_window", "mpg_x_team_net", "rel_pm36",
        "rotation_rank", "position"]

# "Empty volume" candidates (notebooks/07_empty_minutes.ipynb)
EMPTY_VOLUME = ["young_minutes_share", "ts_add36"]

# Tuned settings (notebooks/02_modelling.ipynb)
RIDGE_ALPHA = 30
SPLINE_ALPHA = 300
BOOSTING = dict(leaves=8, min_leaf=40, iters=300)


def add_pick_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add log(pick) and log(pick)^2, the baseline's shape of the pick."""
    df = df.copy()
    df["log_pick"] = np.log(df["pick"])
    df["log_pick2"] = df["log_pick"] ** 2
    return df


def _log_terms(X):
    lp = np.log(np.asarray(X, dtype=float))
    return np.column_stack([lp, lp ** 2])


def baseline():
    """Pick only: quadratic regression on log(pick). Uses the 'pick' column."""
    return make_pipeline(FunctionTransformer(_log_terms), LinearRegression())


def ridge(alpha=RIDGE_ALPHA):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def spline_ridge(alpha=SPLINE_ALPHA, features=ALL):
    """Ridge where every non-pick feature becomes a smooth curve (5 knots)."""
    pick = [f for f in features if f in PICK]
    curves = [f for f in features if f not in PICK]
    return make_pipeline(
        ColumnTransformer([
            ("pick", "passthrough", pick),
            ("curves", SplineTransformer(n_knots=5, degree=3), curves),
        ]),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def boosting(leaves=BOOSTING["leaves"], min_leaf=BOOSTING["min_leaf"],
             iters=BOOSTING["iters"], loss="squared_error", quantile=None):
    params = dict(loss=loss, learning_rate=0.03, max_iter=iters,
                  max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
                  l2_regularization=1.0, early_stopping=False, random_state=0)
    if quantile is not None:
        params["quantile"] = quantile
    return HistGradientBoostingRegressor(**params)


def ensemble(features=ALL):
    """The chosen model: average of Ridge + splines and gradient boosting."""
    return VotingRegressor([("linear", spline_ridge(features=features)),
                            ("boosting", boosting())])
