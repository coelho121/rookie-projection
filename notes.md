# Rookie Projection - Project Notes

## Flow

### 2026-11-17
- Defined the project question: do a rookie's first ~25 NBA games tell us something about his year 3-4 value that draft position didn't?
- Reviewed the core metrics (per-36, BPM, VORP) before choosing the target
- Set up `.venv` and `requirements.txt`; added `data/` to `.gitignore`
- Collected draft classes 2000-2022 (1367 picks) with `src/fetch_draft.py`
- Downloaded regular-season game logs 2000-01 to 2025-26 (player and team) with `src/fetch_gamelogs.py`
- Built debuts and the 25-game window with `src/build_rookie_window.py`: 1200 players debuted, 167 never did

### 2026-11-18
- Got VORP / BPM / minutes / age from the Kaggle mirror of Basketball-Reference with `src/fetch_bbref.py`
- Built the target with `src/build_target.py`: 1180 players in scope
- Built the features with `src/build_features.py`: `features.csv` (1200 players) and `model_table.csv` (1180 in scope, 49 columns)
- Planned the modelling workflow: EDA, time split, pick-only baseline, Ridge and gradient boosting, metrics by segment, interpretation, comparables, robustness, current rookies
- EDA notebook `notebooks/01_eda.ipynb`
- Replaced the plain time split with walk-forward evaluation; set up `src/modelling.py` and the pick-only baseline in `notebooks/02_modelling.ipynb`
- Step 3 in `notebooks/02_modelling.ipynb`: tuned and compared Ridge, Ridge + splines, gradient boosting and an ensemble; ablation by feature group; calibration; quantile models for floor and ceiling

## Decisions

### 2026-11-17
- Target: VORP summed over years 3 and 4 (continuous)
  - VORP combines quality and volume, so players who barely play or leave the league get ~0 automatically, which handles busts with no extra rules
  - Categories (starter / rotation / bust) are derived from the VORP target at the end, for communication only
- BPM and per-36 stats are used as complementary features (early-season inputs), not as targets
- Baseline to beat: draft position alone
- Players who left the league before year 3 are kept in the dataset (avoids survivorship bias)
- Draft data comes from the nba_api `DraftHistory` endpoint: it returns the official NBA `player_id`, which links directly to the rookie game logs
- Game logs come from `LeagueGameLog` (one request per season, player and team modes) instead of `PlayerGameLog` (one request per player): ~50 requests instead of ~1300
- Rookie season = first season with an NBA regular-season game (handles draft-and-stash and players injured in their first year); years 3-4 are counted from the debut season
- Players who never debuted are excluded from modelling (their count is recorded); players who debuted and then left the league stay, with VORP ~0
- Low VORP for late picks is expected and already captured by the draft-position baseline. The real question is whether early games predict over/under-performance relative to draft slot, so evaluation should check where the model beats the baseline (late-pick steals, top-pick busts)
- Early-game window = the 25 team games starting from the player's debut game, inside the debut season (shorter if the season ends first)
  - Same window length for everyone, so no player gets extra information (avoids leakage)
  - Anchoring on the debut means players who debut mid-season still get a window
- Injuries (for the features step): use rate stats and minutes per game *played* as the main signals, and treat games/minutes in the window as sample size, not quality. Consider shrinking small-sample stats toward typical values for the draft slot
- The window is cut by dates (debut to 25th team game), not by game ids, so players traded during the window keep their games with the new team

### 2026-11-18
- Only players aged 25 or younger at debut are modelled (26+ are cut)
  - Older debutants (mostly draft-and-stash) are a different population: their first games show an already-formed player, not a prospect, and their years 3-4 fall at 30+, in decline
  - No leakage (age is known at debut) and no survivorship bias (the cut doesn't depend on the outcome); it only defines the scope
  - Age comes from the Basketball-Reference season tables (age on Feb 1 of the season), applied when building the target
  - At the end, rerun the model without the cut as a robustness check
- Negative VORP is kept in the target (playing many bad minutes is a real cost), even though it ranks some busts below players who left the league (VORP 0)
- How results are communicated: a single predicted VORP is shrunk toward the mean and hides uncertainty, so each player's prediction is compared with historical players who got a similar *predicted* VORP, and their *actual* outcomes give tier probabilities (e.g. 15% All-Star, 40% starter...) plus named comparables
  - Tier cut-offs are set from the real VORP distribution at the end
  - Minutes and BPM of years 3-4 are kept (not as targets) to describe the comparables
- Two final deliverables: (1) does the 25-game window beat draft position alone, by how much and for whom; (2) projections for the current rookie class once they reach 25 games
- Basketball-Reference data comes from the Kaggle dataset "NBA Stats (1947-present)" (Sumitro Datta, CC0), because basketball-reference.com blocks automated clients with a Cloudflare challenge
  - Validated against known BBRef values (LeBron 2008-09 VORP 11.8, Jokić 2021-22 VORP 9.8) and against nba_api logs (games match 99.9%, minutes within 2% in 98.5% of player-seasons)
- NBA and BBRef ids are linked through the draft: same year + normalized name, then same year + pick, then one manual fix
- Traded players: only the season-total row (`2TM`, `3TM`) is used, so VORP isn't counted twice
- Rate stats are summed over the window and then divided (never averaged per game)
- Era adjustment: each rate stat is expressed relative to the league value for that season; league values come from the same formulas applied to team rows (league usage = exactly 20%)
- Shrinkage toward the average rookie: `(value × n + rookie_avg × k) / (n + k)`, with n = the stat's own attempts for shooting stats and minutes for the rest
  - k values (judgment): TS% 200, 3P% 200, 3PA rate 50, FTA rate 100, pts/36 200, usage 200, ast/36 250, TOV% 150, ORB% 200, DRB% 150, stl/36 500, blk/36 250, PF/36 200, +/- per 36 800
  - The rookie average is pooled over all historical rookies (no target information, so no leakage)
- No full BPM is computed for the window; the model gets its components instead
- Test classes: 2017-2022, evaluated once at the end with a **walk-forward** scheme: each test class gets its own model, trained only on players whose year 4 had already ended before that class's rookie season (e.g. class 2017 -> targets known by 2016-17, in practice classes up to 2013)
  - Mirrors real use with no information from the future. A plain split (train 2000-2016) would have used year 3-4 outcomes from seasons after the test rookies debuted
- Development set for cross-validation (model choice, tuning): players whose target was known before the first test rookie season (year 4 ended by 2016-17): 683 players, classes 2000-2013. 5 folds grouped by draft class (GroupKFold)
  - Grouping by class keeps players of the same class/era out of train and validation at the same time
  - A model must beat the baseline in most folds, not only on average
  - All preprocessing that learns from data (e.g. scaling) is fitted inside each fold through a scikit-learn `Pipeline`
  - Cross-validation also gives out-of-fold predictions, later used for the comparables
- Main metric: RMSE (the costly errors are missed stars and top-pick busts), with MAE and Spearman correlation as checks, and results split by segment (picks 1-10, 11-30, 2nd round)
- Baseline = quadratic regression on log(pick), chosen by CV among mean-only, linear on pick, quadratic on log(pick) and isotonic on pick
- Feature models get the pick as log(pick) and log(pick)², like the baseline, so the window features only have to explain what the pick doesn't
- Candidates: Ridge (linear, penalised), Ridge + splines (one smooth curve per feature, GAM-like, still interpretable), gradient boosting (`HistGradientBoostingRegressor`) and an ensemble averaging the best linear model and boosting. Random forest, XGBoost/LightGBM and neural nets were left out (weaker on extremes, same family, or too little data)
- Model comparison uses repeated grouped CV (5 repeats × 5 folds = 25 paired splits); tuning uses the plain 5-fold CV
- Tuned settings: Ridge alpha 30; Ridge + splines alpha 300 (5 knots); boosting 8 leaves, 40 min per leaf, 300 iterations, learning rate 0.03
- Main model: **the ensemble** (Ridge + splines averaged with gradient boosting). Its edge over plain Ridge is small (beats it in 15/25 splits), but every difference goes where this project's priorities are: lower RMSE overall and at picks 1-10, better calibration at the top, bolder top predictions. Plain Ridge only wins on ranking 2nd-rounders, the least important segment
  - First chose plain Ridge by the generic "simpler wins a tie" rule; switched because accuracy at the top picks matters more here than simplicity
  - Interpretation stays possible: spline curves for the linear half, permutation importance for the whole model
  - Chosen before the walk-forward test, so the test result can't influence the choice
- Results are not made "bolder" by stretching predictions (that would be dishonest and raise the error). Upside is shown instead through quantile models (p10 floor, p50 central case, p90 ceiling) and, later, tier probabilities from comparables
- Two kinds of comparables for step 6: neighbours in *predicted value* (for tier probabilities) and neighbours in *feature space* (k-NN on the window profile, for "plays like X" names)
- EDA uses the classes drafted 2000-2016 only, so no decision is informed by the test set (it was done before the walk-forward scheme; classes 2014-2016 only shaped general impressions such as the U-shaped relations)

## Problems

### 2026-11-17
- Windows Smart App Control blocked a DLL in numpy 2.5.5; pinned `numpy==2.2.6` as a workaround
- Basketball-Reference uses its own player ids, so NBA ids will need to be mapped to BBRef ids before building the target
- Late debuts in recent classes (e.g. drafted 2022, debut 2023-24) don't have a completed year 4 yet, so they have no target and are dropped
- Game logs only list games a player appeared in, so injury absences can't be told apart from coach's decisions (the DNP reason is only in per-game box scores, which would take tens of thousands of requests)
- Injuries in years 3-4 lower the target, and no model built on 25 games can foresee them. This is unexplainable noise and goes in the project's limitations

### 2026-11-18
- In 2001 and 2002, NBA pick numbers from pick 29 on are one lower than BBRef's (the NBA renumbered after Minnesota's forfeited pick), so pick number alone can't link the sources
- BBRef values use BPM 2.0 (revised in 2020); older published numbers (e.g. Westbrook 2016-17 VORP 12.4) no longer match

## Findings

### 2026-11-17
- Some classes have fewer than 60 picks (2000-2004, 2022). This is expected: only 29 teams existed before 2004, and some picks were forfeited
- 163 of the 167 players who never debuted are second-rounders; the 4 first-rounders stayed in Europe (e.g. Fran Vázquez)
- Rookies played 15 of the 25 window games on average
- 35 windows are shorter than 25 team games (late-season debuts, 6 of them in the COVID-shortened 2019-20 season)
- Some draft-and-stash players debut 5 to 9 years after the draft (Scola, Mičić), already 27-28 years old

### 2026-11-18
- The target is very skewed: median 0, 242 players (20%) out of the league by year 3, 276 with negative VORP, max 19.2 (Chris Paul)
- Only 14 players debuted older than 25; 8 more lack a completed year 4
- League TS% rose from 51.8% (2000-01) to 58.1% (2025-26) and 3PA rate from 17% to 41.5%, which confirms the need for era adjustment
- Strongest single correlations with the target (Spearman): minutes per game 0.35, games played 0.32, pick −0.26, TS% 0.21, age −0.16. Early minutes beat pick on their own, but they are partly driven by pick
- EDA (`notebooks/01_eda.ipynb`, classes 2000-2016, 849 players):
  - Year 3 and year 4 VORP correlate at r = 0.79, so summing both years gives a stable target
  - Out-of-league share by year 3 grows with pick: ~2% for picks 1-15, ~15% for 21-30, over 60% for 51-60
  - Within each pick segment, more early minutes go with more value later: the minutes signal is not only the pick in disguise
  - Features most related to value beyond the pick (Spearman with actual minus pick-expected): TS% +0.17, mpg +0.16, stl/36 +0.16, +/- per 36 +0.15, ast/36 +0.14
  - Several features have U-shaped relations with the target (ORB%, blk/36, PF/36, 3PA rate), likely a position effect, which favours non-linear models
  - Some busts looked fine early (Wiggins 31 mpg as a rookie), so part of the bust risk is invisible in 25 games
- Baseline CV scores (dev set, 683 players): RMSE 2.12 ± 0.30, MAE 1.38, Spearman 0.27 (mean-only: RMSE 2.31). The pick alone explains little
- Any decreasing function of pick ranks players the same way, so the baseline's Spearman is fixed (~0.27); a better ranking can only come from the window features
- Baseline errors concentrate in picks 1-10 (RMSE 3.5 vs 1.8 for picks 11-30 and 1.2 for the 2nd round)
- Its biggest misses are early stars (Chris Paul, LeBron, Wade, Durant, Giannis, Draymond) and top-pick busts (Bennett, Kwame Brown, Oden, Turner)
- **Window features beat the pick-only baseline** (repeated CV): ensemble RMSE 2.10 → 1.89 (−10%), MAE 1.38 → 1.23, Spearman 0.27 → 0.44, better in 24 of 25 splits. Plain Ridge is almost the same (RMSE 1.90, Spearman 0.47)
- Gain is largest at picks 1-10 (ensemble RMSE 3.52 → 3.00, Spearman 0.19 → 0.52); within picks 11-30 the ranking goes from none (−0.04) to 0.40; 2nd round: RMSE 1.24 → 1.22, Spearman 0.01 → 0.27 (plain Ridge ranks 2nd-rounders better: 0.35)
- Gradient boosting is the weakest feature model (beats Ridge in 5/25 splits): with 683 players a regularised linear model wins, and the U shapes from the EDA don't pay off
- Ablation (Ridge + splines): pick 2.12 → + context 2.12 → + opportunity 1.94 → + production 1.91. Opportunity (minutes, games) brings ~85% of the gain; production adds a smaller extra; context adds nothing beyond the pick. The early games help mostly because they reveal the coach's trust
- The ensemble is well calibrated (top decile predicted 4.3 vs actual 4.35) and less conservative than the baseline (max prediction 8.8 vs 5.6; 18 players predicted ≥ 5 vs 12). Plain Ridge is slightly timid at the top (top decile 4.2 vs 4.5; max 6.9)
- Quantile models: features improve the p90 ceiling most (pinball loss −12%); coverage is a bit narrow (15% of players exceed p90 instead of 10%). Busts get low central cases (Bennett, Kwame Brown ~1 vs ~6 expected from the pick), but superstars (LeBron, CP3, Durant) land far above any ceiling: 25 games can't tell an All-Star from the best player in the world