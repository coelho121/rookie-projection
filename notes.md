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
- Step 5: interpretation notebook `notebooks/03_interpretation.ipynb`; final model definitions moved to `src/models.py` (reproduces CV RMSE 1.889)
- Tested team and role features in `notebooks/04_team_features.ipynb`; downloaded the 1999-00 season for the previous-season team rating
- Step 6: tiers, tier probabilities, style comparables and player cards in `notebooks/05_projections.ipynb`; logic in `src/projection.py`
- Step 7A, robustness in `notebooks/06_robustness.ipynb`: model without the age cut, and 10/15-game windows. `build_rookie_window.py` and `build_features.py` now take `--window N` (writes `*_wN.csv`); `build_features.py` also writes `model_table_all_ages.csv`
- "Empty minutes" follow-up in `notebooks/07_empty_minutes.ipynb` (prompted by Darius Acuff Jr., 2026 class: ~36 mpg and 33% usage on a weak Kings team): tested two more features and built a warning flag for the player cards (`src/projection.py`)
- **Final walk-forward test** on the 2017-2022 classes (331 players) in `notebooks/08_final_test.ipynb`. The model is now frozen

### 2026-11-19
- Step 8, current rookies: `src/fetch_current_season.py` (dated snapshots of the 2026-27 logs + nba_api player info for the 2026 class), `src/build_current_features.py --window N`, and `notebooks/09_current_rookies.ipynb` (first preview with 15-game features, data as of 2026-11-19: 46 of 60 rookies have debuted)
- Refactors with unchanged outputs (verified): `build_rookie_window.build_window()` and `load_logs(paths=...)`, `build_features.team_rows_as_players(paths=...)` and `rate_features(priors=...)`

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
- Interpretation uses out-of-fold permutation importance (single features and whole groups shuffled together, since correlated features share importance) and partial dependence curves on a model fitted to the whole development set
- The honest "added value" of a feature group is the ablation (retrained without it); permutation importance shows how much the fitted model leans on it
- Team and role features (team net rating in the window and the previous season, minutes × team quality, +/- relative to the team, rotation rank, position) were tested as a group, with the rule set before testing: kept only if they beat the current model on the same 25 splits. **Rejected**: they stay in `model_table.csv` (documented) but not in the model
- Tiers on VORP 3+4: Little or no value < 0.5, Bench 0.5-2, Rotation 2-4, Starter 4-6, All-Star level ≥ 6. The All-Star cut is 6, not 8, because the 4-8 range held Kyrie, Westbrook, Dwight and Horford in All-Star years
- Tier probabilities use an adaptive kernel on out-of-fold predictions: Gaussian weights (bandwidth 0.5 VORP) plus the 50 closest players always counting in full, leave-one-out; chosen by Brier score among k-NN, plain kernel and adaptive kernel
  - k-NN gave every top prospect the same neighbours (Lillard = Rose); a plain kernel left top prospects resting on 2-3 players (Carter-Williams 66% All-Star from LeBron and CP3); widening the kernel instead of adding the closest players dragged the most extreme prediction (Rose) into the crowded middle
- Each player card reports the effective number of players behind its percentages
- Style comparables: k-NN (k = 5) on the standardised window profile (position, age, mpg, usage, pts, ast, ORB%, DRB%, blk, stl, 3PA rate, FTA rate, TS%)
- Robustness checks run on the development set before the final test, with the model unchanged (same ensemble and settings), so any change they suggested could still be made honestly. The final test freezes the model
- Shorter windows reuse the same shrinkage settings, so short windows are automatically pulled harder toward the average rookie
- The 25-game window stays the main analysis (it is the project's question); shorter windows only measure how early the signal appears
- "Empty volume" candidates, with rules fixed before running: kept only if mean RMSE drops and they beat the current model in ≥ 15/25 splits
  - `young_minutes_share`: share of the team's window minutes played by other 1st/2nd-year players (team in "development mode"); experience from the first logged season, with the draft year for players already in the league in 1999-00
  - `ts_add36`: points per 36 above a league-average shooter on the same shots (shrunk relative TS% × shot volume), so "empty volume" is strongly negative
  - **Rejected**: both stay in the data, not in the model
- Warning flag on player cards instead of a model change: profile "heavy minutes, weak team" = ≥ 28 mpg (starter minutes) with window net rating ≤ -5 (about the weakest quarter), thresholds fixed before looking at the errors. The card reports the historical average error for the profile with a bootstrap 90% range and the number of players. The model supports the decision; the analyst adds the context the model can't learn
- After the test (decisions made knowing the test results, so recorded as such):
  - The final model for new players keeps the tested specification but is trained on every class with a known outcome (2000-2022), the standard step after evaluation; it also brings in the recent era the test showed was missing
  - Cards for new players show only the "heavy minutes, weak team" warning (`projection.SHOWN_PROFILES`); the "other team" half didn't replicate on the test
- Current-season data: the in-progress season is cached as dated snapshots (`*_asof_YYYY-MM-DD.csv`): same day = reuse, new day = new snapshot, old ones kept for reproducibility
- Current rookies: shrinkage pulls toward the historical average rookie (priors from the historical classes with the same window), not toward this season's rookies; age = age on Feb 1 of the season from the nba_api birth date (same definition as BBRef); position from nba_api mapped to midpoints (Guard 1.5, Forward 3.5, Center 5, hybrids in between) for the style comparables
- Until the rookies reach 25 team games, projections use the model trained on 15-game features (matching the evidence available); switch `WINDOW` to 25 in `09_current_rookies.ipynb` in mid-December
- Final test setup: everything frozen beforehand (ensemble and settings, tiers, adaptive kernel, comparables, warning); one model per test class trained on the past only; reference sets for probabilities, comparables and the warning from the development set only. The warning check on the test classes was added after the first test run, but only evaluates the frozen warning (nothing was changed)
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
- "Empty minutes" (heavy early minutes on weak teams: Carter-Williams, Tyreke Evans, O.J. Mayo, Jonny Flynn) remain a known limitation; team-context features didn't fix them
- Era drift in the target: recent classes reach lower VORP for the same early profile. Mean target 1.07 (classes 2000-06), 1.14 (2007-13), 0.94 (2014-16), 0.86 (2017-22), while window mpg is unchanged (~13); year-3 minutes fell from ~1050-1090 to ~900 (minutes spread across more players). A model trained on older classes over-predicts newer ones (test avg error -0.30 for unflagged players, calibration below the diagonal)

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
- Permutation importance by group (RMSE increase): opportunity +0.33, production +0.17, context +0.04, pick +0.02. With minutes in the model, the pick adds almost nothing: coaches already give top picks minutes
- Top single features: mpg, ast/36, total minutes, ORB%, age, stl/36. TS% and +/- per 36 are ~0: too noisy over 25 games and covered by other features
- Shapes: mpg rises steadily with a jump past ~28 mpg (starter minutes); younger is better until ~23, then flat; assists, steals, rebounding and FT rate are all "more is better" (no U shapes)
- No interaction between minutes and pick: 10 → 30 mpg adds ~+2.0 VORP for picks 3, 15 and 45 alike; the effects simply add up
- The model is closer than the baseline for 61% of players. Corrected upward: CP3, Rose, Lillard, Westbrook, Marc Gasol (#48), Tony Parker (#28); downward: Bennett, Kwame Brown, Bargnani
- Main failure mode: "empty minutes" on bad teams (Carter-Williams predicted 6.6 vs 0.2, Tyreke Evans, O.J. Mayo, Jonny Flynn): heavy early minutes and usage that came from a weak roster, not from quality
- No systematic bias by draft class (no drift, good for the walk-forward test) or by age; picks 1-10 slightly under-predicted (+0.15, the superstars), within noise
- Team features don't help: ensemble + team RMSE 1.899 vs 1.889 (better in only 10/25 splits); Ridge + team 1.903 vs 1.904 (14/25). Added one at a time, none beats noise (best: position −0.002)
- Over-prediction on bad teams is not systematic: by team-quality quintile the errors go −0.22, +0.22, −0.03, −0.08, +0.09, with no trend. The "empty minutes" busts are individual cases, not a learnable pattern
- Team information is largely already in the model: rotation rank overlaps with minutes per game, and team quality with the pick (bad teams pick first)
- Tier shares (dev set): 60% little or no value, 18% bench, 12% rotation, 5% starter, 5% All-Star level
- Tier probabilities beat pick-only: Brier 0.532 vs 0.570 (no information: 0.592), and they are reliable (predicted ≈ observed for P(starter or better))
- Ranking by the model's P(All-Star) finds 17 of the 35 All-Star-level players in the top 50, vs 15 by draft pick
- At the very top the data can't separate LeBron, Rose and CP3 (all ~36% All-Star): too few historical players were predicted that high
- Pick-only reliability shows a CV artefact: with few #1-#2 picks per fold, busts get the highest out-of-fold pick-only probabilities (holding out a class shifts the pick curve against its own players)
- Cards: Lillard 34% All-Star (became All-Star), Marc Gasol 17% All-Star / 18% starter (became starter), Bennett 4% All-Star (bust); misses: Draymond 3% (looked ordinary early), Carter-Williams 38% (empty minutes)
- The age cut doesn't change the result: without it (691 vs 683 players) the gain over the baseline is 10.3% vs 10.1%, Spearman 0.44 in both, 24/25 splits. Only 8 older debutants are in the development set (Splitter, Scola, Navarro...), all with VORP ≤ 2.9; the cut stays for the conceptual reason (formed players, years 3-4 at 30+), not because it moves the numbers
- The signal builds game by game: RMSE gain over the baseline is 3.9% after 10 games (19/25 splits), 7.2% after 15 (21/25) and 10.1% after 25 (24/25); Spearman 0.37 → 0.40 → 0.44. 10 games already hold ~40% of the 25-game gain and 15 games ~70%, and the curve hasn't flattened at 25, so longer windows would likely add more
- Neither empty-volume feature helps: `young_minutes_share` and `ts_add36` each beat the current model in only 9/25 splits (both together 11/25) and nudge RMSE up (+0.001 to +0.003)
- The empty-minutes pattern is real but small and noisy: for heavy-minutes rookies on weak teams the model was too optimistic by 0.7 VORP on average (90% range -1.7 to +0.3, 27 players); for heavy-minutes rookies on other teams it was too cautious by 1.1 (range about 0 to +2.3, 32 players). Early minutes are worth more on a decent team than on a bad one, but ~60 cases are too few for the model to learn it
- The flagged group mixes busts (Carter-Williams, Jonny Flynn, Jay Williams, Dajuan Wagner) and stars (Durant, Anthony Davis, Westbrook, Blake Griffin): heavy minutes on a bad team are a yellow light, not a verdict
- **Final test (2017-2022, walk-forward, 331 players): the model still beats the pick.** RMSE 1.654 → 1.519 (−8.2%; CV was −10%), MAE 1.163 → 1.126, Spearman 0.26 → 0.33 (CV 0.27 → 0.44). Paired bootstrap: gain +0.13, 90% range −0.01 to +0.27, model better in 94% of resamples, so strong but not airtight evidence
- By class: model has lower RMSE in 4 of 6 (2017, 2018, 2020, 2021; worse in 2019 and 2022) and higher Spearman in 4 of 6 (worse in 2019 and 2021)
- By segment: picks 1-10 RMSE 2.57 → 2.44 (Spearman 0.34 → 0.40); picks 11-30 RMSE 1.84 → 1.52 with Spearman 0.05 → 0.35 (the biggest gain: ranking players picked in similar spots); 2nd round RMSE worse 0.86 → 0.94 (Spearman 0.08 → 0.20)
- Tier probabilities hold on test: Brier 0.510 (model) < 0.532 (pick only) < 0.559 (no information)
- Quantile coverage on test: 10.6% above the ceiling (target 10%), but 18% below the floor and 59% below the median: the central case and floor are too optimistic for recent classes (era drift)
- The model is closer than the pick for 52% of test players (61% in CV). Biggest corrections: Haliburton (#12, pick 1.3 → model 5.0, real 9.4), Tatum, Trae Young, Mitchell (#13), Sengun (#16), Luka, Scottie Barnes; Fultz (#1) correctly pulled down (5.2 → 2.2, real 0.0). Biggest misses: Dennis Smith Jr. and RJ Barrett (empty minutes), Bam Adebayo and Robert Williams (quiet starts), Zion (injuries), Garland
- Minutes warning on test: "heavy minutes, weak team" held (avg error −0.94 vs −0.72 in dev, 18 players: Barrett, Ivey, Markkanen...); "heavy minutes, other team" did not (−0.55 vs +1.12 in dev, 15 players), so the "too cautious" half doesn't replicate
- Cards on test players: Luka 38% All-Star (became All-Star, 11.0), Tatum 38% (All-Star), Haliburton 34% (All-Star), Mitchell 30% (starter); Brunson 1% All-Star / 70% little value (became rotation, 3.1); Edwards (#1) only 11% (became starter, 5.8); LaMelo and Zion 36-38% All-Star (became rotation/bench)

### 2026-11-19
- First preview of the 2026 class (15-game features, as of 2026-11-19): top projections Cameron Boozer (#3, 6.1 VORP, 40% All-Star level; looks like Luka, LeBron, Melo), Yaxel Lendeborg (#11, 4.9 vs 1.4 expected from the pick, 38%), Darius Acuff Jr. (#7, 4.7, 38%, with the weak-team warning), Darryn Peterson (#2, 4.2, 30%), AJ Dybantsa (#1, 4.0 vs 4.6 from the pick, 24%, warning; looks like Wiggins and RJ Barrett)
- Biggest early overperformers vs their slot: Lendeborg (#11), Meleek Thomas (#34, 2.6 vs 0.6), Baba Miller (#36); underperformers vs slot: Aday Mara (#12), Nate Ament (#15)
- With the full training set (2000-2022) the weak-team warning reads: too optimistic by 0.5 VORP on average (90% range -1.4 to +0.3, 47 players); it applies to Acuff, Dybantsa, Peterson and Caleb Wilson