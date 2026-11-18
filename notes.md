# Rookie Projection - Project Notes

## Flow

### 2026-11-17
- Defined the project question: do a rookie's first ~25 NBA games tell us something about his year 3-4 value that draft position didn't?
- Reviewed the core metrics (per-36, BPM, VORP) before choosing the target
- Set up `.venv` and `requirements.txt`; added `data/` to `.gitignore`
- Collected draft classes 2000-2022 (1367 picks) with `src/fetch_draft.py`
- Downloaded regular-season game logs 2000-01 to 2025-26 (player and team) with `src/fetch_gamelogs.py`
- Built debuts and the 25-game window with `src/build_rookie_window.py`: 1200 players debuted, 167 never did

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

## Problems

### 2026-11-17
- Windows Smart App Control blocked a DLL in numpy 2.5.5; pinned `numpy==2.2.6` as a workaround
- Basketball-Reference uses its own player ids, so NBA ids will need to be mapped to BBRef ids before building the target
- Late debuts in recent classes (e.g. drafted 2022, debut 2023-24) don't have a completed year 4 yet, so they have no target and are dropped
- Game logs only list games a player appeared in, so injury absences can't be told apart from coach's decisions (the DNP reason is only in per-game box scores, which would take tens of thousands of requests)
- Injuries in years 3-4 lower the target, and no model built on 25 games can foresee them. This is unexplainable noise and goes in the project's limitations

## Findings

### 2026-11-17
- Some classes have fewer than 60 picks (2000-2004, 2022). This is expected: only 29 teams existed before 2004, and some picks were forfeited
- 163 of the 167 players who never debuted are second-rounders; the 4 first-rounders stayed in Europe (e.g. Fran Vázquez)
- Rookies played 15 of the 25 window games on average
- 35 windows are shorter than 25 team games (late-season debuts, 6 of them in the COVID-shortened 2019-20 season)
- Some draft-and-stash players debut 5 to 9 years after the draft (Scola, Mičić), already 27-28 years old