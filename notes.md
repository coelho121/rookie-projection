# Rookie Projection - Project Notes

## Flow

### 2026-11-17
- Defined the project question: do a rookie's first ~25 NBA games tell us something about his year 3-4 value that draft position didn't?
- Reviewed the core metrics (per-36, BPM, VORP) before choosing the target
- Set up `.venv` and `requirements.txt`; added `data/` to `.gitignore`
- Collected draft classes 2000-2022 (1367 picks) with `src/fetch_draft.py`

## Decisions

### 2026-11-17
- Target: VORP summed over years 3 and 4 (continuous)
  - VORP combines quality and volume, so players who barely play or leave the league get ~0 automatically, which handles busts with no extra rules
  - Categories (starter / rotation / bust) are derived from the VORP target at the end, for communication only
- BPM and per-36 stats are used as complementary features (early-season inputs), not as targets
- Baseline to beat: draft position alone
- Players who left the league before year 3 are kept in the dataset (avoids survivorship bias)
- Draft data comes from the nba_api `DraftHistory` endpoint: it returns the official NBA `player_id`, which links directly to the rookie game logs

## Problems

### 2026-11-17
- Windows Smart App Control blocked a DLL in numpy 2.5.5; pinned `numpy==2.2.6` as a workaround
- Basketball-Reference uses its own player ids, so NBA ids will need to be mapped to BBRef ids before building the target

## Findings

### 2026-11-17
- Some classes have fewer than 60 picks (2000-2004, 2022). This is expected: only 29 teams existed before 2004, and some picks were forfeited