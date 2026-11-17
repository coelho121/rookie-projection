# CLAUDE.md — Rookie Projection

## About me
- Martim Coelho, 3rd-year Information Management student at NOVA IMS (Lisbon). This is a portfolio project for a career in sports analytics.
- I'm still getting used to this tooling (Git, VS Code, venvs, scraping). Explain the *why* behind each step at a level I could defend in a job interview. Don't just dump code.
- Talk to me in Portuguese (PT-PT), casual tone. Code, comments, notes and the README are in English.
- I'm on Windows. The virtual environment is `.venv` (activate with `.venv\Scripts\activate`).

## The project
**Question:** do a rookie's first ~25 NBA games tell us something about his value in years 3–4 that draft position didn't?

Full decisions and history live in `notes.md`, so read it at the start of every session.

Key decisions so far:
- **Target:** VORP summed over years 3 and 4 (continuous). Starter / rotation / bust categories are derived from it at the end, only to communicate results.
- **Features:** stats from the first ~25 games (per-36, BPM, efficiency, usage, age, etc.), plus draft position.
- **Baseline to beat:** draft position alone.
- **No survivorship bias:** players who left the league before year 3 stay in the dataset (their VORP ≈ 0).
- **No leakage:** features may only use information available after the first ~25 games of the rookie season.

## Data plan
- Draft classes 2000–2022 (all have a completed year 4), with pick number.
- Rookie game logs via `nba_api`, used to compute stats over the first ~25 games.
- Season advanced stats (BPM, VORP) from Basketball-Reference, used for the year 3–4 target.
- Basketball-Reference blocks clients that make more than about 20 requests per minute, so throttle requests (sleep between them). `nba_api` can be slow or time out, so use timeouts and retries.
- Always cache raw downloads in `data/raw/` and never re-download what's already there. Cleaned data goes in `data/processed/`.

## Repo rules
- `data/` is gitignored. Never commit data; the scripts in `src/` must be able to regenerate it.
- Reusable data and cleaning code goes in `src/` (scripts), and exploration and modelling go in `notebooks/`.
- When installing a package, also add it to `requirements.txt`.
- **I do commits and pushes myself in GitHub Desktop.** Don't run `git commit` or `git push`. When a step is done, suggest a short commit message.

## Notes and README
- `notes.md` has four sections: **Flow, Decisions, Problems, Findings**. Each main topic gets a date heading (`### YYYY-MM-DD`), but subtopics don't.
- When something worth recording happens (a decision, a problem, a finding), propose the new lines per section and add them to `notes.md` once I confirm.
- The README is only written at the very end, from the notes. Until then it stays as a one-line "work in progress".