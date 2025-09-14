from pathlib import Path

def resolve_dir(*parts) -> Path:
    # Repo root relativo al archivo actual
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root.joinpath(*parts),
        Path.cwd().joinpath(*parts),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

ARCHIVE_DIR = resolve_dir("data", "archive")
LIVE_DIR    = resolve_dir("data", "live")

def season_dir(year: int) -> Path:
    return ARCHIVE_DIR / f"season={year}"

def odds_live_path() -> Path:
    return LIVE_DIR / "odds.csv"

def odds_archive_path(year: int) -> Path:
    return season_dir(year) / "odds.csv"

def pnl_path(year: int) -> Path:
    return season_dir(year) / "pnl.csv"

def bets_path(year: int) -> Path:
    return season_dir(year) / "bets.csv"

def stats_path(year: int) -> Path:
    return season_dir(year) / "stats.csv"
