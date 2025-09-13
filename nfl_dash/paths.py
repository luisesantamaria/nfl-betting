from pathlib import Path

def resolve_dir(*parts) -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent.joinpath(*parts),  # repo root relative
        Path.cwd().joinpath(*parts),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

# Data dirs
PORTFOLIO_DIR = resolve_dir("data", "processed", "portfolio")
ARCHIVE_DIR   = resolve_dir("data", "archive")
BETSWEEK_DIR  = resolve_dir("data", "processed", "bets")

ODDS_DIRS = [
    resolve_dir("bootstrap"),
    resolve_dir("data", "bootstrap"),
    resolve_dir("data", "processed", "odds"),
    resolve_dir("data"),
]

LOGOS_DIR = resolve_dir("assets", "logos", "nfl")
