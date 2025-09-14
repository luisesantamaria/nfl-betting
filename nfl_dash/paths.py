from __future__ import annotations
from pathlib import Path

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def resolve_dir(*parts) -> Path:
    candidates = [
        _repo_root().joinpath(*parts),
        Path.cwd().joinpath(*parts),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

ARCHIVE_DIR = resolve_dir("data", "archive")
LIVE_DIR    = resolve_dir("data", "live")

def season_dir(season: int) -> Path:
    return ARCHIVE_DIR.joinpath(f"season={season}")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

ODDS_DIRS = [
    LIVE_DIR,
    ARCHIVE_DIR,
]
