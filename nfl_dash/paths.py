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

ARCHIVE_DIR  = resolve_dir("data", "archive")
LIVE_DIR     = resolve_dir("data", "live")
BETSWEEK_DIR = resolve_dir("data", "live")  # si usas this_week.csv en live
