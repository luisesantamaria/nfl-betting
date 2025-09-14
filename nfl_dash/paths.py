from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

ARCHIVE_DIR = repo_root() / "data" / "archive"
LIVE_DIR    = repo_root() / "data" / "live"
