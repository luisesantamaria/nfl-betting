from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from .paths import ARCHIVE_DIR, LIVE_DIR
from .utils import norm_abbr, american_to_decimal, week_label_from_num

def list_available_seasons() -> list[int]:
    years = []
    if ARCHIVE_DIR.exists():
        for p in ARCHIVE_DIR.glob("season=*"):
            m = re.search(r"season=(\d{4})$", str(p))
            if m:
                years.append(int(m.group(1)))
    years = sorted(set(years))
    return years if years else [2024]

def _season_dir(year: int) -> Path:
    return ARCHIVE_DIR / f"season={year}"

def load_pnl(year: int) -> pd.DataFrame:
    p = _season_dir(year) / "pnl.csv"
    if not p.exists():
        return pd.DataFrame(columns=["week_label","profit","stake","bankroll"])
    df = pd.read_csv(p, low_memory=False)
    for c in ("profit","stake","bankroll"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "week_label" not in df.columns:
        if "week" in df.columns:
            df["week_label"] = df["week"].apply(week_label_from_num)
        else:
            df["week_label"] = "Week 999"
    return df

def load_bets(year: int) -> pd.DataFrame:
    p = _season_dir(year) / "bets.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    for c in ("decimal_odds","ml","stake","profit","model_prob","edge","ev"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "decimal_odds" not in df.columns and "ml" in df.columns:
        df["decimal_odds"] = df["ml"].apply(american_to_decimal)
    for c in ("team","opponent","home_team","away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)
    if "week_label" not in df.columns and "week" in df.columns:
        df["week_label"] = df["week"].apply(week_label_from_num)
    return df

def load_scores_for_bets(year: int) -> pd.DataFrame:
    p = _season_dir(year) / "odds.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    for c in ("home_team","away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)
    for c in ("score_home","score_away"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["season","week","week_label","schedule_date","home_team","away_team","score_home","score_away","event_id"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates()

def load_bets_this_week(year: int) -> pd.DataFrame:
    p = LIVE_DIR / "this_week.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    for c in ("decimal_odds","ml","stake","model_prob","edge","ev"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "decimal_odds" not in df.columns and "ml" in df.columns:
        df["decimal_odds"] = df["ml"].apply(american_to_decimal)
    for c in ("team","opponent"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)
    if "week_label" not in df.columns and "week" in df.columns:
        df["week_label"] = df["week"].apply(week_label_from_num)
    return df

def load_odds_live() -> pd.DataFrame:
    p = LIVE_DIR / "odds.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    for c in ("home_team","away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)
    return df
