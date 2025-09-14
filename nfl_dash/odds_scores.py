# nfl_dash/odds_scores.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

from .paths import ARCHIVE_DIR, LIVE_DIR

# Importa utilidades, con fallback si faltan
try:
    from .utils import norm_abbr, week_label_from_num
except Exception:
    def norm_abbr(x: str) -> str:
        return (x or "").strip()
    def week_label_from_num(n: int) -> str:
        try:
            n = int(n)
        except Exception:
            return "Week 999"
        if 1 <= n <= 18:
            return f"Week {n}"
        mp = {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}
        return mp.get(n, f"Week {n}")

def _read_odds_for_season(year: int) -> pd.DataFrame:
    arch = ARCHIVE_DIR / f"season={year}" / "odds.csv"
    if arch.exists():
        df = pd.read_csv(arch, low_memory=False)
    else:
        live = LIVE_DIR / "odds.csv"
        if live.exists():
            df = pd.read_csv(live, low_memory=False)
        else:
            return pd.DataFrame()

    for c in ("season", "week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "week_label" not in df.columns and "week" in df.columns:
        df["week_label"] = df["week"].apply(week_label_from_num).astype(str)
    if "week_label" in df.columns:
        df["week_label"] = df["week_label"].astype(str)

    for c in ("home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)

    num_cols = [
        "ml_home","ml_away","decimal_home","decimal_away",
        "ml_home_raw","ml_away_raw","decimal_home_raw","decimal_away_raw",
        "spread_home","spread_away","over_under_line",
        "score_home","score_away",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")

    return df

def _canonicalize_bets_columns(bets: pd.DataFrame) -> pd.DataFrame:
    b = bets.copy()
    if "season" in b.columns:
        b["season"] = pd.to_numeric(b["season"], errors="coerce")

    if "week_label" not in b.columns and "week" in b.columns:
        b["week_label"] = b["week"].apply(week_label_from_num).astype(str)
    if "week_label" in b.columns:
        b["week_label"] = b["week_label"].astype(str)

    for c in ("home_team","away_team","team","opponent"):
        if c in b.columns:
            b[c] = b[c].astype(str).map(norm_abbr)

    if ("home_team" not in b.columns or "away_team" not in b.columns) and \
       ("team" in b.columns and "opponent" in b.columns):
        # Sin ‘side’, asumimos team=home (mejor que nada)
        b["home_team"] = b.get("home_team", b["team"])
        b["away_team"] = b.get("away_team", b["opponent"])
    return b

def _choose_merge_keys(bets: pd.DataFrame, odds: pd.DataFrame) -> list[str]:
    if "event_id" in bets.columns and "event_id" in odds.columns:
        return ["event_id"]
    pref = ["season","week_label","home_team","away_team"]
    k = [c for c in pref if c in bets.columns and c in odds.columns]
    if len(k) >= 3:
        return k
    alt = ["season","week_label","team","opponent"]
    k2 = [c for c in alt if c in bets.columns and c in odds.columns]
    if len(k2) >= 3:
        return k2
    alt2 = ["season","schedule_date","home_team","away_team"]
    k3 = [c for c in alt2 if c in bets.columns and c in odds.columns]
    if len(k3) >= 2:
        return k3
    commons = [c for c in bets.columns if c in odds.columns]
    return commons[:1]

def enrich_bets_with_scores(bets: pd.DataFrame, season: int) -> pd.DataFrame:
    if bets is None or bets.empty:
        return bets
    view = _canonicalize_bets_columns(bets)
    odds = _read_odds_for_season(season)
    if odds.empty:
        return view

    keys = _choose_merge_keys(view, odds)
    want_cols = [
        "ml_home","ml_away","decimal_home","decimal_away",
        "ml_home_raw","ml_away_raw","decimal_home_raw","decimal_away_raw",
        "spread_home","spread_away","over_under_line",
        "score_home","score_away",
        "event_id","schedule_date","week","week_label",
        "home_team","away_team",
    ]
    keep_cols = [c for c in want_cols if c in odds.columns]
    subset = list(dict.fromkeys(keys + keep_cols))

    right = odds[subset].copy()
    merged = view.merge(right, on=keys, how="left", validate="m:1")

    for sc in ("score_home","score_away"):
        if sc not in merged.columns:
            merged[sc] = pd.NA
    return merged
