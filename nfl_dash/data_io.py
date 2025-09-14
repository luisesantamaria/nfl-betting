from pathlib import Path
import pandas as pd
import streamlit as st

from .paths import (
    ARCHIVE_DIR, LIVE_DIR,
    season_dir, pnl_path, bets_path, stats_path,
    odds_live_path, odds_archive_path,
)
from .utils import american_to_decimal, norm_abbr

def list_available_seasons() -> list[int]:
    years = []
    if ARCHIVE_DIR.exists():
        for p in sorted(ARCHIVE_DIR.glob("season=*")):
            try:
                y = int(str(p.name).split("=")[-1])
                # considerar season si existe al menos un archivo típico
                if (p / "pnl.csv").exists() or (p / "bets.csv").exists() or (p / "stats.csv").exists():
                    years.append(y)
            except:
                pass
    return sorted(set(years))

@st.cache_data
def load_pnl_weekly(year: int) -> pd.DataFrame:
    fp = pnl_path(year)
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    # normaliza tipos
    for c in ("week", "profit", "stake", "bankroll"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "week_label" not in df.columns:
        def wl(n):
            try:
                n = int(n)
            except:
                return "Week 999"
            if 1 <= n <= 18:
                return f"Week {n}"
            return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
        if "week" in df.columns:
            df["week_label"] = df["week"].apply(wl)
        else:
            df["week_label"] = "Week 999"
    return df

def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

@st.cache_data
def load_ledger(year: int) -> pd.DataFrame:
    fp = _first_existing([bets_path(year)])
    if not fp:
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    # normaliza tipos y nombres de equipo
    for c in ("decimal_odds", "ml", "stake", "profit", "model_prob", "edge", "ev"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "decimal_odds" not in df.columns and "ml" in df.columns:
        df["decimal_odds"] = df["ml"].apply(american_to_decimal)
    for c in ("team","opponent","home_team","away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)
    # week_label derivada si falta
    if "week_label" not in df.columns and "week" in df.columns:
        def wl(n):
            try:
                n = int(n)
            except:
                return "Week 999"
            if 1 <= n <= 18:
                return f"Week {n}"
            return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
        df["week_label"] = df["week"].apply(wl)
    return df

@st.cache_data
def load_bets_this_week(year: int) -> pd.DataFrame:
    # Si algún proceso genera bets en vivo, buscamos primero en live.
    candidates = [
        LIVE_DIR / "bets_this_week.csv",
        season_dir(year) / "this_week.csv",
    ]
    fp = _first_existing(candidates)
    if not fp:
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    for c in ("decimal_odds", "ml", "stake", "model_prob", "edge", "ev"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "decimal_odds" not in df.columns and "ml" in df.columns:
        df["decimal_odds"] = df["ml"].apply(american_to_decimal)
    for c in ("team","opponent"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)
    if "week_label" not in df.columns and "week" in df.columns:
        def wl(n):
            try:
                n = int(n)
            except:
                return "Week 999"
            if 1 <= n <= 18:
                return f"Week {n}"
            return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
        df["week_label"] = df["week"].apply(wl)
    return df

@st.cache_data
def load_odds_for_season(year: int) -> pd.DataFrame:
    # 1) archivo de archivo de la temporada; 2) si no, el live (solo para temporada en curso)
    fp = _first_existing([odds_archive_path(year), odds_live_path()])
    if not fp:
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    for c in ("ml_home","ml_away","decimal_home","decimal_away","spread_home","spread_away","over_under_line",
              "score_home","score_away"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("home_team","away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)
    return df
