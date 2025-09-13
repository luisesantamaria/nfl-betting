import pandas as pd
import streamlit as st
from pathlib import Path
from .paths import PORTFOLIO_DIR, ARCHIVE_DIR, BETSWEEK_DIR
from .config import SEASON_RULES
from .utils import norm_abbr, american_to_decimal

def list_available_seasons():
    seasons = []
    for f in sorted(PORTFOLIO_DIR.glob("pnl_weekly_*.csv")):
        try: seasons.append(int(f.stem.split("_")[-1]))
        except: pass
    for y in SEASON_RULES:
        seasons.append(y)
    return sorted(set(seasons))

@st.cache_data
def load_pnl_weekly(year: int) -> pd.DataFrame:
    f = PORTFOLIO_DIR / f"pnl_weekly_{year}.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
    for col in ("week", "profit", "stake", "bankroll"):
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    if "week_label" not in df.columns:
        df["week_label"] = "Week 999"
    return df

def find_ledger_path(year: int) -> Path | None:
    season_dir = ARCHIVE_DIR / f"season={year}"
    if not season_dir.exists(): return None
    candidates = [*season_dir.glob("bets_ledger*.csv"), season_dir / "ledger.csv", *season_dir.glob("*.csv")]
    for p in candidates:
        if p.exists(): return p
    return None

@st.cache_data
def load_ledger(year: int) -> pd.DataFrame:
    p = find_ledger_path(year)
    if not p: return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    for col in ("decimal_odds", "ml", "stake", "profit"):
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    if "decimal_odds" not in df.columns and "ml" in df.columns:
        df["decimal_odds"] = df["ml"].apply(american_to_decimal)
    for c in ("team","opponent","home_team","away_team"):
        if c in df.columns: df[c] = df[c].astype(str).map(norm_abbr)
    return df

@st.cache_data
def load_bets_this_week(year: int) -> pd.DataFrame:
    candidates = [BETSWEEK_DIR / f"season={year}" / "this_week.csv", BETSWEEK_DIR / "this_week.csv"]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            for col in ("decimal_odds", "ml", "stake", "model_prob", "edge", "ev"):
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
            if "week_label" not in df.columns and "week" in df.columns:
                def week_label_from_num(n):
                    try: n = int(n)
                    except: return "Week 999"
                    if 1 <= n <= 18: return f"Week {n}"
                    return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
                df["week_label"] = df["week"].apply(week_label_from_num)
            for c in ("team","opponent"):
                if c in df.columns: df[c] = df[c].astype(str).map(norm_abbr)
            return df
    return pd.DataFrame()
