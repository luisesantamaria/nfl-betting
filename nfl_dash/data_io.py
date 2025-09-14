import pandas as pd
import streamlit as st
from pathlib import Path
from .paths import ARCHIVE_DIR, BETSWEEK_DIR
from .utils import norm_abbr, american_to_decimal, season_stage
from .config import SEASON_RULES

def list_available_seasons():
    years_from_archive = []
    for d in sorted(ARCHIVE_DIR.glob("season=*")):
        try:
            y = int(str(d.name).split("=")[1])
        except Exception:
            continue
        years_from_archive.append(y)

    candidates = sorted(set(years_from_archive).union(set(SEASON_RULES.keys())))
    out = []
    for y in candidates:
        season_dir = ARCHIVE_DIR / f"season={y}"
        has_bets = (season_dir / "bets.csv").exists()
        has_pnl  = (season_dir / "pnl.csv").exists()

        if has_bets or has_pnl:
            out.append(y)
            continue

        pnl_df = load_pnl_weekly(y)  # no falla si no existe
        stg = season_stage(y, pnl_df)
        if stg in {"preseason", "in_season"}:
            out.append(y)

    return sorted(set(out))

@st.cache_data
def load_pnl_weekly(year: int) -> pd.DataFrame:
    f = ARCHIVE_DIR / f"season={year}" / "pnl.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    for col in ("week", "profit", "stake", "bankroll"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "week_label" not in df.columns:
        def _lab(n):
            try:
                n = int(n)
            except Exception:
                return "Week 999"
            if 1 <= n <= 18:
                return f"Week {n}"
            return {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}.get(n, f"Week {n}")
        if "week" in df.columns:
            df["week_label"] = df["week"].apply(_lab)
        else:
            df["week_label"] = "Week 999"
    return df

def _bets_path(year: int) -> Path:
    return ARCHIVE_DIR / f"season={year}" / "bets.csv"

@st.cache_data
def load_ledger(year: int) -> pd.DataFrame:
    p = _bets_path(year)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)

    if "season" not in df.columns:
        df["season"] = year
    else:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(year).astype(int)

    for col in ("decimal_odds", "ml", "stake", "profit", "model_prob", "edge", "ev"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "decimal_odds" not in df.columns and "ml" in df.columns:
        df["decimal_odds"] = df["ml"].apply(american_to_decimal)

    for c in ("team", "opponent", "home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)

    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")

    return df

@st.cache_data
def load_bets_this_week(year: int) -> pd.DataFrame:
    candidates = [
        BETSWEEK_DIR / f"season={year}" / "this_week.csv",
        BETSWEEK_DIR / "this_week.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            for col in ("decimal_odds", "ml", "stake", "model_prob", "edge", "ev"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "week_label" not in df.columns and "week" in df.columns:
                def week_label_from_num(n):
                    try:
                        n = int(n)
                    except Exception:
                        return "Week 999"
                    if 1 <= n <= 18:
                        return f"Week {n}"
                    return {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}.get(n, f"Week {n}")
                df["week_label"] = df["week"].apply(week_label_from_num)
            for c in ("team", "opponent"):
                if c in df.columns:
                    df[c] = df[c].astype(str).map(norm_abbr)
            if "schedule_date" in df.columns:
                df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
            return df
    return pd.DataFrame()

