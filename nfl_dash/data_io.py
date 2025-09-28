# nfl_dash/data_io.py
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .paths import ARCHIVE_DIR, BETSWEEK_DIR
from .utils import norm_abbr, american_to_decimal, season_stage

ET = ZoneInfo("America/New_York")
LIVE_DIR = Path("data/live")


def _labor_day_et(year: int) -> datetime:
    d = datetime(year, 9, 1, tzinfo=ET)
    while d.weekday() != 0:
        d += timedelta(days=1)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def _week1_tnf_et(year: int) -> datetime:
    labor = _labor_day_et(year)
    thu = labor + timedelta(days=3)
    return thu.replace(hour=20, minute=20)

def _current_season_year(now_utc: datetime | None = None) -> int:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    yr = now_et.year
    return yr if now_et >= _week1_tnf_et(yr) else (yr - 1)

def list_available_seasons():
    years_from_archive = []
    for d in sorted(ARCHIVE_DIR.glob("season=*")):
        try:
            y = int(str(d.name).split("=")[1])
            years_from_archive.append(y)
        except Exception:
            continue

    candidates = set(years_from_archive)
    candidates.add(_current_season_year())

    out = []
    for y in sorted(candidates):
        season_dir = ARCHIVE_DIR / f"season={y}"
        has_bets = (season_dir / "bets.csv").exists()
        has_pnl = (season_dir / "pnl.csv").exists()
        if has_bets or has_pnl:
            out.append(y)
            continue

        pnl_df = load_pnl_weekly(y)
        stg = season_stage(y, pnl_df)
        if stg in {"preseason", "in_season"}:
            out.append(y)

    return sorted(set(out))


@st.cache_data(ttl=60)
def load_pnl_weekly(year: int) -> pd.DataFrame:
    """
    PnL semanal:
      - Pasadas:  data/archive/season=YYYY/pnl.csv
      - Actual:   data/live/pnl.csv (filtrado por season) si no hay en archive
    """
    f_arch = ARCHIVE_DIR / f"season={year}" / "pnl.csv"
    f_live = LIVE_DIR / "pnl.csv"

    if f_arch.exists():
        df = pd.read_csv(f_arch)
    elif f_live.exists():
        df = pd.read_csv(f_live)
        if "season" in df.columns:
            df = df[pd.to_numeric(df["season"], errors="coerce").eq(year)].copy()
    else:
        return pd.DataFrame()

    for col in ("week", "profit", "stake", "bankroll", "yield_%"):
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
            return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
        if "week" in df.columns:
            df["week_label"] = df["week"].apply(_lab)
        else:
            df["week_label"] = "Week 999"

    if "updated_at_utc" in df.columns:
        df["updated_at_utc"] = pd.to_datetime(df["updated_at_utc"], errors="coerce", utc=True)

    if "week_order" in df.columns:
        df = df.sort_values(["season", "week_order", "week"], kind="stable").reset_index(drop=True)

    return df


def _bets_path(year: int) -> Path:
    return ARCHIVE_DIR / f"season={year}" / "bets.csv"


@st.cache_data(ttl=60)
def load_ledger(year: int) -> pd.DataFrame:
    """
    Ledger de TODA la temporada:
      - Pasadas:  data/archive/season=YYYY/bets.csv
      - Actual:   data/live/bets.csv filtrado por season (si no hay en archive)
    """
    p_arch = _bets_path(year)
    p_live = LIVE_DIR / "bets.csv"

    if p_arch.exists():
        df = pd.read_csv(p_arch, low_memory=False)
    elif p_live.exists():
        df = pd.read_csv(p_live, low_memory=False)
        if "season" in df.columns:
            df["season"] = pd.to_numeric(df["season"], errors="coerce")
            df = df[df["season"].eq(year)].copy()
    else:
        return pd.DataFrame()

    # Normalizaciones
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

    # week_label si falta
    if "week_label" not in df.columns and "week" in df.columns:
        def _lab(n):
            try:
                n = int(n)
            except Exception:
                return "Week 999"
            if 1 <= n <= 18:
                return f"Week {n}"
            return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
        df["week_label"] = df["week"].apply(_lab)

    return df


@st.cache_data(ttl=60)
def load_bets_this_week(year: int) -> pd.DataFrame:
    """
    Bets de ESTA semana:
      1) data/bets_week/season=YYYY/this_week.csv
      2) data/bets_week/this_week.csv
      3) data/live/bets.csv -> mayor week_order o week
    """
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
                    return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
                df["week_label"] = df["week"].apply(week_label_from_num)
            for c in ("team", "opponent"):
                if c in df.columns:
                    df[c] = df[c].astype(str).map(norm_abbr)
            if "schedule_date" in df.columns:
                df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
            return df

    live_bets = LIVE_DIR / "bets.csv"
    if not live_bets.exists():
        return pd.DataFrame()

    df = pd.read_csv(live_bets, low_memory=False)

    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        df = df[df["season"] == year].copy()
    if df.empty:
        return df

    if "week_order" in df.columns:
        w = int(pd.to_numeric(df["week_order"], errors="coerce").max())
        df = df[df["week_order"] == w].copy()
    elif "week" in df.columns:
        w = int(pd.to_numeric(df["week"], errors="coerce").max())
        df = df[df["week"] == w].copy()

    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
        df = df.sort_values("schedule_date")

    for col in ("decimal_odds", "ml", "stake", "model_prob", "edge", "ev"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for c in ("team", "opponent"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)

    if "week_label" not in df.columns and "week" in df.columns:
        def week_label_from_num(n):
            try:
                n = int(n)
            except Exception:
                return "Week 999"
            if 1 <= n <= 18:
                return f"Week {n}"
            return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
        df["week_label"] = df["week"].apply(week_label_from_num)

    return df
