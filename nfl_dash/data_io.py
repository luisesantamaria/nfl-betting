# nfl_dash/data_io.py
import numpy as np
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
    Carga el PnL semanal para la temporada solicitada.
    - Temporadas pasadas: data/archive/season=YYYY/pnl.csv
    - Temporada actual (o si no hay archivo en archive): data/live/pnl.csv (filtrado por season)
    """
    f_arch = ARCHIVE_DIR / f"season={year}" / "pnl.csv"
    f_live = LIVE_DIR / "pnl.csv"

    if f_arch.exists():
        df = pd.read_csv(f_arch)
    elif f_live.exists():
        df = pd.read_csv(f_live)
        if "season" in df.columns:
            df = df[df["season"] == year].copy()
    else:
        return pd.DataFrame()

    # Normalizaciones de tipos
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
            return {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}.get(n, f"Week {n}")
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


@st.cache_data(ttl=60)
def load_bets_this_week(year: int) -> pd.DataFrame:
    """
    Carga las apuestas de la semana vigente para mostrar en Overview.
    Prioridad:
      1) data/bets_week/season=YYYY/this_week.csv
      2) data/bets_week/this_week.csv
      3) Fallback para season actual: data/live/bets.csv (semana más reciente)

    En el fallback, garantiza columnas de marcador:
      - home_score/away_score (mapeando desde score_home/score_away si es necesario)
      - team_score/opponent_score (derivadas si hace falta)
    """
    # 1) y 2) - flujo existente
    candidates = [
        BETSWEEK_DIR / f"season={year}" / "this_week.csv",
        BETSWEEK_DIR / "this_week.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)

            # Normaliza numéricos
            for col in ("decimal_odds", "ml", "stake", "model_prob", "edge", "ev", "profit"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # week_label si falta
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

            # normaliza equipos
            for c in ("team", "opponent"):
                if c in df.columns:
                    df[c] = df[c].astype(str).map(norm_abbr)

            if "schedule_date" in df.columns:
                df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")

            # Si vienen como score_home/score_away, mapea a home_score/away_score
            if {"home_score", "away_score"}.issubset(df.columns) is False and \
               {"score_home", "score_away"}.issubset(df.columns):
                df["home_score"] = pd.to_numeric(df["score_home"], errors="coerce")
                df["away_score"] = pd.to_numeric(df["score_away"], errors="coerce")

            return df

    # 3) Fallback - lee data/live/bets.csv
    live_bets = LIVE_DIR / "bets.csv"
    if not live_bets.exists():
        return pd.DataFrame()

    df = pd.read_csv(live_bets, low_memory=False)

    # Filtra por season
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        df = df[df["season"] == year].copy()
    if df.empty:
        return df

    # Semana vigente: mayor week_order (si existe) o mayor week
    if "week_order" in df.columns:
        w = int(pd.to_numeric(df["week_order"], errors="coerce").max())
        df = df[df["week_order"] == w].copy()
    elif "week" in df.columns:
        w = int(pd.to_numeric(df["week"], errors="coerce").max())
        df = df[df["week"] == w].copy()

    # Orden por kickoff si existe
    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
        df = df.sort_values("schedule_date")

    # Normaliza numéricos mostrados
    for col in ("decimal_odds", "ml", "stake", "model_prob", "edge", "ev", "profit"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normaliza equipos
    for c in ("team", "opponent", "home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)

    # week_label si hace falta
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

    # ---------- Garantizar columnas de marcador que usan las tarjetas ----------
    # 1) home_score/away_score desde score_home/score_away si no existen
    if {"home_score", "away_score"}.issubset(df.columns) is False:
        if {"score_home", "score_away"}.issubset(df.columns):
            df["home_score"] = pd.to_numeric(df["score_home"], errors="coerce")
            df["away_score"] = pd.to_numeric(df["score_away"], errors="coerce")
        else:
            # Si no existen, las creamos (NaN) para que el componente no truene
            if "home_score" not in df.columns:
                df["home_score"] = np.nan
            if "away_score" not in df.columns:
                df["away_score"] = np.nan

    # 2) team_score/opponent_score si falta (derivadas de home/away cuando podamos)
    if {"team_score", "opponent_score"}.issubset(df.columns) is False:
        can_map = {"team", "home_team", "away_team"}.issubset(df.columns)
        if can_map:
            is_team_home = (df["team"] == df["home_team"])
            df["team_score"] = np.where(is_team_home, df["home_score"], df["away_score"])
            df["opponent_score"] = np.where(is_team_home, df["away_score"], df["home_score"])
        else:
            # al menos crea columnas vacías para que el render no falle
            if "team_score" not in df.columns:
                df["team_score"] = np.nan
            if "opponent_score" not in df.columns:
                df["opponent_score"] = np.nan

    return df
