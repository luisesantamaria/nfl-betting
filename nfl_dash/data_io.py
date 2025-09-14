import pandas as pd
import streamlit as st
from pathlib import Path

from .paths import ARCHIVE_DIR, LIVE_DIR, season_dir
from .utils import norm_abbr, american_to_decimal


def list_available_seasons() -> list[int]:
    years = set()
    for p in ARCHIVE_DIR.glob("season=*"):
        try:
            y = int(p.name.split("=")[1])
            if (p / "pnl.csv").exists() or (p / "bets.csv").exists():
                years.add(y)
        except Exception:
            pass
    return sorted(years)


@st.cache_data
def load_pnl_weekly(year: int) -> pd.DataFrame:
    p = season_dir(year) / "pnl.csv"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p, low_memory=False)

    for c in ("week", "profit", "stake", "bankroll"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "week_label" not in df.columns and "week" in df.columns:
        def week_label_from_num(n):
            try:
                n = int(n)
            except Exception:
                return "Week 999"
            if 1 <= n <= 18:
                return f"Week {n}"
            return {
                19: "Wild Card",
                20: "Divisional",
                21: "Conference",
                22: "Super Bowl",
            }.get(n, f"Week {n}")

        df["week_label"] = df["week"].apply(week_label_from_num)

    return df


def find_ledger_path(year: int) -> Path | None:
    sd = season_dir(year)
    for candidate in [
        sd / "bets.csv",
        sd / "ledger.csv",
        *sd.glob("bets_ledger*.csv"),
        *sd.glob("*.csv"),
    ]:
        if candidate.exists():
            return candidate
    return None


@st.cache_data
def load_ledger(year: int) -> pd.DataFrame:
    p = find_ledger_path(year)
    if not p:
        return pd.DataFrame()

    df = pd.read_csv(p, low_memory=False)

    for col in ("decimal_odds", "ml", "stake", "profit"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "decimal_odds" not in df.columns and "ml" in df.columns:
        df["decimal_odds"] = df["ml"].apply(american_to_decimal)

    for c in ("team", "opponent", "home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)

    return df


@st.cache_data
def load_bets_this_week(year: int) -> pd.DataFrame:
    # Ahora buscamos solo en data/live
    for name in ("this_week.csv", "bets_this_week.csv"):
        p = LIVE_DIR / name
        if p.exists():
            df = pd.read_csv(p, low_memory=False)

            for c in ("decimal_odds", "ml", "stake", "model_prob", "edge", "ev"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            if "week_label" not in df.columns and "week" in df.columns:
                def week_label_from_num(n):
                    try:
                        n = int(n)
                    except Exception:
                        return "Week 999"
                    if 1 <= n <= 18:
                        return f"Week {n}"
                    return {
                        19: "Wild Card",
                        20: "Divisional",
                        21: "Conference",
                        22: "Super Bowl",
                    }.get(n, f"Week {n}")

                df["week_label"] = df["week"].apply(week_label_from_num)

            for c in ("team", "opponent"):
                if c in df.columns:
                    df[c] = df[c].astype(str).map(norm_abbr)

            return df

    return pd.DataFrame()
