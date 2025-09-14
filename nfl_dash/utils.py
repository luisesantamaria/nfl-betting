import os
import pandas as pd

ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX = {lab: i for i, lab in enumerate(ORDER_LABELS)}

TEAM_LOGO = {
    "ARI": "assets/logos/ARI.png", "ATL": "assets/logos/ATL.png", "BAL": "assets/logos/BAL.png",
    "BUF": "assets/logos/BUF.png", "CAR": "assets/logos/CAR.png", "CHI": "assets/logos/CHI.png",
    "CIN": "assets/logos/CIN.png", "CLE": "assets/logos/CLE.png", "DAL": "assets/logos/DAL.png",
    "DEN": "assets/logos/DEN.png", "DET": "assets/logos/DET.png", "GB": "assets/logos/GB.png",
    "HOU": "assets/logos/HOU.png", "IND": "assets/logos/IND.png", "JAX": "assets/logos/JAX.png",
    "KC": "assets/logos/KC.png", "LA": "assets/logos/LA.png", "LAC": "assets/logos/LAC.png",
    "LV": "assets/logos/LV.png", "MIA": "assets/logos/MIA.png", "MIN": "assets/logos/MIN.png",
    "NE": "assets/logos/NE.png", "NO": "assets/logos/NO.png", "NYG": "assets/logos/NYG.png",
    "NYJ": "assets/logos/NYJ.png", "PHI": "assets/logos/PHI.png", "PIT": "assets/logos/PIT.png",
    "SEA": "assets/logos/SEA.png", "SF": "assets/logos/SF.png", "TB": "assets/logos/TB.png",
    "TEN": "assets/logos/TEN.png", "WAS": "assets/logos/WAS.png",
}

TEAM_FIX = {"STL": "LA", "LAR": "LA", "SD": "LAC", "SDG": "LAC", "OAK": "LV", "LVR": "LV", "WSH": "WAS", "JAC": "JAX"}

def week_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "week_label" in out.columns:
        out["week_label"] = out["week_label"].astype(str)
    elif "week" in out.columns:
        out["week_label"] = out["week"].apply(week_label_from_num).astype(str)
    else:
        out["week_label"] = "Week 999"
    out["week_order"] = out["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    return out

def week_label_from_num(n: int) -> str:
    n = int(n)
    if 1 <= n <= 18:
        return f"Week {n}"
    return {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}.get(n, f"Week {n}")

def path_bets(season: int) -> str:
    return f"data/archive/season={season}/bets.csv"

def path_pnl(season: int) -> str:
    return f"data/archive/season={season}/pnl.csv"

def path_stats(season: int) -> str:
    return f"data/archive/season={season}/stats.csv"

def path_odds_archive(season: int) -> str:
    return f"data/archive/season={season}/odds.csv"

def path_odds_live() -> str:
    return "data/live/odds.csv"

def load_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, parse_dates=parse_dates or [])

def load_bets(season: int) -> pd.DataFrame:
    df = load_csv_safe(path_bets(season))
    if df.empty:
        return df
    for c in ("schedule_date",):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.upper().str.strip().map(lambda x: TEAM_FIX.get(x, x))
    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].astype(str).str.upper().str.strip().map(lambda x: TEAM_FIX.get(x, x))
    if "week_label" in df.columns:
        df["week_label"] = df["week_label"].astype(str)
    num_cols = ["decimal_odds", "model_prob", "edge", "ev", "stake", "profit"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_pnl(season: int) -> pd.DataFrame:
    df = load_csv_safe(path_pnl(season))
    if df.empty:
        return df
    if "week_label" in df.columns:
        df["week_label"] = df["week_label"].astype(str)
    for c in ("profit", "stake", "bankroll"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_scores_for_bets(season: int) -> pd.DataFrame:
    arch = load_csv_safe(path_odds_archive(season), parse_dates=["schedule_date"])
    if arch.empty:
        live = load_csv_safe(path_odds_live(), parse_dates=["schedule_date"])
        base = live
    else:
        base = arch
    if base.empty:
        return base
    keep = [c for c in [
        "season", "week", "week_label", "schedule_date",
        "home_team", "away_team", "score_home", "score_away"
    ] if c in base.columns]
    df = base[keep].copy()
    for c in ("home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip().map(lambda x: TEAM_FIX.get(x, x))
    return df

def team_logo(team: str) -> str:
    t = str(team).upper().strip()
    t = TEAM_FIX.get(t, t)
    return TEAM_LOGO.get(t, "")
