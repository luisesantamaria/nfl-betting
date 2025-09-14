# nfl_dash/utils.py

from __future__ import annotations
from datetime import date, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd

# ----------------------------
# Semanas (orden & etiquetas)
# ----------------------------
ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX: Dict[str, int] = {lab: i for i, lab in enumerate(ORDER_LABELS)}

def week_label_from_num(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return "Week 999"
    if 1 <= n <= 18:
        return f"Week {n}"
    return {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}.get(n, f"Week {n}")

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

# ----------------------------
# Odds utils
# ----------------------------
def american_to_decimal(m: Optional[float]) -> float:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return np.nan
    m = float(m)
    return 1.0 + (100.0 / abs(m) if m < 0 else m / 100.0)

def decimal_to_american(d: Optional[float]) -> float:
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return np.nan
    d = float(d)
    return round((d - 1.0) * 100.0, 0) if d >= 2.0 else round(-100.0 / (d - 1.0), 0)

# ----------------------------
# Normalización de nombres/equipos
# ----------------------------
_TEAM_FIX = {
    # Abreviaturas históricas → actuales
    "STL": "LA", "LAR": "LA", "LA RAMS": "LA", "ST. LOUIS RAMS": "LA",
    "SD": "LAC", "SDG": "LAC", "SAN DIEGO CHARGERS": "LAC", "LA CHARGERS": "LAC",
    "OAK": "LV", "LVR": "LV", "OAKLAND RAIDERS": "LV", "LAS VEGAS RAIDERS": "LV",
    "WSH": "WAS", "WASHINGTON FOOTBALL TEAM": "WAS", "WASHINGTON REDSKINS": "WAS",
    "JAC": "JAX",

    # Abreviaturas de tres letras alternativas
    "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB",

    # Nombres completos → abrev
    "ARIZONA CARDINALS": "ARI", "ATLANTA FALCONS": "ATL", "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF", "CAROLINA PANTHERS": "CAR", "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN", "CLEVELAND BROWNS": "CLE", "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN", "DETROIT LIONS": "DET", "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU", "INDIANAPOLIS COLTS": "IND", "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC", "LOS ANGELES RAMS": "LA", "LOS ANGELES CHARGERS": "LAC",
    "LAS VEGAS RAIDERS": "LV", "MIAMI DOLPHINS": "MIA", "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE", "NEW ORLEANS SAINTS": "NO", "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ", "PHILADELPHIA EAGLES": "PHI", "PITTSBURGH STEELERS": "PIT",
    "SEATTLE SEAHAWKS": "SEA", "SAN FRANCISCO 49ERS": "SF", "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN", "WASHINGTON COMMANDERS": "WAS",

    # Variaciones comunes en feed de odds
    "LA": "LA", "LAC": "LAC", "LV": "LV", "WAS": "WAS",
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF", "CAR": "CAR", "CHI": "CHI",
    "CIN": "CIN", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB": "GB",
    "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KC": "KC", "MIA": "MIA", "MIN": "MIN",
    "NE": "NE", "NO": "NO", "NYG": "NYG", "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT",
    "SEA": "SEA", "SF": "SF", "TB": "TB", "TEN": "TEN",
}

def norm_abbr(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip().upper()
    return _TEAM_FIX.get(t, t)

def team_logo(team: str) -> Optional[str]:
    # No usamos logos locales. Mantener stub para compatibilidad.
    return None

# ----------------------------
# Estado de temporada
# ----------------------------
def _labor_day(year: int) -> date:
    d = date(year, 9, 1)
    while d.weekday() != 0:  # Monday = 0
        d += timedelta(days=1)
    return d

def _super_bowl_sunday(season: int) -> date:
    y = season + 1
    d = date(y, 2, 1)
    while d.weekday() != 6:  # Sunday = 6
        d += timedelta(days=1)
    d += timedelta(days=7)   # segundo domingo de febrero
    return d

def _week1_thursday(season: int) -> pd.Timestamp:
    ld = _labor_day(season)          # primer lunes de septiembre
    thu = ld + timedelta(days=3)     # jueves de esa misma semana
    return pd.Timestamp(thu).tz_localize("UTC")

def season_stage(season: int, pnl: pd.DataFrame | None = None) -> str:
    """
    Devuelve: 'Preseason' | 'In Season' | 'Season Ended'
    Inicio: jueves posterior a Labor Day (UTC).
    Fin: lunes posterior al Super Bowl (UTC).
    Si el PnL contiene 'Super Bowl' en week_label, se marca como terminada.
    """
    now_ts   = pd.Timestamp.utcnow().tz_localize("UTC")
    start_ts = _week1_thursday(season)
    end_ts   = pd.Timestamp(_super_bowl_sunday(season)).tz_localize("UTC") + pd.Timedelta(days=1)

    if pnl is not None and not pnl.empty and "week_label" in pnl.columns:
        labels = set(map(str, pnl["week_label"].dropna().tolist()))
        if "Super Bowl" in labels:
            return "Season Ended"

    if now_ts < start_ts:
        return "Preseason"
    if now_ts >= end_ts:
        return "Season Ended"
    return "In Season"

__all__ = [
    "ORDER_LABELS",
    "ORDER_INDEX",
    "week_label_from_num",
    "week_sort_key",
    "american_to_decimal",
    "decimal_to_american",
    "norm_abbr",
    "team_logo",
    "season_stage",
]
