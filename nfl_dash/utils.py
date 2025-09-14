from __future__ import annotations
import math
from typing import Dict, Any
import pandas as pd

# Config opcional (fechas de temporada). Si no hay reglas, calculamos defaults.
try:
    from .config import SEASON_RULES
except Exception:
    SEASON_RULES: Dict[int, Dict[str, Any]] = {}

# -------------------------------------------------
# Orden y etiquetas de semanas
# -------------------------------------------------
ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + [
    "Wild Card", "Divisional", "Conference", "Super Bowl"
]
ORDER_INDEX: Dict[str, int] = {lab: i for i, lab in enumerate(ORDER_LABELS)}

def week_label_from_num(n) -> str:
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

# -------------------------------------------------
# Moneyline → decimal odds
# -------------------------------------------------
def american_to_decimal(ml) -> float | None:
    try:
        x = float(ml)
    except Exception:
        return None
    if x == 0 or (hasattr(math, "isnan") and math.isnan(x)):
        return None
    if x > 0:
        return 1.0 + (x / 100.0)
    return 1.0 + (100.0 / abs(x))

# -------------------------------------------------
# Normalización de equipos (abreviaturas)
# -------------------------------------------------
_ABBR_MAP = {
    # AFC
    "baltimore ravens": "BAL", "buffalo bills": "BUF", "cincinnati bengals": "CIN",
    "cleveland browns": "CLE", "denver broncos": "DEN", "houston texans": "HOU",
    "indianapolis colts": "IND", "jacksonville jaguars": "JAX", "kansas city chiefs": "KC",
    "las vegas raiders": "LV", "los angeles chargers": "LAC", "miami dolphins": "MIA",
    "new england patriots": "NE", "new york jets": "NYJ", "pittsburgh steelers": "PIT",
    "tennessee titans": "TEN",
    # NFC
    "arizona cardinals": "ARI", "atlanta falcons": "ATL", "carolina panthers": "CAR",
    "chicago bears": "CHI", "dallas cowboys": "DAL", "detroit lions": "DET",
    "green bay packers": "GB", "los angeles rams": "LAR", "minnesota vikings": "MIN",
    "new orleans saints": "NO", "new york giants": "NYG", "philadelphia eagles": "PHI",
    "san francisco 49ers": "SF", "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB", "washington commanders": "WAS",
    # históricos / variantes
    "washington football team": "WAS", "oakland raiders": "LV",
    "san diego chargers": "LAC", "st. louis rams": "LAR", "st louis rams": "LAR",
}
def norm_abbr(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).strip().lower()
    abbr = _ABBR_MAP.get(key)
    if abbr:
        return abbr
    # Si ya viene abreviado (<=4) la dejamos como está
    return name if len(str(name)) <= 4 else name

# -------------------------------------------------
# Logos (proxy para el módulo logos.py)
# -------------------------------------------------
def team_logo(name: str, size: int = 48) -> str | None:
    try:
        from .logos import get_logo_url as _get
    except Exception:
        try:
            from .logos import logo_url as _get  # fallback si el nombre difiere
        except Exception:
            return None
    abbr = norm_abbr(name)
    return _get(abbr if abbr else name, size=size)

# -------------------------------------------------
# Capa de conveniencia para carga de datos
# (Overview los importa desde utils)
# -------------------------------------------------
def load_pnl(year: int) -> pd.DataFrame:
    """
    Intenta usar nfl_dash.data_io.load_pnl (nuevo). Si no existe,
    cae a load_pnl_weekly (legacy).
    """
    try:
        from .data_io import load_pnl as _lp
        return _lp(year)
    except Exception:
        try:
            from .data_io import load_pnl_weekly as _lpw
            return _lpw(year)
        except Exception:
            return pd.DataFrame()

def load_bets(year: int) -> pd.DataFrame:
    """
    Proxy a nfl_dash.data_io.load_ledger (nuestro histórico de apuestas).
    """
    try:
        from .data_io import load_ledger as _ll
        return _ll(year)
    except Exception:
        return pd.DataFrame()

def load_scores_for_bets(year: int) -> pd.DataFrame:
    """
    Proxy al módulo odds_scores.py si existe; si no, devuelve DF vacío.
    """
    try:
        from .odds_scores import load_scores_for_bets as _lsfb
        return _lsfb(year)
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------
# Estado de temporada (para mostrar "In Season", etc.)
# -------------------------------------------------
def _first_monday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    while d.weekday() != 0:  # Monday
        d += pd.Timedelta(days=1)
    return d

def _first_thursday_on_or_after(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts
    while d.weekday() != 3:  # Thursday
        d += pd.Timedelta(days=1)
    return d

def _second_sunday_feb(year: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=2, day=1, tz="UTC")
    while d.weekday() != 6:  # Sunday
        d += pd.Timedelta(days=1)
    d += pd.Timedelta(days=7)  # second Sunday
    return d.replace(hour=23, minute=59, second=59)

def _default_bounds(year: int) -> Dict[str, pd.Timestamp]:
    labor_day = _first_monday(year, 9)
    kickoff_thu = _first_thursday_on_or_after(labor_day)
    regular_start = kickoff_thu.normalize()
    post_end = _second_sunday_feb(year + 1)
    return {"regular_start": regular_start, "post_end": post_end}

def season_stage(year: int, now_utc: pd.Timestamp | None = None) -> str:
    if now_utc is None:
        now_utc = pd.Timestamp.utcnow().tz_localize("UTC")

    rules: Dict[str, Any] = SEASON_RULES.get(year, {})
    if "regular_start" in rules and pd.notna(rules["regular_start"]):
        regular_start = pd.to_datetime(rules["regular_start"], utc=True)
    else:
        regular_start = _default_bounds(year)["regular_start"]

    if "post_end" in rules and pd.notna(rules["post_end"]):
        post_end = pd.to_datetime(rules["post_end"], utc=True)
    else:
        post_end = _default_bounds(year)["post_end"]

    if now_utc < regular_start:
        return "Preseason"
    if now_utc <= post_end:
        return "In Season"
    return "Season Ended"
