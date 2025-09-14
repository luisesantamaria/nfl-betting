# nfl_dash/utils.py
from __future__ import annotations
import re
import pandas as pd

# -------------------- Semanas y orden --------------------
ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX  = {lab: i for i, lab in enumerate(ORDER_LABELS)}

def week_label_from_num(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return "Week 999"
    if 1 <= n <= 18:
        return f"Week {n}"
    return {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}.get(n, f"Week {n}")

def week_label_to_num(lbl: str) -> int | None:
    if lbl is None:
        return None
    s = str(lbl).strip()
    if s.startswith("Week "):
        try:
            return int(s.replace("Week ", ""))
        except Exception:
            return None
    mapping = {"Wild Card": 19, "Divisional": 20, "Conference": 21, "Super Bowl": 22}
    return mapping.get(s)

def add_week_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "week_label" in out.columns:
        out["week_label"] = out["week_label"].astype(str)
    elif "week" in out.columns:
        out["week_label"] = out["week"].apply(week_label_from_num).astype(str)
    else:
        out["week_label"] = "Week 999"
    out["__order"] = out["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    return out

def week_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    return add_week_order(df)

# -------------------- Normalización de equipos --------------------
# Mapea nombres completos y sinónimos a abreviaturas estándar.
_NAME2ABBR = {
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "LAS VEGAS RAIDERS":"LV","LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR","MIAMI DOLPHINS":"MIA",
    "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
    "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT","SEATTLE SEAHAWKS":"SEA",
    "SAN FRANCISCO 49ERS":"SF","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WSH",
}

_ABBR_SYNONYMS = {
    "ARZ":"ARI","JAC":"JAX","KAN":"KC","KCC":"KC","GNB":"GB","GBP":"GB","NWE":"NE","NEP":"NE","NOR":"NO","NOS":"NO",
    "WAS":"WSH","WFT":"WSH","LA":"LAR","STL":"LAR","SD":"LAC","SDG":"LAC","SDC":"LAC","OAK":"LV","OAKLAND":"LV",
    "SFO":"SF","SFN":"SF","TAM":"TB","TBB":"TB","PHILA":"PHI","CLV":"CLE",
}

def _norm_token(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"[\.\-_/]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.upper()

def norm_abbr(x: str) -> str:
    """
    Devuelve abreviatura estandarizada (ARI, KC, LAR, WSH, etc.) a partir
    de abbrs o nombres completos.
    """
    if not x:
        return ""
    t = _norm_token(x)

    # si ya es abbr conocida
    if t in _ABBR_SYNONYMS.values():
        return t
    # sinónimos
    if t in _ABBR_SYNONYMS:
        return _ABBR_SYNONYMS[t]

    # nombres completos
    if t in _NAME2ABBR:
        return _NAME2ABBR[t]

    # Heurística: "CITY TEAM" → prueba mapping directo
    if t.replace("THE ", "") in _NAME2ABBR:
        return _NAME2ABBR[t.replace("THE ", "")]

    # último recurso: si es de 2-3 letras, usar como abbr mayúscula
    if 2 <= len(t) <= 3:
        return t
    return t  # deja el texto por si logos.py lo resuelve por nombre

# -------------------- Conversión de cuotas --------------------
def american_to_decimal(ml: float | int | str) -> float | None:
    try:
        m = float(ml)
    except Exception:
        return None
    if m > 0:
        return 1.0 + (m / 100.0)
    elif m < 0:
        return 1.0 + (100.0 / abs(m))
    else:
        return 1.0

def decimal_to_american(dec: float | int | str) -> float | None:
    try:
        d = float(dec)
    except Exception:
        return None
    if d <= 1:
        return 0.0
    if d >= 2:
        return round((d - 1.0) * 100.0, 0)
    else:
        return round(-100.0 / (d - 1.0), 0)

# -------------------- KPIs de PnL --------------------
def kpis_from_pnl(pnl: pd.DataFrame):
    x = pnl.copy()
    for c in ("profit", "stake", "bankroll"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    # inicial = bankroll de la primera fila válida; si no, 1000 (fallback)
    if "bankroll" in x.columns and x["bankroll"].notna().any():
        initial = float(x["bankroll"].dropna().iloc[0])
        final   = float(x["bankroll"].dropna().iloc[-1])
    else:
        initial = 1000.0
        final   = initial + float(x.get("profit", pd.Series([0])).sum())

    total_profit = float(x.get("profit", pd.Series([0])).sum())
    total_stake  = float(x.get("stake",  pd.Series([0])).sum())
    yield_pct    = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0

    profits = x.get("profit", pd.Series([0]*len(x))).fillna(0.0)
    stakes  = x.get("stake",  pd.Series([0]*len(x))).fillna(0.0)

    return initial, final, total_profit, total_stake, yield_pct, profits, stakes

# -------------------- Estado de temporada --------------------
def season_stage(season: int, pnl_df: pd.DataFrame | None = None) -> str:
    """
    Heurística simple: evita dependencias externas y funciona para UI.
    - Antes de agosto del año 'season' → 'preseason'
    - De agosto del año 'season' a febrero del año 'season'+1 → 'in_season'
    - Después de febrero del año 'season'+1 → 'ended'
    Si 'season' < año actual-1 → 'ended'
    """
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    # ended si claramente es de años pasados
    if now.year > season + 1:
        return "ended"
    # ventanas por meses (aprox NFL)
    preseason_start = pd.Timestamp(year=season, month=4, day=1, tz="UTC")   # draft-ish
    regular_start   = pd.Timestamp(year=season, month=8, day=1, tz="UTC")   # agosto
    season_end      = pd.Timestamp(year=season+1, month=3, day=1, tz="UTC") # marzo siguiente

    if now < preseason_start:
        return "locked"
    if now < regular_start:
        return "preseason"
    if now <= season_end:
        # Si ya vemos un Super Bowl en pnl → ended
        if pnl_df is not None and not pnl_df.empty:
            lbls = set(pnl_df.get("week_label", pd.Series(dtype=str)).astype(str))
            if "Super Bowl" in lbls:
                return "ended"
        return "in_season"
    return "ended"
