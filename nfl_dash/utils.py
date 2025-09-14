from __future__ import annotations
import re
from typing import Dict, Optional, Tuple

# -----------------------------
# Normalización de equipos
# -----------------------------

# Abreviaturas “estándar” internas (3 letras, mayúsculas)
_STD_ABBR: Dict[str, str] = {
    # NFC
    "ARIZONA CARDINALS": "ARI", "ARI": "ARI", "ARZ": "ARI",
    "ATLANTA FALCONS": "ATL", "ATL": "ATL",
    "CAROLINA PANTHERS": "CAR", "CAR": "CAR",
    "CHICAGO BEARS": "CHI", "CHI": "CHI",
    "DALLAS COWBOYS": "DAL", "DAL": "DAL",
    "DETROIT LIONS": "DET", "DET": "DET",
    "GREEN BAY PACKERS": "GB", "GNB": "GB", "GB": "GB",
    "LOS ANGELES RAMS": "LAR", "LA RAMS": "LAR", "ST. LOUIS RAMS": "LAR", "LAR": "LAR", "LA": "LAR",
    "MINNESOTA VIKINGS": "MIN", "MIN": "MIN",
    "NEW ORLEANS SAINTS": "NO", "NOR": "NO", "NO": "NO",
    "NEW YORK GIANTS": "NYG", "NYG": "NYG",
    "PHILADELPHIA EAGLES": "PHI", "PHI": "PHI",
    "SAN FRANCISCO 49ERS": "SF", "SFO": "SF", "SF": "SF",
    "SEATTLE SEAHAWKS": "SEA", "SEA": "SEA",
    "TAMPA BAY BUCCANEERS": "TB", "TAM": "TB", "TB": "TB",
    "WASHINGTON COMMANDERS": "WAS", "WASHINGTON FOOTBALL TEAM": "WAS", "WASHINGTON REDSKINS": "WAS", "WSH": "WAS", "WAS": "WAS",

    # AFC
    "BALTIMORE RAVENS": "BAL", "BAL": "BAL",
    "BUFFALO BILLS": "BUF", "BUF": "BUF",
    "CINCINNATI BENGALS": "CIN", "CIN": "CIN",
    "CLEVELAND BROWNS": "CLE", "CLE": "CLE",
    "DENVER BRONCOS": "DEN", "DEN": "DEN",
    "HOUSTON TEXANS": "HOU", "HOU": "HOU",
    "INDIANAPOLIS COLTS": "IND", "IND": "IND",
    "JACKSONVILLE JAGUARS": "JAX", "JAC": "JAX", "JAX": "JAX",
    "KANSAS CITY CHIEFS": "KC", "KAN": "KC", "KC": "KC",
    "LAS VEGAS RAIDERS": "LV", "OAKLAND RAIDERS": "LV", "OAK": "LV", "LVR": "LV", "LV": "LV",
    "LOS ANGELES CHARGERS": "LAC", "LA CHARGERS": "LAC", "SAN DIEGO CHARGERS": "LAC", "SD": "LAC", "SDG": "LAC", "LAC": "LAC",
    "MIAMI DOLPHINS": "MIA", "MIA": "MIA",
    "NEW ENGLAND PATRIOTS": "NE", "NWE": "NE", "NE": "NE",
    "NEW YORK JETS": "NYJ", "NYJ": "NYJ",
    "PITTSBURGH STEELERS": "PIT", "PIT": "PIT",
    "TENNESSEE TITANS": "TEN", "TEN": "TEN",
}

# Código que usa ESPN en su CDN (minúsculas, a veces 2 y a veces 3 letras)
# Ruta típica: https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/{code}.png
_ESPN_CODE: Dict[str, str] = {
    # NFC
    "ARI": "ari", "ATL": "atl", "CAR": "car", "CHI": "chi", "DAL": "dal", "DET": "det",
    "GB": "gb", "LAR": "lar", "MIN": "min", "NO": "no", "NYG": "nyg", "PHI": "phi",
    "SF": "sf", "SEA": "sea", "TB": "tb", "WAS": "wsh",

    # AFC
    "BAL": "bal", "BUF": "buf", "CIN": "cin", "CLE": "cle", "DEN": "den", "HOU": "hou",
    "IND": "ind", "JAX": "jax", "KC": "kc", "LV": "lv", "LAC": "lac", "MIA": "mia",
    "NE": "ne", "NYJ": "nyj", "PIT": "pit", "TEN": "ten",
}

def _clean_team_name(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_abbr(team: str) -> str:
    """Devuelve la abreviatura interna (ARI, LAC, KC, NO, etc.)."""
    s = _clean_team_name(team).upper()
    return _STD_ABBR.get(s, s)

def to_espn_code(team: str) -> Optional[str]:
    """Devuelve el código que usa ESPN en su CDN (minúsculas)."""
    abbr = norm_abbr(team)
    return _ESPN_CODE.get(abbr)

def logo_url(team: str, scoreboard: bool = True, size: int = 500) -> Optional[str]:
    """URL del logo en el CDN de ESPN para un equipo dado."""
    code = to_espn_code(team)
    if not code:
        return None
    sub = "scoreboard" if scoreboard else ""
    if sub:
        return f"https://a.espncdn.com/i/teamlogos/nfl/{size}/scoreboard/{code}.png"
    return f"https://a.espncdn.com/i/teamlogos/nfl/{size}/{code}.png"

# -----------------------------
# Odds helpers
# -----------------------------

def american_to_decimal(m) -> float:
    if m is None:
        return float("nan")
    try:
        m = float(m)
    except Exception:
        return float("nan")
    return 1 + (100 / abs(m) if m < 0 else m / 100.0)

def decimal_to_american(d) -> float:
    if d is None:
        return float("nan")
    try:
        d = float(d)
    except Exception:
        return float("nan")
    return round((d - 1) * 100, 0) if d >= 2.0 else round(-100 / (d - 1), 0)

# -----------------------------
# Weeks helpers
# -----------------------------

ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX  = {lab: i for i, lab in enumerate(ORDER_LABELS)}

def week_label_from_num(n: int) -> str:
    n = int(n)
    if 1 <= n <= 18:
        return f"Week {n}"
    return {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}.get(n, f"Week {n}")
