# nfl_dash/logos.py
from __future__ import annotations
import re

# Base ESPN
_BASE = "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/{slug}.png"

# Normaliza un token (quita espacios, puntos y guiones; uppercase)
def _norm_token(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    s = re.sub(r"[\s\.\-_/]+", "", s)  # quita espaciado y separadores
    return s.upper()

# Slugs oficiales ESPN por equipo
SLUG_BY_TEAM = {
    "ARI": "ari", "ATL": "atl", "BAL": "bal", "BUF": "buf", "CAR": "car", "CHI": "chi",
    "CIN": "cin", "CLE": "cle", "DAL": "dal", "DEN": "den", "DET": "det", "GB": "gb",
    "HOU": "hou", "IND": "ind", "JAX": "jac", "KC": "kc", "LV": "lv", "LAC": "lac",
    "LAR": "lar", "MIA": "mia", "MIN": "min", "NE": "ne", "NO": "no",
    "NYG": "nyg", "NYJ": "nyj", "PHI": "phi", "PIT": "pit", "SEA": "sea",
    "SF": "sf", "TB": "tb", "TEN": "ten", "WSH": "wsh",
}

# Sinónimos comunes de abreviaturas -> canonical (clave de SLUG_BY_TEAM)
ABBR_SYNONYMS = {
    # Cardinals
    "ARZ": "ARI",
    # Jaguars
    "JAC": "JAX",
    # Chiefs
    "KAN": "KC", "KCC": "KC",
    # Packers
    "GNB": "GB", "GBP": "GB",
    # Patriots
    "NWE": "NE", "NEP": "NE",
    # Saints
    "NOR": "NO", "NOS": "NO",
    # Commanders
    "WAS": "WSH", "WFT": "WSH", "WASHINGTON": "WSH",
    # Rams
    "LA": "LAR", "STL": "LAR", "RAMS": "LAR",
    # Chargers
    "SD": "LAC", "SDG": "LAC", "SDC": "LAC",
    # Raiders
    "OAK": "LV", "OAKLAND": "LV",
    # 49ers
    "SFO": "SF", "SFN": "SF", "SF49ERS": "SF",
    # Bucs
    "TAM": "TB", "TBB": "TB",
    # Others duplicates of city names that a veces llegan en CSV
    "PHILA": "PHI",
    "CLV": "CLE",
}

# Nombres completos → canonical abbr
NAME_TO_ABBR = {
    "ARIZONACARDINALS": "ARI",
    "ATLANTAFALCONS": "ATL",
    "BALTIMORERAVENS": "BAL",
    "BUFFALOBILLS": "BUF",
    "CAROLINAPANTHERS": "CAR",
    "CHICAGOBEARS": "CHI",
    "CINCINNATIBENGALS": "CIN",
    "CLEVELANDBROWNS": "CLE",
    "DALLASCOWBOYS": "DAL",
    "DENVERBRONCOS": "DEN",
    "DETROITLIONS": "DET",
    "GREENBAYPACKERS": "GB",
    "HOUSTONTEXANS": "HOU",
    "INDIANAPOLISCOlts".upper(): "IND",
    "JACKSONVILLEJAGUARS": "JAX",
    "KANSASCITYCHIEFS": "KC",
    "LASVEGASRAIDERS": "LV",
    "LOSANGELESCHARGERS": "LAC",
    "LOSANGELESRAMS": "LAR",
    "MIAMIDOLPHINS": "MIA",
    "MINNESOTAVIKINGS": "MIN",
    "NEWENGLANDPATRIOTS": "NE",
    "NEWORLEANSSAINTS": "NO",
    "NEWYORKGIANTS": "NYG",
    "NEWYORKJETS": "NYJ",
    "PHILADELPHIAEAGLES": "PHI",
    "PITTSBURGHSTEELERS": "PIT",
    "SEATTLESEAHAWKS": "SEA",
    "SANFRANCISCO49ERS": "SF",
    "TAMPABAYBUCCANEERS": "TB",
    "TENNESSEETITANS": "TEN",
    "WASHINGTONCOMMANDERS": "WSH",
}

def _abbr_to_slug(token: str) -> str | None:
    """Convierte una abreviatura o sinónimo a slug ESPN."""
    if not token:
        return None
    key = _norm_token(token)
    # Primero: si ya es canonical
    if key in SLUG_BY_TEAM:
        return SLUG_BY_TEAM[key]
    # Segundo: sinónimos -> canonical
    if key in ABBR_SYNONYMS:
        canon = ABBR_SYNONYMS[key]
        return SLUG_BY_TEAM.get(canon)
    return None

def _name_to_slug(name: str) -> str | None:
    """Convierte un nombre completo (o cercano) a slug ESPN."""
    if not name:
        return None
    key = _norm_token(name)
    # Quita “THE”
    key = key.replace("THE", "")
    # NORMALIZACIONES menores
    key = key.replace("SAINTS", "SAINTS")  # placeholder para consistencia
    abbr = NAME_TO_ABBR.get(key)
    if not abbr:
        return None
    return SLUG_BY_TEAM.get(abbr)

def get_logo_url(team: str) -> str | None:
    """
    Acepta abreviaturas (KC, JAX, LAR, WSH, etc.) o nombres completos
    (Kansas City Chiefs, Jacksonville Jaguars, Los Angeles Rams...).
    Devuelve URL del PNG en ESPN o None si no hay match.
    """
    if not team:
        return None

    # 1) Abreviatura directa o sinónimo
    slug = _abbr_to_slug(team)
    if slug:
        return _BASE.format(slug=slug)

    # 2) Intentar con nombre completo
    slug = _name_to_slug(team)
    if slug:
        return _BASE.format(slug=slug)

    # 3) Si viene ya como slug (raro, pero por si acaso)
    t = str(team).lower().strip()
    if t in SLUG_BY_TEAM.values():
        return _BASE.format(slug=t)

    return None
