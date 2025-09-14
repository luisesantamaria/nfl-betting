# nfl_dash/logos.py
from __future__ import annotations

# Mapea tu abreviatura normalizada (p. ej. "JAX", "LAR", "WSH") al slug que usa ESPN
_ESPN_SLUG = {
    "ARI":"ari","ATL":"atl","BAL":"bal","BUF":"buf","CAR":"car","CHI":"chi","CIN":"cin",
    "CLE":"cle","DAL":"dal","DEN":"den","DET":"det","GB":"gb","HOU":"hou","IND":"ind",
    "JAX":"jac","KC":"kc","LV":"lv","LAC":"lac","LAR":"lar","MIA":"mia","MIN":"min",
    "NE":"ne","NO":"no","NYG":"nyg","NYJ":"nyj","PHI":"phi","PIT":"pit","SEA":"sea",
    "SF":"sf","TB":"tb","TEN":"ten","WSH":"wsh",
}

_BASE = "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/{slug}.png"

def get_logo_url(team_abbr: str) -> str | None:
    if not team_abbr:
        return None
    key = str(team_abbr).upper().strip()
    slug = _ESPN_SLUG.get(key)
    if not slug:
        return None
    return _BASE.format(slug=slug)
