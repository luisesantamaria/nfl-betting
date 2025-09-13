import requests
import streamlit as st
from .paths import LOGOS_DIR

ESPN_SLUG = {
    "ARI":"ari","ATL":"atl","BAL":"bal","BUF":"buf","CAR":"car","CHI":"chi","CIN":"cin","CLE":"cle",
    "DAL":"dal","DEN":"den","DET":"det","GB":"gb","HOU":"hou","IND":"ind","JAX":"jax","KC":"kc",
    "LA":"lar","LAR":"lar","LAC":"lac","LV":"lv","MIA":"mia","MIN":"min","NE":"ne","NO":"no",
    "NYG":"nyg","NYJ":"nyj","PHI":"phi","PIT":"pit","SEA":"sea","SF":"sf","TB":"tb","TEN":"ten",
    "WAS":"wsh","WSH":"wsh"
}

@st.cache_data(ttl=60*60*24)
def get_logo_url(abbr: str) -> str | None:
    if not abbr: return None
    a = abbr.upper().strip()
    local = LOGOS_DIR / f"{a}.png"
    if local.exists(): return local.as_posix()
    candidates = [
        f"https://static.www.nfl.com/t_q-best/league/api/clubs/logos/{a}",
        f"https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/{ESPN_SLUG.get(a, a.lower())}.png",
        f"https://a.espncdn.com/i/teamlogos/nfl/500/{ESPN_SLUG.get(a, a.lower())}.png",
    ]
    for url in candidates:
        try:
            r = requests.head(url, timeout=2, allow_redirects=True)
            if r.status_code == 200: return url
        except Exception:
            continue
    return None
