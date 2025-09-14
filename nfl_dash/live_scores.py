import time
import requests
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Connection": "keep-alive",
}

# endpoint principal y fallback
ENDPOINTS = [
    "https://site.web.api.espn.com/apis/v2/sports/football/nfl/scoreboard",
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
]


def _safe_get(params: dict, timeout: int = 20, retries: int = 3):
    """GET con headers + reintentos/backoff y fallback de dominio."""
    last_err = None
    for attempt in range(retries):
        for base in ENDPOINTS:
            try:
                r = requests.get(base, params=params, headers=HEADERS, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (403, 429, 503):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                # otros códigos -> probamos siguiente endpoint
            except requests.RequestException as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
                continue
    return None


def _norm(s):
    return str(s).strip() if s is not None else ""


def fetch_espn_scoreboard_df(season: int, week: int) -> pd.DataFrame:
    """
    Devuelve un DataFrame con TODOS los juegos de esa semana:
    columnas: season, week, start_time, state(pre/in/post), short, home_team, home_score, away_team, away_score
    """
    params = {"seasontype": 2, "week": int(week), "dates": int(season)}
    data = _safe_get(params=params, timeout=20, retries=3)
    if not data:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "start_time",
                "state",
                "short",
                "home_team",
                "home_score",
                "away_team",
                "away_score",
            ]
        )

    rows = []
    for ev in data.get("events", []) or []:
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]

        st_type = (comp.get("status") or {}).get("type") or {}
        state = _norm(st_type.get("state")).lower()  # pre | in | post
        short = _norm(st_type.get("shortDetail"))
        start_time = _norm(comp.get("date"))

        home_team = away_team = None
        home_score = away_score = None
        for c in comp.get("competitors", []) or []:
            team = (c.get("team") or {})
            name = team.get("displayName") or team.get("name")
            score = c.get("score")
            try:
                score = int(score) if score is not None else None
            except Exception:
                score = None
            if c.get("homeAway") == "home":
                home_team, home_score = name, score
            elif c.get("homeAway") == "away":
                away_team, away_score = name, score

        if home_team and away_team:
            rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "start_time": pd.to_datetime(start_time, errors="coerce", utc=True),
                    "state": state,
                    "short": short,
                    "home_team": home_team,
                    "home_score": home_score,
                    "away_team": away_team,
                    "away_score": away_score,
                }
            )

    cols = [
        "season",
        "week",
        "start_time",
        "state",
        "short",
        "home_team",
        "home_score",
        "away_team",
        "away_score",
    ]
    return pd.DataFrame(rows, columns=cols)
