import requests
import pandas as pd

def _norm(s):
    return str(s).strip() if s is not None else ""

def fetch_espn_scoreboard_df(season: int, week: int) -> pd.DataFrame:
    url = "https://site.web.api.espn.com/apis/v2/sports/football/nfl/scoreboard"
    params = {"seasontype": 2, "week": int(week), "dates": int(season)}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = []
    for ev in data.get("events", []) or []:
        comps = (ev.get("competitions") or [])
        if not comps:
            continue
        comp = comps[0]
        st_type = (comp.get("status") or {}).get("type") or {}
        state = _norm(st_type.get("state")).lower()
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
            rows.append({
                "season": int(season),
                "week": int(week),
                "start_time": pd.to_datetime(start_time, errors="coerce", utc=True),
                "state": state,              # pre | in | post
                "short": short,              # formatted (e.g., "Q2 12:34", or kickoff ET)
                "home_team": home_team,
                "home_score": home_score,
                "away_team": away_team,
                "away_score": away_score,
            })
    cols = ["season","week","start_time","state","short","home_team","home_score","away_team","away_score"]
    return pd.DataFrame(rows, columns=cols)
