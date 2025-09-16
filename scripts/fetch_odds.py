# scripts/fetch_odds.py

import os, time, requests, pandas as pd

API_KEY = os.environ.get("ODDS_API_KEY")  # 👈 ahora usa la misma var que tu workflow
HEADERS = {"Ocp-Apim-Subscription-Key": API_KEY}

BASE_URL = "https://api.sportsdata.io/v3/nfl/odds/json/GameOddsByWeek/{season}/{stype}/{week}"

TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    return TEAM_NAME_TO_ABBR.get(name.strip(), name.strip().upper())

def fetch_week_odds(season: int, stype: int, week: int):
    url = BASE_URL.format(season=season, stype=stype, week=week)
    print(f"🔎 Fetching odds → {url}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"⚠️ API call failed: {url} (status={r.status_code})")
        return []
    data = r.json()
    rows = []
    for g in data:
        odds_list = g.get("PregameOdds") or []
        for o in odds_list:
            rows.append({
                "season": season,
                "week": week if stype == 1 else 18 + week,
                "week_label": f"Week {week}" if stype == 1 else {
                    1: "Wild Card", 2: "Divisional", 3: "Conference", 4: "Super Bowl"
                }.get(week, f"Week {week}"),
                "schedule_date": g.get("Day") or g.get("DateTime"),
                "home_team": normalize_team(g.get("HomeTeam") or ""),
                "away_team": normalize_team(g.get("AwayTeam") or ""),
                "ml_home": o.get("MoneyLineHome"),
                "ml_away": o.get("MoneyLineAway"),
                "spread_home": o.get("PointSpreadHome"),
                "spread_away": o.get("PointSpreadAway"),
                "over_under_line": o.get("OverUnder"),
            })
    return rows

def main():
    season = int(os.environ.get("TARGET_SEASON", 2025))
    week_override = os.environ.get("WEEK")
    out_path = "data/live/odds.csv"

    all_rows = []
    if week_override:  # 👈 respeta el override de la semana
        try:
            week_override = int(week_override)
            print(f"📌 Override → season={season}, week={week_override}")
            all_rows.extend(fetch_week_odds(season, 1, week_override))
        except Exception:
            print(f"⚠️ Invalid WEEK override: {week_override}")
    else:
        for stype, weeks in [(1, range(1,19)), (3, range(1,5))]:
            for wk in weeks:
                all_rows.extend(fetch_week_odds(season, stype, wk))
                time.sleep(0.35)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("⚠️ No odds data retrieved.")
        return

    df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
    df.to_csv(out_path, index=False)
    print(f"✅ Odds saved: {out_path} | Rows: {len(df)}")

if __name__ == "__main__":
    main()
