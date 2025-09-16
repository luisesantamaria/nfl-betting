#!/usr/bin/env python3
# scripts/fetch_odds.py
import os
import sys
import argparse
import math
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

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

def american_to_decimal(m):
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return np.nan
    try:
        m = float(m)
    except Exception:
        return np.nan
    return 1.0 + (100.0/abs(m) if m < 0 else m/100.0)

def decimal_to_american(d):
    try:
        d = float(d)
    except Exception:
        return np.nan
    if d <= 1.0 or math.isnan(d):
        return np.nan
    return round((d - 1.0) * 100.0, 0) if d >= 2.0 else round(-100.0 / (d - 1.0), 0)

def best_decimal(prices):
    clean = [american_to_decimal(p) for p in prices if p is not None]
    clean = [c for c in clean if not (isinstance(c, float) and math.isnan(c))]
    return max(clean) if clean else np.nan

def fetch_events(api_key: str, sport: str, regions: str, markets: str) -> list:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    url = ODDS_API_URL.format(sport=sport)
    r = requests.get(url, params=params, timeout=30)
    print(f"[odds] status={r.status_code} x-requests-remaining={r.headers.get('x-requests-remaining')}")
    r.raise_for_status()
    try:
        data = r.json()
    except json.JSONDecodeError:
        print("[odds] JSON decode error; body (trunc):", str(r.text)[:200])
        data = []
    if not isinstance(data, list):
        print("[odds] unexpected body type; body (trunc):", str(data)[:200])
        return []
    print(f"[odds] events returned: {len(data)}")
    return data

def parse_rows(events: list, want_markets: set) -> pd.DataFrame:
    rows = []
    for ev in events:
        event_id = ev.get("id")
        commence = ev.get("commence_time")
        home = normalize_team((ev.get("home_team") or "").strip())
        away = normalize_team((ev.get("away_team") or "").strip())

        ml_home_prices, ml_away_prices = [], []
        spread_home, spread_away = np.nan, np.nan
        total_line = np.nan

        for bk in ev.get("bookmakers", []) or []:
            for mk in bk.get("markets", []) or []:
                k = mk.get("key")
                if k not in want_markets:
                    continue

                if k == "h2h":
                    for oc in mk.get("outcomes", []) or []:
                        name, price = oc.get("name"), oc.get("price")
                        if not name:
                            continue
                        nm = normalize_team(name.strip())
                        if nm == home:
                            ml_home_prices.append(price)
                        elif nm == away:
                            ml_away_prices.append(price)

                elif k == "spreads":
                    for oc in mk.get("outcomes", []) or []:
                        name, point = oc.get("name"), oc.get("point")
                        nm = normalize_team(name.strip()) if name else ""
                        if nm and point is not None:
                            try:
                                if nm == home and pd.isna(spread_home):
                                    spread_home = float(point)
                                elif nm == away and pd.isna(spread_away):
                                    spread_away = float(point)
                            except Exception:
                                pass

                elif k == "totals":
                    for oc in mk.get("outcomes", []) or []:
                        point = oc.get("point")
                        if point is not None and pd.isna(total_line):
                            try:
                                total_line = float(point)
                            except Exception:
                                pass

        dec_home = best_decimal(ml_home_prices)
        dec_away = best_decimal(ml_away_prices)

        rows.append(dict(
            season=np.nan,
            week=np.nan,
            week_label=np.nan,
            schedule_date=commence,
            home_team=home,
            away_team=away,
            ml_home_raw=decimal_to_american(dec_home),
            ml_away_raw=decimal_to_american(dec_away),
            decimal_home_raw=dec_home,
            decimal_away_raw=dec_away,
            ml_home=np.nan,
            ml_away=np.nan,
            decimal_home=np.nan,
            decimal_away=np.nan,
            spread_home=spread_home,
            spread_away=spread_away,
            over_under_line=total_line,
            event_id=event_id
        ))

    return pd.DataFrame(rows) if rows else pd.DataFrame()
