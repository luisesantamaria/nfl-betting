#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
ODDS_LIVE_PATH = REPO_ROOT / "data" / "live" / "odds.csv"

def _norm_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = "".join(ch for ch in s if ch.isalnum())
    return s

def _pair_key(home: str, away: str) -> str:
    a = _norm_key(home); b = _norm_key(away)
    if not a or not b:
        return ""
    return f"{a}_{b}" if a < b else f"{b}_{a}"

def fetch_espn_scoreboard(season: int, week: int) -> pd.DataFrame:
    url = "https://site.web.api.espn.com/apis/v2/sports/football/nfl/scoreboard"
    params = {"seasontype": 2, "week": week, "dates": season}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = []
    for ev in data.get("events", []) or []:
        comps = (ev.get("competitions") or [])
        if not comps: continue
        comp = comps[0]
        st_type = (comp.get("status") or {}).get("type") or {}
        state = (st_type.get("state") or "").lower()
        short = (st_type.get("shortDetail") or "")
        start = comp.get("date")

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
                "season": season,
                "week": week,
                "pair": _pair_key(home_team, away_team),
                "state": state,
                "short": short,
                "start_time": pd.to_datetime(start, errors="coerce", utc=True),
                "home_score": home_score,
                "away_score": away_score,
            })
    cols = ["season","week","pair","state","short","start_time","home_score","away_score"]
    return pd.DataFrame(rows, columns=cols)

def update_odds_with_scores(odds_df: pd.DataFrame) -> pd.DataFrame:
    x = odds_df.copy()
    for col in ("season","week","home_team","away_team"):
        if col not in x.columns:
            x[col] = None
    x["season"] = pd.to_numeric(x["season"], errors="coerce").astype("Int64")
    x["week"]   = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
    x["pair"]   = [_pair_key(h, a) for h, a in zip(x.get("home_team",""), x.get("away_team",""))]

    keys = (
        x.dropna(subset=["season","week"])
         [["season","week"]].drop_duplicates()
         .astype(int).itertuples(index=False, name=None)
    )

    merged_all = x
    for (season, week) in keys:
        live = fetch_espn_scoreboard(season, week)
        if live.empty:
            continue
        keep = ["season","week","pair","home_score","away_score","state","short","start_time"]
        live = live[keep].copy()

        merged = merged_all.merge(live, on=["season","week","pair"], how="left", suffixes=("","_live"))
        if "score_home" not in merged.columns:
            merged["score_home"] = pd.NA
        if "score_away" not in merged.columns:
            merged["score_away"] = pd.NA

        mask_h = merged["home_score"].notna()
        mask_a = merged["away_score"].notna()
        merged.loc[mask_h, "score_home"] = merged.loc[mask_h, "home_score"].astype("Int64")
        merged.loc[mask_a, "score_away"] = merged.loc[mask_a, "away_score"].astype("Int64")

        merged_all = merged.drop(columns=["home_score","away_score"])

    return merged_all

def main():
    if not ODDS_LIVE_PATH.exists():
        print(f"[scores] file not found: {ODDS_LIVE_PATH}")
        return 0
    odds = pd.read_csv(ODDS_LIVE_PATH, low_memory=False)
    if odds.empty:
        print("[scores] odds is empty; nothing to do.")
        return 0
    updated = update_odds_with_scores(odds)
    changed = not updated.equals(odds)
    updated.to_csv(ODDS_LIVE_PATH, index=False)
    print("[scores] wrote live odds (changed)" if changed else "[scores] wrote live odds (no changes)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
