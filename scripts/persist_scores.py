#!/usr/bin/env python
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.utils import norm_abbr

ET = ZoneInfo("America/New_York")
ODDS_PATH = os.path.join("data", "live", "odds.csv")

def _labor_day_et(year: int) -> datetime:
    d = datetime(year, 9, 1, tzinfo=ET)
    while d.weekday() != 0:
        d += timedelta(days=1)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def _week1_tnf_et(year: int) -> datetime:
    labor = _labor_day_et(year)
    thu = labor + timedelta(days=3)
    return thu.replace(hour=20, minute=20)

def _tuesday_anchor_et(year: int) -> datetime:
    tnf = _week1_tnf_et(year)
    week_monday = (tnf - timedelta(days=tnf.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return week_monday + timedelta(days=1)

def autodetect_week(season: int, now_utc: datetime | None = None) -> int:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    anchor = _tuesday_anchor_et(season)
    if now_et < anchor:
        wk = 1
    else:
        wk = int(((now_et - anchor).days // 7) + 1)
    return max(1, min(22, wk))

def _make_pair(home: pd.Series, away: pd.Series) -> pd.Series:
    ha = home.astype(str).map(norm_abbr)
    aa = away.astype(str).map(norm_abbr)
    return np.where(ha < aa, ha + "_" + aa, aa + "_" + ha)

def main():
    if not os.path.exists(ODDS_PATH):
        print(f"[persist] SKIP: {ODDS_PATH} no existe.")
        return

    odds = pd.read_csv(ODDS_PATH, low_memory=False)

    # Asegura columnas finales
    for c in ("score_home", "score_away"):
        if c not in odds.columns:
            odds[c] = pd.NA

    odds["season"] = pd.to_numeric(odds.get("season"), errors="coerce").astype("Int64")
    odds["week"]   = pd.to_numeric(odds.get("week"), errors="coerce").astype("Int64")
    for c in ("home_team", "away_team"):
        if c in odds.columns:
            odds[c] = odds[c].astype(str)

    season_vals = odds["season"].dropna().unique()
    if len(season_vals) == 0:
        print("[persist] SKIP: season vacío en odds.")
        return
    season = int(season_vals[0])
    week = autodetect_week(season)

    try:
        sb = fetch_espn_scoreboard_df(season=season, week=week)
    except Exception as e:
        print(f"[persist] ERROR fetch ESPN: {e}")
        return

    if sb.empty:
        print("[persist] No scoreboard rows para esta semana.")
        return

    sb = sb.copy()
    sb["state"] = sb["state"].astype(str).str.lower()
    finals = sb[sb["state"].eq("post")].copy()
    if finals.empty:
        print("[persist] No hay juegos Final para persistir todavía.")
        return

    finals["home_team"] = finals["home_team"].astype(str)
    finals["away_team"] = finals["away_team"].astype(str)
    finals["pair"] = _make_pair(finals["home_team"], finals["away_team"])
    odds["pair"] = _make_pair(odds["home_team"], odds["away_team"])

    m = odds.merge(
        finals[["season", "week", "pair", "home_team", "away_team", "home_score", "away_score"]],
        on=["season", "week", "pair"],
        how="left",
        suffixes=("", "_sb"),
        validate="m:1"
    )

    same_home = m["home_team"].astype(str).eq(m["home_team_sb"].astype(str))
    sh = np.where(same_home, m["home_score"], m["away_score"])
    sa = np.where(same_home, m["away_score"], m["home_score"])
    sh = pd.to_numeric(sh, errors="coerce")
    sa = pd.to_numeric(sa, errors="coerce")

    m["score_home"] = np.where(pd.notna(sh), sh, m["score_home"])
    m["score_away"] = np.where(pd.notna(sa), sa, m["score_away"])

    # Limpia columnas auxiliares del merge
    drop_cols = [c for c in m.columns if c.endswith("_sb")] + ["home_score", "away_score", "pair"]
    m = m.drop(columns=[c for c in drop_cols if c in m.columns], errors="ignore")

    # Mantén exactamente el orden del odds original (que ya incluye score_* porque las agregamos arriba)
    base_cols = [c for c in odds.columns if c in m.columns]
    out = m[base_cols]

    out.to_csv(ODDS_PATH, index=False)
    print(f"[persist] wrote {ODDS_PATH} | rows={len(out)} at {datetime.now(timezone.utc).isoformat()}")

if __name__ == "__main__":
    main()
