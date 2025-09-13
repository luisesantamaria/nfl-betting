#!/usr/bin/env python3
# scripts/fetch_odds.py
import os, sys, io, math, time, shutil, argparse
import numpy as np
import pandas as pd
import requests
from datetime import timezone
from datetime import datetime as dt

SPORT_KEY = "americanfootball_nfl"
ODDS_BASE = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
SCORES_BASE = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/scores"

REQ_TIMEOUT = 25
DAYS_BACK_SCORES = 14

FULL_TO_ABBR = {
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "LAS VEGAS RAIDERS":"LV","LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LA","MIAMI DOLPHINS":"MIA",
    "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO",
    "NEW YORK GIANTS":"NYG","NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT",
    "SAN FRANCISCO 49ERS":"SF","SEATTLE SEAHAWKS":"SEA","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN",
    "WASHINGTON COMMANDERS":"WAS",
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
    "DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","KC":"KC",
    "LA":"LA","LAC":"LAC","LV":"LV","MIA":"MIA","MIN":"MIN","NE":"NE","NO":"NO","NYG":"NYG","NYJ":"NYJ",
    "PHI":"PHI","PIT":"PIT","SEA":"SEA","SF":"SF","TB":"TB","TEN":"TEN","WAS":"WAS",
    "ST. LOUIS RAMS":"LA","LA RAMS":"LA","SAN DIEGO CHARGERS":"LAC","OAKLAND RAIDERS":"LV",
    "WASHINGTON FOOTBALL TEAM":"WAS","WASHINGTON REDSKINS":"WAS","JAC":"JAX","WSH":"WAS","LVR":"LV","SD":"LAC","SDG":"LAC","LAR":"LA"
}

def to_abbr(name: str) -> str:
    if not name: return ""
    s = str(name).upper().strip()
    return FULL_TO_ABBR.get(s, s)

def american_to_decimal(m):
    if m is None or (isinstance(m, float) and math.isnan(m)): return np.nan
    m = float(m)
    return 1 + (100/abs(m) if m < 0 else m/100)

def nfl_season_from_ts(ts_utc: pd.Timestamp) -> int:
    y = ts_utc.year
    return y - 1 if ts_utc.month <= 2 else y

def week_label_from_num(n: int) -> str:
    if 1 <= n <= 18: return f"Week {n}"
    return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(int(n), f"Week {int(n)}")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def get_env_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("ERROR: ODDS_API_KEY env var is missing.", file=sys.stderr)
        return ""
    return key

def safe_get(url, params):
    try:
        r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            print(f"[warn] GET {url} -> {r.status_code} {r.text[:180]}")
            return None
        return r
    except Exception as e:
        print(f"[warn] GET {url} failed: {e}")
        return None

def fetch_odds(markets="h2h,spreads,totals", regions="us", odds_format="american", date_format="iso") -> list[dict]:
    key = get_env_key()
    if not key:
        return []
    params = {
        "apiKey": key, "regions": regions, "markets": markets,
        "oddsFormat": odds_format, "dateFormat": date_format,
    }
    r = safe_get(ODDS_BASE, params)
    if not r:
        return []
    try:
        data = r.json()
    except Exception as e:
        print(f"[warn] odds json parse error: {e}")
        return []
    return data if isinstance(data, list) else []

def fetch_scores(days_from=DAYS_BACK_SCORES, date_format="iso") -> list[dict]:
    key = get_env_key()
    if not key:
        return []
    params = {"apiKey": key, "daysFrom": days_from, "dateFormat": date_format}
    r = safe_get(SCORES_BASE, params)
    if not r:
        return []
    try:
        data = r.json()
    except Exception as e:
        print(f"[warn] scores json parse error: {e}")
        return []
    return data if isinstance(data, list) else []

def aggregate_markets_one_event(ev: dict) -> dict:
    event_id = ev.get("id","")
    home_nm  = ev.get("home_team","")
    away_nm  = ev.get("away_team","")
    start_iso= ev.get("commence_time","")
    start_ts = pd.to_datetime(start_iso, utc=True, errors="coerce")

    ml_home, ml_away = [], []
    sp_home, sp_away = [], []
    ou_points = []

    for book in (ev.get("bookmakers") or []):
        for m in (book.get("markets") or []):
            key = m.get("key")
            outcomes = m.get("outcomes") or []
            if key == "h2h":
                for o in outcomes:
                    nm = to_abbr(o.get("name"))
                    price = pd.to_numeric(o.get("price"), errors="coerce")
                    if nm == to_abbr(home_nm):
                        ml_home.append(price)
                    elif nm == to_abbr(away_nm):
                        ml_away.append(price)
            elif key == "spreads":
                for o in outcomes:
                    nm = to_abbr(o.get("name"))
                    pt = pd.to_numeric(o.get("point"), errors="coerce")
                    if nm == to_abbr(home_nm):
                        sp_home.append(pt)
                    elif nm == to_abbr(away_nm):
                        sp_away.append(pt)
            elif key == "totals":
                for o in outcomes:
                    pt = pd.to_numeric(o.get("point"), errors="coerce")
                    if not pd.isna(pt):
                        ou_points.append(pt)

    def med_or_nan(arr):
        arr = [float(x) for x in arr if pd.notna(x)]
        return float(np.median(arr)) if arr else np.nan

    mlh = med_or_nan(ml_home)
    mla = med_or_nan(ml_away)
    sph = med_or_nan(sp_home)
    spa = med_or_nan(sp_away)
    oul = med_or_nan(ou_points)

    row = {
        "event_id": event_id,
        "schedule_date": start_ts,
        "home_team": to_abbr(home_nm),
        "away_team": to_abbr(away_nm),
        "ml_home": mlh if not math.isnan(mlh) else np.nan,
        "ml_away": mla if not math.isnan(mla) else np.nan,
        "decimal_home": american_to_decimal(mlh) if not math.isnan(mlh) else np.nan,
        "decimal_away": american_to_decimal(mla) if not math.isnan(mla) else np.nan,
        "spread_home": sph if not math.isnan(sph) else np.nan,
        "spread_away": spa if not math.isnan(spa) else np.nan,
        "over_under_line": oul if not math.isnan(oul) else np.nan,
    }
    if pd.notna(start_ts):
        row["season"] = nfl_season_from_ts(start_ts)
    else:
        now = pd.Timestamp.now(tz=timezone.utc)
        row["season"] = nfl_season_from_ts(now)
    return row

def scores_to_map(scores_list: list[dict]) -> dict:
    out = {}
    for s in scores_list:
        eid = s.get("id","")
        if not eid: continue
        home = to_abbr(s.get("home_team",""))
        away = to_abbr(s.get("away_team",""))
        sh, sa = np.nan, np.nan
        for x in (s.get("scores") or []):
            nm = to_abbr(x.get("name",""))
            sc = pd.to_numeric(x.get("score"), errors="coerce")
            if nm == home: sh = sc
            elif nm == away: sa = sc
        out[eid] = (sh, sa, bool(s.get("completed", False)))
    return out

def upsert_by_key(existing: pd.DataFrame, incoming: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if existing is None or existing.empty:
        return incoming.sort_values(keys).reset_index(drop=True)
    merged = existing.merge(incoming, on=keys, how="outer", suffixes=("_old",""))
    for col in incoming.columns:
        if col in keys: continue
        if col in merged.columns and f"{col}_old" in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[f"{col}_old"])
            merged.drop(columns=[f"{col}_old"], inplace=True, errors="ignore")
    if "schedule_date" in merged.columns:
        merged["schedule_date"] = pd.to_datetime(merged["schedule_date"], utc=True, errors="coerce")
        merged = merged.sort_values(["season","schedule_date","home_team","away_team"], na_position="last")
    else:
        merged = merged.sort_values(["season","home_team","away_team"])
    return merged.reset_index(drop=True)

def enrich_week_labels(df: pd.DataFrame) -> pd.DataFrame:
    try:
        seasons = sorted([int(x) for x in df["season"].dropna().unique().tolist()])
        frames = []
        for y in seasons:
            url = f"https://github.com/nflverse/nflverse-data/releases/download/schedules/sched_{y}.csv"
            sched = pd.read_csv(url, low_memory=False)
            cols = [c.lower() for c in sched.columns]
            sched.columns = cols
            hcol = "home_team" if "home_team" in cols else ("team_home" if "team_home" in cols else None)
            acol = "away_team" if "away_team" in cols else ("team_away" if "team_away" in cols else None)
            dcol = "gameday" if "gameday" in cols else ("gametime" if "gametime" in cols else None)
            if not hcol or not acol:
                continue
            sched["schedule_date"] = pd.to_datetime(sched[dcol], errors="coerce", utc=True) if dcol else pd.NaT
            need = ["season","week","season_type","schedule_date",hcol,acol]
            need = [c for c in need if c in sched.columns]
            frames.append(sched[need].rename(columns={hcol:"home_team", acol:"away_team"}))
        if not frames:
            return df
        S = pd.concat(frames, ignore_index=True)
        S["home_team"] = S["home_team"].astype(str).upper().map(FULL_TO_ABBR).fillna(S["home_team"])
        S["away_team"] = S["away_team"].astype(str).upper().map(FULL_TO_ABBR).fillna(S["away_team"])
        df2 = df.copy()
        df2["d_day"] = pd.to_datetime(df2["schedule_date"], utc=True, errors="coerce").dt.date
        S["d_day"] = pd.to_datetime(S["schedule_date"], utc=True, errors="coerce").dt.date
        M = df2.merge(S[["season","week","d_day","home_team","away_team"]],
                      on=["season","d_day","home_team","away_team"], how="left")
        if "week" in M.columns:
            M["week_label"] = M["week"].apply(lambda x: week_label_from_num(int(x)) if pd.notna(x) else np.nan)
        M.drop(columns=["d_day"], inplace=True, errors="ignore")
        return M
    except Exception as e:
        print(f"[warn] enrich_week_labels failed: {e}")
        return df

def write_empty(live_path: str):
    ensure_dir(os.path.dirname(live_path))
    cols = [
        "season","week","week_label","schedule_date",
        "home_team","away_team",
        "ml_home","ml_away","decimal_home","decimal_away",
        "spread_home","spread_away","over_under_line",
        "score_home","score_away","event_id"
    ]
    pd.DataFrame(columns=cols).to_csv(live_path, index=False)
    print(f"[write-empty] {live_path}")

def maybe_archive_if_season_finished(live_path: str):
    if not os.path.exists(live_path): return
    try:
        df = pd.read_csv(live_path, low_memory=False)
    except Exception:
        return
    if df.empty or "season" not in df.columns or "week" not in df.columns:
        return
    w22 = df[df["week"] == 22]
    if w22.empty: return
    if {"score_home","score_away"}.issubset(df.columns):
        ok = w22["score_home"].notna().all() and w22["score_away"].notna().all()
        if ok:
            season = int(df["season"].dropna().mode().iloc[0])
            arch_dir = os.path.join("data","archive", f"season={season}")
            ensure_dir(arch_dir)
            dst = os.path.join(arch_dir, "odds.csv")
            shutil.move(live_path, dst)
            print(f"[archive] Season {season} finished. Moved → {dst}")
            write_empty(live_path)

def main():
    parser = argparse.ArgumentParser(description="Fetch NFL odds (The Odds API) → data/live/odds.csv + auto-archive on season end.")
    parser.add_argument("--markets", type=str, default="h2h,spreads,totals")
    parser.add_argument("--regions", type=str, default="us")
    parser.add_argument("--days-from", type=int, default=DAYS_BACK_SCORES)
    parser.add_argument("--live-file", type=str, default="data/live/odds.csv")
    args = parser.parse_args()

    live_path = args.live_file
    ensure_dir(os.path.dirname(live_path))

    # 1) intentar traer odds
    events = fetch_odds(markets=args.markets, regions=args.regions)
    if not events:
        print("[info] No odds events returned. Creating empty live file.")
        write_empty(live_path)
        return

    # 2) construir df nuevo (solo temporada actual)
    rows = [aggregate_markets_one_event(e) for e in events]
    df_new = pd.DataFrame(rows)
    if df_new.empty:
        print("[info] Empty odds dataframe after aggregation. Creating empty live file.")
        write_empty(live_path)
        return

    df_new["schedule_date"] = pd.to_datetime(df_new["schedule_date"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz=timezone.utc)
    current_season = (now.year - 1) if now.month <= 2 else now.year
    df_new = df_new[df_new["season"] == current_season].copy()

    # 3) scores
    scores = fetch_scores(days_from=args.days_from)
    m = {}
    try:
        m = {s.get("id",""): (
                pd.to_numeric(next((x.get("score") for x in (s.get("scores") or []) if to_abbr(x.get("name","")) == to_abbr(s.get("home_team",""))), np.nan), errors="coerce"),
                pd.to_numeric(next((x.get("score") for x in (s.get("scores") or []) if to_abbr(x.get("name","")) == to_abbr(s.get("away_team",""))), np.nan), errors="coerce"),
                bool(s.get("completed", False))
            ) for s in scores}
    except Exception as e:
        print(f"[warn] map scores failed: {e}")

    if "event_id" in df_new.columns:
        sh, sa = [], []
        for eid in df_new["event_id"].astype(str):
            tup = m.get(eid)
            if tup:
                sh.append(pd.to_numeric(tup[0], errors="coerce")); sa.append(pd.to_numeric(tup[1], errors="coerce"))
            else:
                sh.append(np.nan); sa.append(np.nan)
        df_new["score_home"] = sh
        df_new["score_away"] = sa

    # 4) enriquecer week/label
    df_new = enrich_week_labels(df_new)

    # 5) upsert
    if os.path.exists(live_path) and os.path.getsize(live_path) > 0:
        old = pd.read_csv(live_path, low_memory=False)
    else:
        old = pd.DataFrame()
    final = upsert_by_key(old, df_new, ["event_id"])

    # 6) columnas + guardar
    col_order = [
        "season","week",

    # 5) Orden/columnas finales
    col_order = [
        "season","week","week_label","schedule_date",
        "home_team","away_team",
        "ml_home","ml_away","decimal_home","decimal_away",
        "spread_home","spread_away","over_under_line",
        "score_home","score_away","event_id"
    ]
    for c in col_order:
        if c not in final.columns:
            final[c] = np.nan
    final = final[col_order].copy()
    final.to_csv(live_path, index=False)
    print(f"[write] {live_path} ({len(final)} rows)")

    # 6) Auto-archive si terminó la temporada
    maybe_archive_if_season_finished(live_path)

if __name__ == "__main__":
    main()
