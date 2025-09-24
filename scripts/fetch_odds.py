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
    "Arizona Cardinals": "ARI","Atlanta Falcons": "ATL","Baltimore Ravens": "BAL","Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR","Chicago Bears": "CHI","Cincinnati Bengals": "CIN","Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL","Denver Broncos": "DEN","Detroit Lions": "DET","Green Bay Packers": "GB",
    "Houston Texans": "HOU","Indianapolis Colts": "IND","Jacksonville Jaguars": "JAX","Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV","Los Angeles Chargers": "LAC","Los Angeles Rams": "LA","Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN","New England Patriots": "NE","New Orleans Saints": "NO","New York Giants": "NYG",
    "New York Jets": "NYJ","Philadelphia Eagles": "PHI","Pittsburgh Steelers": "PIT","San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA","Tampa Bay Buccaneers": "TB","Tennessee Titans": "TEN","Washington Commanders": "WAS",
}

def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    if name.upper() in TEAM_NAME_TO_ABBR.values():
        return name.upper()
    return TEAM_NAME_TO_ABBR.get(name, name.upper())

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

def fmt_tsZ(ts: pd.Timestamp) -> str:
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_events(api_key: str, sport: str, regions: str, markets: str,
                 commence_from: str | None = None,
                 commence_to: str | None = None) -> list:
    params = {"apiKey": api_key,"regions": regions,"markets": markets,
              "oddsFormat": "american","dateFormat": "iso"}
    if commence_from: params["commenceTimeFrom"] = commence_from
    if commence_to:   params["commenceTimeTo"]   = commence_to
    url = ODDS_API_URL.format(sport=sport)
    print(f"[odds] GET {url} params={params}")
    r = requests.get(url, params=params, timeout=30)
    print(f"[odds] status={r.status_code} x-requests-remaining={r.headers.get('x-requests-remaining')} x-requests-used={r.headers.get('x-requests-used')}")
    if r.status_code != 200:
        print("[odds] body(trunc):", r.text[:400])
    r.raise_for_status()
    try:
        data = r.json()
    except json.JSONDecodeError:
        print("[odds] JSON decode error; body (trunc):", str(r.text)[:400]); data = []
    if not isinstance(data, list):
        print("[odds] unexpected body type; body (trunc):", str(data)[:400]); return []
    print(f"[odds] events returned: {len(data)}")
    if not data: print("[odds] EMPTY payload. Response headers:", dict(r.headers))
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
                if k not in want_markets: continue
                if k == "h2h":
                    for oc in mk.get("outcomes", []) or []:
                        name, price = oc.get("name"), oc.get("price")
                        if not name: continue
                        nm = normalize_team(name.strip())
                        if nm == home: ml_home_prices.append(price)
                        elif nm == away: ml_away_prices.append(price)
                elif k == "spreads":
                    for oc in mk.get("outcomes", []) or []:
                        name, point = oc.get("name"), oc.get("point")
                        nm = normalize_team(name.strip()) if name else ""
                        if nm and point is not None:
                            try:
                                if nm == home and pd.isna(spread_home): spread_home = float(point)
                                elif nm == away and pd.isna(spread_away): spread_away = float(point)
                            except Exception: pass
                elif k == "totals":
                    for oc in mk.get("outcomes", []) or []:
                        point = oc.get("point")
                        if point is not None and pd.isna(total_line):
                            try: total_line = float(point)
                            except Exception: pass

        dec_home = best_decimal(ml_home_prices)
        dec_away = best_decimal(ml_away_prices)

        rows.append(dict(
            season=np.nan, week=np.nan, week_label=np.nan, schedule_date=commence,
            home_team=home, away_team=away,
            ml_home_raw=decimal_to_american(dec_home), ml_away_raw=decimal_to_american(dec_away),
            decimal_home_raw=dec_home, decimal_away_raw=dec_away,
            ml_home=np.nan, ml_away=np.nan, decimal_home=np.nan, decimal_away=np.nan,
            spread_home=spread_home, spread_away=spread_away, over_under_line=total_line,
            price_factor=np.nan, event_id=event_id
        ))
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def compute_season_from_ts(ts_utc: pd.Timestamp) -> int:
    if pd.isna(ts_utc):
        now = datetime.now(timezone.utc)
        return now.year if now.month >= 3 else now.year - 1
    y, m = ts_utc.year, ts_utc.month
    return y if m >= 3 else y - 1

def first_thursday_after_labor_day_ts(year: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=9, day=1, tz="UTC")
    while d.weekday() != 0: d += pd.Timedelta(days=1)  # Labor Day (Mon)
    return (d + pd.Timedelta(days=3)).normalize()      # Thu

LABEL_POST = {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}

def infer_week_fields(ts_utc: pd.Timestamp):
    if pd.isna(ts_utc): return (np.nan, np.nan)
    year = compute_season_from_ts(ts_utc)
    anchor = first_thursday_after_labor_day_ts(year)
    dd = (ts_utc.normalize() - anchor).days
    w = 1 + (dd // 7)
    if w < 1: return (np.nan, np.nan)
    if w <= 18: return (int(w), f"Week {int(w)}")
    k = min(4, int(w) - 18)
    wk = 18 + k
    return (wk, LABEL_POST.get(wk, f"Week {wk}"))

def current_season_week(now_utc: datetime | None = None):
    if now_utc is None: now_utc = datetime.now(timezone.utc)
    ts = pd.Timestamp(now_utc)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    season = compute_season_from_ts(ts)
    w, lbl = infer_week_fields(ts)
    return season, w, lbl

def week_window_utc(year:int, week:int) -> tuple[pd.Timestamp, pd.Timestamp]:
    anchor = first_thursday_after_labor_day_ts(year)
    start = anchor + pd.Timedelta(days=7*(week-1))
    end   = start + pd.Timedelta(days=7)
    return start, end

def apply_price_factor(parsed: pd.DataFrame, factor: float) -> pd.DataFrame:
    out = parsed.copy()
    for side in ("home", "away"):
        dr = f"decimal_{side}_raw"
        if dr in out.columns:
            out[f"decimal_{side}"] = out[dr].astype(float) / float(factor)
            out[f"ml_{side}"] = out[f"decimal_{side}"].apply(decimal_to_american)
        else:
            out[f"decimal_{side}"] = np.nan
            out[f"ml_{side}"] = np.nan
        mr = f"ml_{side}_raw"
        if mr not in out.columns:
            out[mr] = out[f"decimal_{side}"].apply(decimal_to_american)
    out["price_factor"] = float(factor)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-file", type=str, default="data/live/odds.csv")
    ap.add_argument("--sport", type=str, default="americanfootball_nfl")
    ap.add_argument("--regions", type=str, default="us,us2")
    ap.add_argument("--markets", type=str, default="h2h,spreads,totals")
    ap.add_argument("--only-current-week", dest="only_current_week", action="store_true", default=True)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--week-shift", type=int, default=0, help="Shift target week by N (e.g., 1 = next week)")
    ap.add_argument("--price-factor", type=float, default=float(os.environ.get("ODDS_PRICE_FACTOR", "1.022")))
    ap.add_argument("--archive-force", action="store_true", default=False)
    args = ap.parse_args()

    live_path = args.live_file
    os.makedirs(os.path.dirname(live_path), exist_ok=True)

    cur_season, cur_week, cur_label = current_season_week()
    print(f"[odds] now → season={cur_season}, week={cur_week} ({cur_label})")

    if pd.isna(cur_week):
        print("[odds] offseason or preseason detected → skip fetch.")
        return

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        print("[odds] ERROR: ODDS_API_KEY env var is empty.", file=sys.stderr)
        sys.exit(1)

    want_markets = set([m.strip() for m in args.markets.split(",") if m.strip()])

    # Target week/season (+ opcional shift)
    target_season = args.season if args.season is not None else cur_season
    base_week     = args.week if args.week is not None else cur_week
    target_week   = None if pd.isna(base_week) else int(base_week) + int(args.week_shift)
    print(f"[odds] target(base) season={target_season} week={base_week} shift={args.week_shift} → target_week={target_week}")

    # --- función para pedir una ventana y parsear ---
    def fetch_window_for_week(season:int, week:int) -> pd.DataFrame:
        w_start, w_end = week_window_utc(int(season), int(week))
        startZ, endZ = fmt_tsZ(w_start), fmt_tsZ(w_end)
        print(f"[odds] window UTC → start={startZ} end={endZ}")
        ev = fetch_events(api_key, args.sport, args.regions, args.markets, startZ, endZ)
        df = parse_rows(ev, want_markets)
        return df

    # 1) pide la semana target
    if target_week is None:
        events = fetch_events(api_key, args.sport, args.regions, args.markets)
        df_new = parse_rows(events, want_markets)
    else:
        df_new = fetch_window_for_week(target_season, target_week)

        # 2) fallback: si viene vacío y estamos en días previos al kickoff, intenta week+1
        if df_new.empty and args.only_current_week:
            print(f"[odds] no events for week {target_week}; trying fallback week {target_week+1}")
            try:
                df_fallback = fetch_window_for_week(target_season, target_week + 1)
                if not df_fallback.empty:
                    print(f"[odds] fallback added rows: {len(df_fallback)} (week {target_week+1})")
                    df_new = df_fallback
                    # importa que etiquetemos bien luego con infer_week_fields
            except Exception as e:
                print(f"[odds] fallback error: {e}")

    base_cols = [
        "season","week","week_label","schedule_date","home_team","away_team",
        "ml_home","ml_away","decimal_home","decimal_away",
        "ml_home_raw","ml_away_raw","decimal_home_raw","decimal_away_raw",
        "spread_home","spread_away","over_under_line","price_factor","event_id"
    ]

    if df_new.empty:
        parsed = pd.DataFrame(columns=base_cols)
    else:
        parsed = df_new.copy()
        parsed["schedule_date"] = pd.to_datetime(parsed["schedule_date"], errors="coerce", utc=True)
        parsed["season"] = parsed["schedule_date"].apply(compute_season_from_ts).astype("Int64")
        weeks = parsed["schedule_date"].apply(infer_week_fields)
        parsed["week"] = [w for (w, lbl) in weeks]
        parsed["week_label"] = [lbl for (w, lbl) in weeks]
        parsed = parsed[parsed["week"].notna()]
        parsed = apply_price_factor(parsed, args.price_factor)
        for c in base_cols:
            if c not in parsed.columns: parsed[c] = np.nan

    # Cargar lo previo
    if os.path.exists(live_path):
        try:
            prev = pd.read_csv(live_path, low_memory=False)
        except Exception as e:
            print(f"[odds] warn: couldn't read previous live file: {e}")
            prev = pd.DataFrame(columns=base_cols)
    else:
        prev = pd.DataFrame(columns=base_cols)

    # Filtrar solo semana objetivo (si aplica)
    if args.only_current_week and target_week is not None:
        before = len(parsed)
        parsed = parsed[(parsed["season"] == target_season) & (parsed["week"] == target_week) |  # semana target
                        (parsed["season"] == target_season) & (parsed["week"] == target_week + 1)]  # o fallback si aplicó
        print(f"[odds] parsed week-filter {target_season}/{target_week} (+fallback): {before} → {len(parsed)} rows")

    # Append + dedupe (sumar sin sustituir)
    if parsed.empty:
        combined = prev.copy()
        print("[odds] no new rows; leaving file as-is.")
    else:
        combined = pd.concat([prev, parsed], ignore_index=True, sort=False)

    if "schedule_date" in combined.columns:
        combined["schedule_date"] = pd.to_datetime(combined["schedule_date"], errors="coerce", utc=True)
        mask = combined.get("week").isna() & combined["schedule_date"].notna() if "week" in combined.columns else False
        if isinstance(mask, pd.Series) and mask.any():
            w2 = combined.loc[mask, "schedule_date"].apply(infer_week_fields)
            combined.loc[mask, "week"] = [w for (w, lbl) in w2]
            combined.loc[mask, "week_label"] = [lbl for (w, lbl) in w2]

    if "event_id" in combined.columns and combined["event_id"].notna().any():
        combined = combined.drop_duplicates(subset=["event_id"], keep="last")
    else:
        for k in ["season","week","home_team","away_team"]:
            if k not in combined.columns: combined[k] = np.nan
        combined = combined.drop_duplicates(subset=["season","week","home_team","away_team"], keep="last")

    order_keys = [k for k in ["season","week","schedule_date","home_team"] if k in combined.columns]
    if order_keys:
        combined = combined.sort_values(order_keys, kind="mergesort")
    if "season" in combined.columns:
        combined["season"] = pd.to_numeric(combined["season"], errors="coerce").astype("Int64")
    if "week" in combined.columns:
        combined["week"] = pd.to_numeric(combined["week"], errors="coerce").astype("Int64")

    combined.to_csv(live_path, index=False)
    print(f"[odds] wrote {live_path} | rows={len(combined)} | factor={args.price_factor}")

if __name__ == "__main__":
    main()
