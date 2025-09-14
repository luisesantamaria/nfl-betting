#!/usr/bin/env python3
import os
import sys
import argparse
import math
import json
import shutil
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

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
        home = (ev.get("home_team") or "").strip()
        away = (ev.get("away_team") or "").strip()

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
                        if name.strip() == home:
                            ml_home_prices.append(price)
                        elif name.strip() == away:
                            ml_away_prices.append(price)

                elif k == "spreads":
                    for oc in mk.get("outcomes", []) or []:
                        name, point = oc.get("name"), oc.get("point")
                        if name and point is not None:
                            try:
                                if name.strip() == home and pd.isna(spread_home):
                                    spread_home = float(point)
                                elif name.strip() == away and pd.isna(spread_away):
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

def compute_season_from_ts(ts_utc: pd.Timestamp) -> int:
    if pd.isna(ts_utc):
        now = datetime.now(timezone.utc)
        return now.year if now.month >= 3 else now.year - 1
    y, m = ts_utc.year, ts_utc.month
    return y if m >= 3 else y - 1

def first_thursday_after_labor_day_ts(year: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=9, day=1, tz="UTC")
    while d.weekday() != 0:
        d += pd.Timedelta(days=1)
    thu = d + pd.Timedelta(days=3)
    return thu.normalize()

LABEL_POST = {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}

def infer_week_fields(ts_utc: pd.Timestamp):
    if pd.isna(ts_utc):
        return (np.nan, np.nan)
    year = compute_season_from_ts(ts_utc)
    anchor = first_thursday_after_labor_day_ts(year)
    dd = (ts_utc.normalize() - anchor).days
    w = 1 + (dd // 7)
    if w < 1:
        return (np.nan, np.nan)
    if w <= 18:
        return (int(w), f"Week {int(w)}")
    k = min(4, int(w) - 18)
    wk = 18 + k
    return (wk, LABEL_POST.get(wk, f"Week {wk}"))

def current_season_week(now_utc: datetime | None = None):
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    ts = pd.Timestamp(now_utc)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    season = compute_season_from_ts(ts)
    w, lbl = infer_week_fields(ts)
    return season, w, lbl

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

def detect_file_season(df: pd.DataFrame) -> int | None:
    if "schedule_date" in df.columns:
        ts = pd.to_datetime(df["schedule_date"], errors="coerce", utc=True)
        if not ts.isna().all():
            seasons = ts.apply(compute_season_from_ts)
            if not seasons.dropna().empty:
                return int(seasons.mode().iloc[0])
    if "season" in df.columns:
        s = pd.to_numeric(df["season"], errors="coerce")
        if s.notna().any():
            return int(s.mode().iloc[0])
    return None

def season_is_over(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    wk = pd.to_numeric(df.get("week"), errors="coerce")
    if wk.notna().any() and wk.max() >= 22:
        return True
    fseason = detect_file_season(df)
    if fseason is None:
        return False
    now_season = compute_season_from_ts(pd.Timestamp(datetime.now(timezone.utc)))
    return now_season > fseason

def archive_if_needed(live_path: str, archive_dir: str, force: bool) -> None:
    if not os.path.exists(live_path):
        return
    try:
        df = pd.read_csv(live_path, low_memory=False)
    except Exception:
        return
    if df.empty and not force:
        return
    file_season = detect_file_season(df)
    if file_season is None:
        file_season = compute_season_from_ts(pd.Timestamp(datetime.now(timezone.utc)))
    if not force and not season_is_over(df):
        return
    season_dir = os.path.join(archive_dir, f"season={file_season}")
    os.makedirs(season_dir, exist_ok=True)
    dest_path = os.path.join(season_dir, "odds.csv")
    if os.path.exists(dest_path) and not force:
        return
    shutil.copy2(live_path, dest_path)
    cols = list(df.columns)
    pd.DataFrame(columns=cols).to_csv(live_path, index=False)
    print(f"[archive] wrote {dest_path} and reset {live_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-file", type=str, default="data/live/odds.csv")
    ap.add_argument("--archive-dir", type=str, default="data/archive")
    ap.add_argument("--sport", type=str, default="americanfootball_nfl")
    ap.add_argument("--regions", type=str, default="us")
    ap.add_argument("--markets", type=str, default="h2h,spreads,totals")
    ap.add_argument("--only-current-week", dest="only_current_week", action="store_true", default=True)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--price-factor", type=float, default=float(os.environ.get("ODDS_PRICE_FACTOR", "1.022")))
    ap.add_argument("--archive-on-season-end", dest="archive_on_season_end", action="store_true", default=True)
    ap.add_argument("--no-archive-on-season-end", dest="archive_on_season_end", action="store_false")
    ap.add_argument("--archive-force", action="store_true", default=False)
    args = ap.parse_args()

    live_path = args.live_file
    os.makedirs(os.path.dirname(live_path), exist_ok=True)

    cur_season, cur_week, cur_label = current_season_week()
    print(f"[odds] now → season={cur_season}, week={cur_week} ({cur_label})")

    # ⛔ Preseason/Off-season guard: si aún no hay Week 1, no llamamos API
    if pd.isna(cur_week):
        print("[odds] offseason or preseason detected → skip fetch.")
        if args.archive_force or args.archive_on_season_end:
            archive_if_needed(live_path, args.archive_dir, force=args.archive_force)
        return

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        print("[odds] ERROR: ODDS_API_KEY env var is empty.", file=sys.stderr)
        sys.exit(1)

    want_markets = set([m.strip() for m in args.markets.split(",") if m.strip()])
    events = fetch_events(api_key, args.sport, args.regions, args.markets)
    df_new = parse_rows(events, want_markets)

    col_order = [
        "season", "week", "week_label", "schedule_date",
        "home_team", "away_team",
        "ml_home", "ml_away", "decimal_home", "decimal_away",
        "ml_home_raw", "ml_away_raw", "decimal_home_raw", "decimal_away_raw",
        "spread_home", "spread_away", "over_under_line",
        "price_factor",
        "event_id"
    ]

    if df_new.empty:
        parsed = pd.DataFrame(columns=col_order)
    else:
        parsed = df_new.copy()
        parsed["schedule_date"] = pd.to_datetime(parsed["schedule_date"], errors="coerce", utc=True)
        parsed["season"] = parsed["schedule_date"].apply(compute_season_from_ts).astype("Int64")
        weeks = parsed["schedule_date"].apply(infer_week_fields)
        parsed["week"] = [w for (w, lbl) in weeks]
        parsed["week_label"] = [lbl for (w, lbl) in weeks]
        parsed = parsed[parsed["week"].notna()]
        parsed = apply_price_factor(parsed, args.price_factor)
        parsed = parsed.reindex(columns=col_order)

    if os.path.exists(live_path):
        try:
            prev = pd.read_csv(live_path, low_memory=False)
        except Exception:
            prev = pd.DataFrame(columns=col_order)
    else:
        prev = pd.DataFrame(columns=col_order)
    prev = prev.reindex(columns=col_order)

    if parsed.empty:
        combined = prev.copy()
    else:
        combined = pd.concat([prev, parsed], ignore_index=True)

    if "schedule_date" in combined.columns:
        combined["schedule_date"] = pd.to_datetime(combined["schedule_date"], errors="coerce", utc=True)
        mask = combined["week"].isna() & combined["schedule_date"].notna()
        if mask.any():
            w2 = combined.loc[mask, "schedule_date"].apply(infer_week_fields)
            combined.loc[mask, "week"] = [w for (w, lbl) in w2]
            combined.loc[mask, "week_label"] = [lbl for (w, lbl) in w2]

    target_season = args.season if args.season is not None else cur_season
    target_week = args.week if args.week is not None else cur_week

    if args.only_current_week and pd.notna(target_week):
        before = len(combined)
        combined = combined[(combined["season"] == target_season) & (combined["week"] == target_week)]
        print(f"[odds] week-filter {target_season}/{target_week}: {before} → {len(combined)} rows")

    if "event_id" in combined.columns and combined["event_id"].notna().any():
        combined = combined.drop_duplicates(subset=["event_id"], keep="last")
    else:
        combined = combined.drop_duplicates(subset=["season","week","home_team","away_team"], keep="last")

    combined = combined.sort_values(["season","week","schedule_date","home_team"], kind="mergesort")
    combined["season"] = pd.to_numeric(combined["season"], errors="coerce").astype("Int64")
    combined["week"] = pd.to_numeric(combined["week"], errors="coerce").astype("Int64")
    combined = combined.reindex(columns=col_order)

    combined.to_csv(live_path, index=False)
    print(f"[odds] wrote {live_path} | rows={len(combined)} | factor={args.price_factor}")

    if args.archive_force or args.archive_on_season_end:
        archive_if_needed(live_path, args.archive_dir, force=args.archive_force)

if __name__ == "__main__":
    main()
