#!/usr/bin/env python3
# scripts/fetch_odds.py
#
# Descarga odds (The Odds API) y mantiene data/live/odds.csv
# - Lee la API key desde la variable de entorno ODDS_API_KEY (configurada en Secrets)
# - Soporta markets: h2h,spreads,totals (param --markets)
# - Dedup por event_id (o por season/week/home/away si no hay id)
# - Mantiene cabeceras aunque la API no regrese eventos (no rompe el dashboard)

import os
import sys
import argparse
from datetime import datetime, timezone
import math
import json
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

def best_decimal(prices):
    """
    Elige la cuota más favorable para el apostador (máxima decimal).
    """
    clean = [american_to_decimal(p) for p in prices if p is not None]
    clean = [c for c in clean if not (isinstance(c, float) and math.isnan(c))]
    return max(clean) if clean else np.nan

def fetch_events(api_key: str, sport: str, regions: str, markets: str) -> list:
    params = {
        "apiKey": api_key,
        "regions": regions,           # "us" o "us,us2"
        "markets": markets,          # "h2h,spreads,totals"
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    url = ODDS_API_URL.format(sport=sport)
    r = requests.get(url, params=params, timeout=30)
    status = r.status_code
    # Logs útiles de cabeceras (no exponen la key)
    print(f"[odds] status={status} x-requests-remaining={r.headers.get('x-requests-remaining')}")
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
        # Campos base
        event_id = ev.get("id")
        commence = ev.get("commence_time")  # ISO
        home = (ev.get("home_team") or "").strip()
        away = (ev.get("away_team") or "").strip()

        # contenedores
        ml_home_prices, ml_away_prices = [], []
        spread_home, spread_away = np.nan, np.nan
        total_line = np.nan

        # Recorremos bookmakers/markets
        for bk in ev.get("bookmakers", []) or []:
            for mk in bk.get("markets", []) or []:
                mk_key = mk.get("key")  # 'h2h', 'spreads', 'totals'
                if mk_key not in want_markets:
                    continue

                # h2h
                if mk_key == "h2h":
                    for oc in mk.get("outcomes", []) or []:
                        name = oc.get("name")
                        price = oc.get("price")
                        if not name:
                            continue
                        if name.strip() == home:
                            ml_home_prices.append(price)
                        elif name.strip() == away:
                            ml_away_prices.append(price)

                # spreads
                elif mk_key == "spreads":
                    # Quedarnos con el primer point que veamos (o el más común)
                    # Algunas casas dan múltiples líneas; nos alcanza con una
                    for oc in mk.get("outcomes", []) or []:
                        name = oc.get("name")
                        point = oc.get("point")
                        if name and point is not None:
                            if name.strip() == home and pd.isna(spread_home):
                                try:
                                    spread_home = float(point)
                                except Exception:
                                    pass
                            elif name.strip() == away and pd.isna(spread_away):
                                try:
                                    spread_away = float(point)
                                except Exception:
                                    pass

                # totals
                elif mk_key == "totals":
                    # Tomamos un único número de total (point); 'Over'/'Under' comparten el mismo
                    for oc in mk.get("outcomes", []) or []:
                        point = oc.get("point")
                        if point is not None and pd.isna(total_line):
                            try:
                                total_line = float(point)
                            except Exception:
                                pass

        # Elegimos las mejores cuotas (decimales)
        dec_home = best_decimal(ml_home_prices)
        dec_away = best_decimal(ml_away_prices)

        # Convertimos también a moneyline americano (aprox) para columnas homogéneas
        def decimal_to_american(d):
            try:
                d = float(d)
            except Exception:
                return np.nan
            if d <= 1.0 or math.isnan(d):
                return np.nan
            return round((d - 1.0) * 100.0, 0) if d >= 2.0 else round(-100.0 / (d - 1.0), 0)

        ml_home = decimal_to_american(dec_home)
        ml_away = decimal_to_american(dec_away)

        rows.append(dict(
            season=np.nan,                # se rellena abajo
            week=np.nan,                  # no disponible en esta API
            week_label=np.nan,            # idem
            schedule_date=commence,
            home_team=home,
            away_team=away,
            ml_home=ml_home,
            ml_away=ml_away,
            decimal_home=dec_home,
            decimal_away=dec_away,
            spread_home=spread_home,
            spread_away=spread_away,
            over_under_line=total_line,
            score_home=np.nan,            # esta API/endpoint no trae score final
            score_away=np.nan,
            event_id=event_id
        ))

    return pd.DataFrame(rows) if rows else pd.DataFrame()

def compute_season_from_date(ts_utc: pd.Timestamp) -> int:
    """
    Regla simple:
      - Si mes >= marzo → temporada = año del timestamp
      - Si mes <= febrero → temporada = año-1
    """
    if pd.isna(ts_utc):
        now = datetime.now(timezone.utc)
        return now.year if now.month >= 3 else now.year - 1
    y = ts_utc.year
    m = ts_utc.month
    return y if m >= 3 else y - 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-file", type=str, default="data/live/odds.csv")
    ap.add_argument("--sport", type=str, default="americanfootball_nfl")
    ap.add_argument("--regions", type=str, default="us")
    ap.add_argument("--markets", type=str, default="h2h,spreads,totals")
    args = ap.parse_args()

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        print("[odds] ERROR: ODDS_API_KEY env var is empty.", file=sys.stderr)
        sys.exit(1)

    want_markets = set([m.strip() for m in args.markets.split(",") if m.strip()])
    events = fetch_events(api_key, args.sport, args.regions, args.markets)

    df_new = parse_rows(events, want_markets)

    # Esquema final (orden)
    col_order = [
        "season", "week", "week_label", "schedule_date",
        "home_team", "away_team",
        "ml_home", "ml_away", "decimal_home", "decimal_away",
        "spread_home", "spread_away", "over_under_line",
        "score_home", "score_away",
        "event_id"
    ]

    if df_new.empty:
        print("[odds] parsed 0 rows; will keep existing file or write headers only.")
        out = pd.DataFrame(columns=col_order)
    else:
        out = df_new.copy()

        # Fechas/temporada
        out["schedule_date"] = pd.to_datetime(out["schedule_date"], errors="coerce", utc=True)
        out["season"] = out["schedule_date"].apply(compute_season_from_date)

        # opcional: si quisieras intentar una week_label básica (no exacta)
        # out["week_label"] = np.nan

        out = out.reindex(columns=col_order)

    # Merge con archivo existente
    live_path = args.live_file
    os.makedirs(os.path.dirname(live_path), exist_ok=True)

    if os.path.exists(live_path):
        try:
            prev = pd.read_csv(live_path, low_memory=False)
        except Exception:
            prev = pd.DataFrame(columns=col_order)
    else:
        prev = pd.DataFrame(columns=col_order)

    combined = pd.concat([prev, out], ignore_index=True)

    # De-dup: preferimos usar event_id si existe
    if "event_id" in combined.columns and combined["event_id"].notna().any():
        combined = combined.drop_duplicates(subset=["event_id"], keep="last")
    else:
        combined = combined.drop_duplicates(
            subset=["season", "week", "home_team", "away_team"], keep="last"
        )

    # Orden
    if "schedule_date" in combined.columns:
        try:
            combined["schedule_date"] = pd.to_datetime(combined["schedule_date"], errors="coerce", utc=True)
        except Exception:
            pass
        combined = combined.sort_values(["season", "schedule_date", "home_team"], kind="mergesort")
    else:
        combined = combined.sort_values(["season", "home_team"], kind="mergesort")

    combined = combined.reindex(columns=col_order)

    combined.to_csv(live_path, index=False)
    print(f"[odds] wrote {live_path} | rows={len(combined)} (added {len(out)})")

if __name__ == "__main__":
    main()
