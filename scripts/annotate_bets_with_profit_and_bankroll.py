#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anota las apuestas con marcadores/estado y genera pnl semanal.

Entrada:
  - data/live/bets.csv  (debe tener: season, week, side, team, opponent, stake, ml o decimal_odds)

Salida:
  - data/live/bets.csv  (mismo archivo, enriquecido con: team_score, opponent_score, status, status_short, profit)
  - data/live/pnl.csv   (por temporada/semana: profit, stake, bankroll, week_label)

Requiere:
  - Paquete del repo: nfl_dash.utils (norm_abbr, american_to_decimal)
  - Internet para consultar ESPN
"""

from __future__ import annotations
import sys
import json
import time
import math
import requests
import pandas as pd
from pathlib import Path

# --- Paths
ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "data" / "live"
BETS_CSV = LIVE_DIR / "bets.csv"
PNL_CSV  = LIVE_DIR / "pnl.csv"

# --- Imports del paquete
sys.path.insert(0, str(ROOT))
from nfl_dash.utils import norm_abbr, american_to_decimal  # noqa: E402

# --- ESPN API
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Connection": "keep-alive",
}
ENDPOINTS = [
    "https://site.web.api.espn.com/apis/v2/sports/football/nfl/scoreboard",
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
]


def _safe_get(params: dict, timeout: int = 20, retries: int = 3):
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
            except requests.RequestException as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
                continue
    if last_err:
        print(f"[WARN] ESPN fetch failed: {last_err}", file=sys.stderr)
    return None


def fetch_espn_scoreboard_df(season: int, week: int) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas:
      home_team, away_team, home_score, away_score, state(pre/in/post), short, start_time
    """
    params = {"seasontype": 2, "week": int(week), "dates": int(season)}
    data = _safe_get(params=params, timeout=20, retries=3)
    rows = []
    if data:
        for ev in data.get("events", []) or []:
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]

            st_type = (comp.get("status") or {}).get("type") or {}
            state = str(st_type.get("state") or "").lower()  # pre|in|post
            short = str(st_type.get("shortDetail") or "")
            start_time = comp.get("date")

            home_team = away_team = None
            home_score = away_score = None
            for c in comp.get("competitors", []) or []:
                team = c.get("team") or {}
                name = team.get("displayName") or team.get("name")
                try:
                    score = int(c.get("score")) if c.get("score") is not None else None
                except Exception:
                    score = None
                if str(c.get("homeAway")) == "home":
                    home_team, home_score = name, score
                else:
                    away_team, away_score = name, score

            if home_team and away_team:
                rows.append(
                    {
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "state": state,
                        "short": short,
                        "start_time": pd.to_datetime(start_time, errors="coerce", utc=True),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["home_abbr"] = df["home_team"].astype(str).map(norm_abbr)
    df["away_abbr"] = df["away_team"].astype(str).map(norm_abbr)
    return df


def _compute_profit(stake: float, dec_odds: float, team_score, opp_score, status: str):
    """
    Regresa profit o NaN si no finalizó; 0 si push.
    """
    if pd.isna(team_score) or pd.isna(opp_score) or str(status).upper() not in {"FINAL", "POST"}:
        return pd.NA
    try:
        s = float(stake) if stake is not None else 0.0
        d = float(dec_odds) if dec_odds is not None else None
    except Exception:
        return pd.NA

    if d is None or s <= 0:
        return pd.NA

    if team_score > opp_score:
        return round(s * (d - 1.0), 2)
    elif team_score < opp_score:
        return round(-s, 2)
    else:
        # push
        return 0.0


def _week_label(n: int) -> str:
    if 1 <= n <= 18:
        return f"Week {n}"
    mapping = {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}
    return mapping.get(int(n), f"Week {n}")


def main():
    if not BETS_CSV.exists():
        print(f"[INFO] Bets file not found: {BETS_CSV}", file=sys.stderr)
        return

    bets = pd.read_csv(BETS_CSV, low_memory=False)

    # Normaliza columnas básicas
    for c in ("season", "week", "stake", "decimal_odds", "ml"):
        if c in bets.columns:
            bets[c] = pd.to_numeric(bets[c], errors="coerce")

    # decimal_odds desde ml si falta
    if "decimal_odds" not in bets.columns:
        bets["decimal_odds"] = pd.NA
    if "ml" in bets.columns:
        missing = bets["decimal_odds"].isna()
        bets.loc[missing, "decimal_odds"] = bets.loc[missing, "ml"].apply(american_to_decimal)

    # normaliza teams/sides
    for c in ("team", "opponent"):
        if c in bets.columns:
            bets[c] = bets[c].astype(str).map(norm_abbr)
        else:
            bets[c] = ""

    if "side" not in bets.columns:
        bets["side"] = ""
    bets["side"] = bets["side"].astype(str).str.lower()

    # temporada/semana objetivo: toma maxima (la actual)
    season = int(pd.to_numeric(bets.get("season", pd.Series([pd.NA])), errors="coerce").dropna().max())
    # Si no hay season, nada que hacer
    if not season or math.isnan(season):
        print("[WARN] No season detected in bets.csv", file=sys.stderr)
        return

    # Trae scoreboard por semana única de las filas de esa season
    weeks = sorted(pd.to_numeric(bets.loc[bets["season"] == season, "week"], errors="coerce").dropna().unique().tolist())
    if not weeks:
        print("[WARN] No week values for current season.", file=sys.stderr)

    # Índice ESPN por (home_abbr, away_abbr)
    idx_by_week: dict[int, dict[tuple[str, str], dict]] = {}
    for wk in weeks:
        es = fetch_espn_scoreboard_df(season=season, week=int(wk))
        tmp = {}
        for r in es.to_dict("records"):
            ha, aa = r["home_abbr"], r["away_abbr"]
            tmp[(ha, aa)] = r
            tmp[(aa, ha)] = r
        idx_by_week[int(wk)] = tmp

    # Columnas de salida
    bets["team_score"] = pd.NA
    bets["opponent_score"] = pd.NA
    bets["status"] = pd.NA         # FINAL / LIVE / OPEN
    bets["status_short"] = pd.NA
    bets["profit"] = pd.NA

    # Anota por fila
    for i, r in bets.iterrows():
        if int(r.get("season", season)) != season:
            continue
        wk = r.get("week")
        if pd.isna(wk):
            continue
        wk = int(wk)
        t = r.get("team", "")
        o = r.get("opponent", "")
        side = str(r.get("side", "")).lower()

        # Determinar (home,away) de la vista-bet
        if side == "home":
            home, away = t, o
        elif side == "away":
            home, away = o, t
        else:
            # intentar inferir por key directo
            home, away = None, None

        es_idx = idx_by_week.get(wk, {})

        srow = None
        if home and away:
            srow = es_idx.get((home, away))
        else:
            # intento 1
            srow = es_idx.get((t, o))
            if not srow:
                srow = es_idx.get((o, t))

        if not srow:
            # no encontrado: queda OPEN
            bets.at[i, "status"] = "OPEN"
            continue

        # estado y marcadores
        state = str(srow.get("state") or "").lower()
        status = "FINAL" if state == "post" else ("LIVE" if state == "in" else "OPEN")
        bets.at[i, "status"] = status
        bets.at[i, "status_short"] = srow.get("short")

        hs = srow.get("home_score")
        as_ = srow.get("away_score")

        # asignar scores del lado apostado
        # Si no definimos home/away por side, lo deducimos de srow:
        if home is None or away is None:
            ha = srow.get("home_abbr"); aa = srow.get("away_abbr")
            # si el team coincide con el home del juego, apostaste al home
            if t == ha:
                home, away = ha, aa
                side = "home"
            elif t == aa:
                home, away = ha, aa
                side = "away"
            else:
                # fallback: no podemos mapear -> sin scores
                continue

        if side == "home":
            team_sc, opp_sc = hs, as_
        else:
            team_sc, opp_sc = as_, hs

        bets.at[i, "team_score"] = team_sc
        bets.at[i, "opponent_score"] = opp_sc

        # profit sólo si FINAL
        stake = r.get("stake")
        dec = r.get("decimal_odds")
        bets.at[i, "profit"] = _compute_profit(stake, dec, team_sc, opp_sc, status)

    # Guardar bets enriquecidas
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    bets.to_csv(BETS_CSV, index=False)

    # ---- PNL semanal (por season)
    df = bets[bets["season"] == season].copy()

    # Semana numérica
    df["week_num"] = pd.to_numeric(df.get("week"), errors="coerce")

    # Sólo cuenta profit final (FINAL) para pnl; NaN en otros
    mask_final = df["status"].astype(str).str.upper().eq("FINAL")
    df.loc[~mask_final, "profit"] = pd.NA

    grp = df.groupby("week_num", dropna=True)
    agg = grp.agg(
        profit=("profit", lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
        stake=("stake",  lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()),
    ).reset_index()

    if agg.empty:
        # crear archivo vacío mínimo
        out = pd.DataFrame(columns=["season", "week", "week_label", "profit", "stake", "bankroll"])
        out.to_csv(PNL_CSV, index=False)
        print("[INFO] pnl.csv written (empty).")
        return

    agg = agg.sort_values("week_num")
    agg["season"] = season
    agg["week"] = agg["week_num"].astype(int)
    agg["week_label"] = agg["week"].apply(_week_label)

    # bankroll acumulado desde 1000
    initial = 1000.0
    profits = pd.to_numeric(agg["profit"], errors="coerce").fillna(0.0).values
    bankroll = [initial + profits[:i+1].sum() for i in range(len(profits))]
    agg["bankroll"] = bankroll

    out_cols = ["season", "week", "week_label", "profit", "stake", "bankroll"]
    agg[out_cols].to_csv(PNL_CSV, index=False)
    print("[INFO] bets.csv and pnl.csv written successfully.")


if __name__ == "__main__":
    main()
