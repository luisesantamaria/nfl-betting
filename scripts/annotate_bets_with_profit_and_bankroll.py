import os, sys, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------- paths ----------
BETS = Path("data/live/bets.csv")
ODDS = Path("data/live/odds.csv")
PNL  = Path("data/live/pnl.csv")  # opcional, solo para inf derivadas

BANKROLL_INITIAL = float(os.getenv("BANKROLL_INITIAL", "1000"))

# --------- utils ----------
def _norm_abbr(x: str) -> str:
    # Si tienes nfl_dash.utils.norm_abbr, úsalo; este fallback cubre los casos típicos (WSH/WAS).
    M = {
        "WSH": "WAS", "WASH": "WAS",
        "JAX": "JAX", "JAC": "JAX",
        "LA": "LAR", "RAMS": "LAR",
        "OAK": "LV", "LVR": "LV",
        "ARZ": "ARI",
        "TB": "TB", "TBB": "TB",
        "NO": "NO", "NOS": "NO",
        "NEP": "NE", "GBP": "GB", "SFO": "SF",
    }
    x = str(x or "").strip().upper()
    return M.get(x, x)

def _status_from(scores_known, is_final, started):
    if is_final:
        return "FINAL"
    if scores_known and started:
        return "LIVE"
    return "OPEN"

def _profit_of(stake, dec_odds, team_score, opp_score, status):
    if status != "FINAL":
        return np.nan
    if pd.isna(stake) or pd.isna(dec_odds) or pd.isna(team_score) or pd.isna(opp_score):
        return np.nan
    if math.isclose(float(team_score), float(opp_score), abs_tol=1e-9):
        return 0.0
    return float(stake) * (float(dec_odds) - 1.0) if team_score > opp_score else -float(stake)

# --------- main ----------
def main():
    if not BETS.exists() or not ODDS.exists():
        print("Faltan bets.csv u odds.csv", file=sys.stderr)
        sys.exit(1)

    bets = pd.read_csv(BETS, low_memory=False)
    odds = pd.read_csv(ODDS, low_memory=False)

    # Tipos
    for c in ("season","week","week_order"):
        if c in bets.columns: bets[c] = pd.to_numeric(bets[c], errors="coerce")
    for c in ("season","week"):
        if c in odds.columns: odds[c] = pd.to_numeric(odds[c], errors="coerce")

    bets["schedule_date"] = pd.to_datetime(bets.get("schedule_date"), errors="coerce", utc=True)
    odds["schedule_date"] = pd.to_datetime(odds.get("schedule_date"), errors="coerce", utc=True)

    # Normaliza equipos
    for c in ("team","opponent"):
        if c in bets.columns: bets[c] = bets[c].astype(str).map(_norm_abbr)
    for c in ("home_team","away_team"):
        if c in odds.columns: odds[c] = odds[c].astype(str).map(_norm_abbr)

    # Merge candidatos por season/week/week_label y filtro por par de equipos
    need_cols = ["season","week","week_label","schedule_date","home_team","away_team","score_home","score_away","home_win"]
    if not all(c in odds.columns for c in need_cols):
        print("odds.csv no tiene columnas suficientes para anotar.", file=sys.stderr)
        sys.exit(1)

    bets = bets.reset_index().rename(columns={"index": "bet_idx"})
    cand = bets.merge(odds[need_cols], on=["season","week","week_label"], how="left", suffixes=("","_o"))

    same_pair = (
        ((cand["team"] == cand["home_team"]) & (cand["opponent"] == cand["away_team"])) |
        ((cand["team"] == cand["away_team"]) & (cand["opponent"] == cand["home_team"]))
    )
    cand = cand.loc[same_pair].copy()

    cand["abs_time_diff"] = (cand["schedule_date_o"] - cand["schedule_date"]).abs().dt.total_seconds()
    cand = cand.sort_values(["bet_idx","abs_time_diff"]).groupby("bet_idx", as_index=False).first()

    # Scores por lado del pick
    cand["team_is_home"]   = (cand["team"] == cand["home_team"])
    cand["team_score"]     = np.where(cand["team_is_home"], cand["score_home"], cand["score_away"])
    cand["opponent_score"] = np.where(cand["team_is_home"], cand["score_away"], cand["score_home"])

    # Status / Result / Profit
    now_utc = datetime.now(timezone.utc)
    started = cand["schedule_date"].notna() & (cand["schedule_date"] <= now_utc)
    scores_known = cand["team_score"].notna() & cand["opponent_score"].notna()
    is_final = cand["home_win"].notna()
    cand["status"] = [_status_from(sk, f, st) for sk, f, st in zip(scores_known, is_final, started)]

    # result (solo FINAL)
    def _result(r):
        if r["status"] != "FINAL":
            return None
        if math.isclose(float(r["team_score"]), float(r["opponent_score"]), abs_tol=1e-9):
            return "PUSH"
        return "WIN" if r["team_score"] > r["opponent_score"] else "LOSS"
    cand["result"] = cand.apply(_result, axis=1)

    # profit (solo FINAL)
    # Asegura stake y decimal_odds numéricos
    bets["stake"] = pd.to_numeric(bets.get("stake"), errors="coerce")
    bets["decimal_odds"] = pd.to_numeric(bets.get("decimal_odds"), errors="coerce")
    merged = bets.merge(cand[["bet_idx","team_score","opponent_score","status","result"]], on="bet_idx", how="left")

    merged["profit_calc"] = [
        _profit_of(st, od, ts, os, stt)
        for st, od, ts, os, stt in zip(
            merged["stake"], merged["decimal_odds"], merged["team_score"], merged["opponent_score"], merged["status"]
        )
    ]

    # Escribe 'profit' (conserva si ya estaba relleno)
    if "profit" in merged.columns:
        merged["profit"] = merged["profit"].combine_first(merged["profit_calc"])
        merged = merged.drop(columns=["profit_calc"])
    else:
        merged = merged.rename(columns={"profit_calc":"profit"})

    # --------- Cálculo de bankroll_after por apuesta (solo FINAL) ----------
    # Orden estable: por season, week_order (si existe), week, schedule_date
    sort_cols = []
    if "season" in merged.columns:      sort_cols.append("season")
    if "week_order" in merged.columns:  sort_cols.append("week_order")
    if "week" in merged.columns:        sort_cols.append("week")
    if "schedule_date" in merged.columns:
        merged["schedule_date"] = pd.to_datetime(merged["schedule_date"], errors="coerce", utc=True)
        sort_cols.append("schedule_date")
    merged = merged.sort_values(sort_cols).reset_index(drop=True)

    running = BANKROLL_INITIAL
    bankroll_after = []
    for _, row in merged.iterrows():
        if str(row.get("status","")) == "FINAL" and pd.notna(row.get("profit")):
            running = running + float(row["profit"])
            bankroll_after.append(running)
        else:
            bankroll_after.append(np.nan)
    merged["bankroll_after"] = bankroll_after

    # --------- Señal de semana finalizada y bankroll semanal final ----------
    if {"season","week"}.issubset(merged.columns):
        grp = merged.groupby(["season","week"])
        week_is_final = grp["status"].apply(lambda s: (s == "FINAL").all()).rename("week_is_final")
        # último bankroll_after no nulo de la semana
        week_last_bank = grp["bankroll_after"].apply(lambda s: s.dropna().iloc[-1] if s.dropna().size else np.nan)\
                                               .rename("bankroll_week_final")
        merged = merged.merge(week_is_final, on=["season","week"], how="left")
        merged = merged.merge(week_last_bank, on=["season","week"], how="left")
    else:
        merged["week_is_final"] = False
        merged["bankroll_week_final"] = np.nan

    # Limpieza
    merged = merged.drop(columns=["bet_idx"], errors="ignore")

    # Guardar
    merged.to_csv(BETS, index=False)
    print("bets.csv actualizado con scores/status/result/profit/bankroll_after/bankroll_week_final.")

if __name__ == "__main__":
    main()
