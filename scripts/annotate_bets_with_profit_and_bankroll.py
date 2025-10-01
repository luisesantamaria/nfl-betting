# scripts/annotate_bets_with_profit_and_bankroll.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

LIVE_BETS_FILE = Path("data/live/bets.csv")
LIVE_ODDS_FILE = Path("data/live/odds.csv")

INITIAL_BANKROLL = 1000.0
FREEZE_FINAL_WEEKS = True  # congelar semanas cerradas

TEAM_FIX = {
    "STL":"LA","LAR":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV","WSH":"WAS","JAC":"JAX",
    "GNB":"GB","KAN":"KC","NWE":"NE","NOR":"NO","SFO":"SF","TAM":"TB"
}
ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX = {lab: i for i, lab in enumerate(ORDER_LABELS)}

def _norm_team(x: str) -> str:
    s = str(x).upper().strip()
    return TEAM_FIX.get(s, s)

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def _week_label_from_num(n: int) -> str:
    if 1 <= n <= 18:
        return f"Week {n}"
    return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")

def _normalize_week_order(df: pd.DataFrame) -> pd.Series:
    if "week_order" in df.columns:
        return _safe_num(df["week_order"]).fillna(999).astype(int)
    if "week" in df.columns:
        return _safe_num(df["week"]).fillna(999).astype(int)
    if "week_label" in df.columns:
        wk = df["week_label"].astype(str).str.extract(r"(\d+)")[0]
        return _safe_num(wk).fillna(999).astype(int)
    return pd.Series([999] * len(df), index=df.index, dtype=int)

def _make_game_id(week_label: str, a: str, b: str) -> str:
    a = str(a); b = str(b)
    pair = a + "_" + b if a < b else b + "_" + a
    return f"{week_label} | {pair}"

def _decide_result(team_score, opp_score):
    if pd.isna(team_score) or pd.isna(opp_score):
        return pd.NA
    try:
        a, b = float(team_score), float(opp_score)
    except Exception:
        return pd.NA
    if a > b: return "WIN"
    if a < b: return "LOSS"
    return "PUSH"

def _compute_profit(stake, decimal_odds, result):
    stake = _safe_num(stake)
    dec = _safe_num(decimal_odds)
    if pd.isna(stake) or pd.isna(dec) or pd.isna(result):
        return pd.NA
    r = str(result)
    if r == "WIN":  return float(stake) * (float(dec) - 1.0)
    if r == "LOSS": return -float(stake)
    if r == "PUSH": return 0.0
    return pd.NA

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    need = [
        "season","week","week_label","schedule_date","side","team","opponent",
        "decimal_odds","stake","team_score","opponent_score","status","status_short",
        "result","profit","bankroll_after","bankroll_week_final","game_id","week_order"
    ]
    for c in need:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def _prepare_bets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("team","opponent"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(_norm_team)
    for col in ("stake","profit","decimal_odds","bankroll_after","bankroll_week_final"):
        if col in df.columns:
            df[col] = _safe_num(df[col])
    df["schedule_date"] = pd.to_datetime(df.get("schedule_date"), errors="coerce", utc=True)
    if "week_label" not in df.columns or df["week_label"].isna().any():
        if "week" in df.columns:
            df["week_label"] = df["week"].apply(lambda x: _week_label_from_num(int(x)) if pd.notna(x) else pd.NA)
    df["week_order"] = _normalize_week_order(df)
    df["team_score"] = _safe_num(df.get("team_score"))
    df["opponent_score"] = _safe_num(df.get("opponent_score"))
    df["season"] = _safe_num(df.get("season")).astype("Int64")
    df["week"] = _safe_num(df.get("week")).astype("Int64")
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.lower()
    if "game_id" not in df.columns or df["game_id"].isna().any():
        df["game_id"] = [
            _make_game_id(wl, _norm_team(t), _norm_team(o)) if pd.notna(wl) else pd.NA
            for wl, t, o in zip(df["week_label"], df["team"], df["opponent"])
        ]
    return df

def _load_live_odds() -> pd.DataFrame:
    if not LIVE_ODDS_FILE.exists():
        return pd.DataFrame()
    odds = pd.read_csv(LIVE_ODDS_FILE, low_memory=False)
    for c in ("home_team","away_team"):
        if c in odds.columns:
            odds[c] = odds[c].astype(str).map(_norm_team)
    odds["schedule_date"] = pd.to_datetime(odds.get("schedule_date"), errors="coerce", utc=True)
    odds["season"] = _safe_num(odds.get("season")).astype("Int64")
    odds["week"] = _safe_num(odds.get("week")).astype("Int64")
    if "week_label" not in odds.columns or odds["week_label"].isna().any():
        odds["week_label"] = odds["week"].apply(lambda x: _week_label_from_num(int(x)) if pd.notna(x) else pd.NA)
    odds["game_id"] = [
        _make_game_id(wl, _norm_team(h), _norm_team(a)) if pd.notna(wl) else pd.NA
        for wl, h, a in zip(odds["week_label"], odds["home_team"], odds["away_team"])
    ]
    odds["score_home"] = _safe_num(odds.get("score_home"))
    odds["score_away"] = _safe_num(odds.get("score_away"))
    odds = (odds.sort_values(["season","week","schedule_date"])
                 .drop_duplicates(subset=["season","week","game_id"], keep="last"))
    return odds

def _merge_scores_incomplete_weeks(bets: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Completa scores y estado solo en semanas no finalizadas."""
    if bets.empty or odds.empty:
        return bets
    m = bets.merge(
        odds[["season","week","game_id","score_home","score_away"]],
        on=["season","week","game_id"], how="left"
    )

    # Tomar scores según 'side'
    team_score_from_odds = np.where(
        m["side"].str.lower().eq("home"), m["score_home"],
        np.where(m["side"].str.lower().eq("away"), m["score_away"], np.nan)
    )
    opp_score_from_odds = np.where(
        m["side"].str.lower().eq("home"), m["score_away"],
        np.where(m["side"].str.lower().eq("away"), m["score_home"], np.nan)
    )

    # Completar SOLO si faltan
    m["team_score"] = m["team_score"].where(m["team_score"].notna(), team_score_from_odds)
    m["opponent_score"] = m["opponent_score"].where(m["opponent_score"].notna(), opp_score_from_odds)

    has_scores = m["team_score"].notna() & m["opponent_score"].notna()
    needs_status = has_scores & m["status"].isna()
    needs_short  = has_scores & m["status_short"].isna()
    m.loc[needs_status, "status"] = "FINAL"
    m.loc[needs_short,  "status_short"] = "Final"

    return m.drop(columns=["score_home","score_away"], errors="ignore")

def _is_week_final_by_profit(df_week: pd.DataFrame) -> bool:
    """Regla pedida: semana final si TODAS las filas tienen profit NO nulo."""
    if df_week.empty:
        return False
    all_profit = df_week["profit"].notna().all()
    if bool(all_profit):
        return True
    # Fallback: si todas tienen result no nulo
    all_result = df_week["result"].notna().all()
    return bool(all_result)

def _process_live_freezing(bets_now: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if bets_now.empty:
        return bets_now

    bets_now = _ensure_cols(bets_now)
    bets_now = _prepare_bets(bets_now)

    # Congelar semanas que ya traen bankroll_week_final (no las tocamos)
    if FREEZE_FINAL_WEEKS:
        by_wk = bets_now.groupby(["season","week"], dropna=False)["bankroll_week_final"].apply(
            lambda s: s.notna().any()
        )
        freeze_pairs = set(k for k, v in by_wk.items() if bool(v))
        mask_final = bets_now.apply(lambda r: (r.get("season"), r.get("week")) in freeze_pairs, axis=1)
        mask_final = mask_final.fillna(False).astype(bool)
    else:
        mask_final = pd.Series([False]*len(bets_now), index=bets_now.index, dtype=bool)

    frozen_df = bets_now.loc[mask_final].copy()
    working_df = bets_now.loc[~mask_final].copy()

    # Completar scores y estatus en working_df
    working_df = _merge_scores_incomplete_weeks(working_df, odds)

    # Calcular result y profit donde ya hay score y no están
    has_scores = working_df["team_score"].notna() & working_df["opponent_score"].notna()
    # result
    calc_result = [
        _decide_result(a, b) if (pd.notna(a) and pd.notna(b)) else pd.NA
        for a, b in zip(working_df["team_score"], working_df["opponent_score"])
    ]
    working_df["result"] = working_df["result"].where(working_df["result"].notna(), calc_result)
    # profit
    need_profit = has_scores & working_df["profit"].isna()
    working_df.loc[need_profit, "profit"] = [
        _compute_profit(st, dec, res)
        for st, dec, res in zip(
            working_df.loc[need_profit, "stake"],
            working_df.loc[need_profit, "decimal_odds"],
            working_df.loc[need_profit, "result"],
        )
    ]

    # Orden estable para carrera por semana
    working_df["__sort_week"] = working_df["week_order"].astype(int)
    working_df["__sort_time"] = working_df["schedule_date"]

    out_blocks = []
    wk_final_bank_by_order = {}

    # Precargar cierres desde bets_now (incluye congeladas)
    prev = bets_now.dropna(subset=["bankroll_week_final"])
    for wkord, g in prev.groupby("week_order"):
        try:
            wk_final_bank_by_order[int(wkord)] = float(g["bankroll_week_final"].dropna().iloc[-1])
        except Exception:
            pass

    # Recorre temporada->semana
    for season, df_season in working_df.sort_values(["season","__sort_week","__sort_time"], kind="stable").groupby("season", sort=True):
        df_season = df_season.copy()
        for wkord, df_week in df_season.groupby("__sort_week", sort=True):
            df_week = df_week.copy().sort_values(["__sort_time"], kind="stable")
            prev_ord = int(wkord) - 1
            baseline = float(wk_final_bank_by_order.get(prev_ord, INITIAL_BANKROLL))
            running = baseline
            last_final_bank = pd.NA

            # Acumular bankroll por apuestas ya liquidadas (profit no nulo)
            for idx, row in df_week.iterrows():
                prof = df_week.at[idx, "profit"]
                if pd.notna(prof):
                    running += float(prof)
                    df_week.at[idx, "bankroll_after"] = running
                    last_final_bank = running
                else:
                    df_week.at[idx, "bankroll_after"] = pd.NA

            # Regla de cierre de semana
            week_complete = _is_week_final_by_profit(df_week)
            df_week["bankroll_week_final"] = last_final_bank if week_complete else pd.NA

            if week_complete and pd.notna(last_final_bank):
                wk_final_bank_by_order[int(wkord)] = float(last_final_bank)

            # Redondeo en filas trabajadas
            for c in ["stake", "profit", "bankroll_after", "bankroll_week_final"]:
                if c in df_week.columns:
                    df_week[c] = _safe_num(df_week[c]).round(2)

            out_blocks.append(df_week)

    updated_working = (
        pd.concat(out_blocks, ignore_index=True) if out_blocks else working_df
    ).drop(columns=["__sort_week","__sort_time"], errors="ignore")

    # Unir: congeladas intactas + actualizadas
    final_df = pd.concat([frozen_df, updated_working], ignore_index=True)
    final_df = final_df.sort_values(["week_order","schedule_date"], kind="stable")

    # Quitar columna legacy 'bankroll' si existe
    final_df = final_df.drop(columns=["bankroll"], errors="ignore")
    return final_df

def main():
    if not LIVE_BETS_FILE.exists():
        print("⚠️ No existe data/live/bets.csv; nada que procesar.")
        return

    bets_now = pd.read_csv(LIVE_BETS_FILE, low_memory=False)
    odds = _load_live_odds()

    out = _process_live_freezing(bets_now, odds)

    # Persistir
    out.to_csv(LIVE_BETS_FILE, index=False)
    print("✅ LIVE actualizado: semanas finalizadas se congelan; semanas abiertas se completan con scores, result/profit y bankroll_after. 'bankroll_week_final' solo si la semana quedó completa (todos los profit != NA).")

if __name__ == "__main__":
    main()
