# scripts/annotate_bets_with_profit_and_bankroll.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

LIVE_BETS_FILE = Path("data/live/bets.csv")
LIVE_ODDS_FILE = Path("data/live/odds.csv")

INITIAL_BANKROLL = 1000.0

ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX = {lab: i for i, lab in enumerate(ORDER_LABELS)}

TEAM_FIX = {
    "STL": "LA", "LAR": "LA", "SD": "LAC", "SDG": "LAC", "OAK": "LV", "LVR": "LV",
    "WSH": "WAS", "JAC": "JAX", "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB"
}

def _norm_team(x: str) -> str:
    s = str(x).upper().strip()
    return TEAM_FIX.get(s, s)

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def _week_label_from_num(n: int) -> str:
    if 1 <= n <= 18:
        return f"Week {n}"
    return {19:"Wild Card", 20:"Divisional", 21:"Conference", 22:"Super Bowl"}.get(n, f"Week {n}")

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

def _ensure_bets_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "season", "week", "week_label", "schedule_date",
        "side", "team", "opponent", "decimal_odds", "stake",
        "team_score", "opponent_score", "status", "status_short",
        "result", "profit", "bankroll_after", "bankroll_week_final",
        "week_is_final", "game_id", "week_order"
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def _prepare_bets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normaliza teams
    for c in ("team", "opponent"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(_norm_team)

    # numéricos
    for col in ("stake", "profit", "decimal_odds", "bankroll_after", "bankroll_week_final"):
        if col in df.columns:
            df[col] = _safe_num(df[col])

    # fechas
    df["schedule_date"] = pd.to_datetime(df.get("schedule_date"), errors="coerce", utc=True)

    # week_label / week_order
    if "week_label" not in df.columns or df["week_label"].isna().any():
        if "week" in df.columns:
            df["week_label"] = df["week"].apply(lambda x: _week_label_from_num(int(x)) if pd.notna(x) else pd.NA)
    df["week_order"] = _normalize_week_order(df)

    # scores
    df["team_score"] = _safe_num(df["team_score"])
    df["opponent_score"] = _safe_num(df["opponent_score"])

    # season/week
    df["season"] = _safe_num(df["season"]).astype("Int64")
    df["week"] = _safe_num(df["week"]).astype("Int64")

    # side
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.lower()

    # game_id si faltara
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

    # normaliza
    for c in ("home_team", "away_team"):
        if c in odds.columns:
            odds[c] = odds[c].astype(str).map(_norm_team)
    odds["schedule_date"] = pd.to_datetime(odds.get("schedule_date"), errors="coerce", utc=True)
    odds["season"] = _safe_num(odds.get("season")).astype("Int64")
    odds["week"] = _safe_num(odds.get("week")).astype("Int64")
    if "week_label" not in odds.columns or odds["week_label"].isna().any():
        odds["week_label"] = odds["week"].apply(lambda x: _week_label_from_num(int(x)) if pd.notna(x) else pd.NA)

    # game_id como en bets
    odds["game_id"] = [
        _make_game_id(wl, _norm_team(h), _norm_team(a)) if pd.notna(wl) else pd.NA
        for wl, h, a in zip(odds["week_label"], odds["home_team"], odds["away_team"])
    ]

    # scores
    odds["score_home"] = _safe_num(odds.get("score_home"))
    odds["score_away"] = _safe_num(odds.get("score_away"))

    # dedup por juego -> última fila (por si hay múltiples libros/actualizaciones)
    odds = (odds.sort_values(["season", "week", "schedule_date"])
                 .drop_duplicates(subset=["season", "week", "game_id"], keep="last"))
    return odds

def _merge_scores(bets: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if bets.empty or odds.empty:
        return bets
    m = bets.merge(
        odds[["season","week","game_id","score_home","score_away"]],
        on=["season","week","game_id"], how="left"
    )

    # asigna scores al lado correcto (solo si faltan en bets)
    team_score_from_odds = np.where(
        m["side"].str.lower().eq("home"), m["score_home"],
        np.where(m["side"].str.lower().eq("away"), m["score_away"], np.nan)
    )
    opp_score_from_odds = np.where(
        m["side"].str.lower().eq("home"), m["score_away"],
        np.where(m["side"].str.lower().eq("away"), m["score_home"], np.nan)
    )

    m["team_score"] = m["team_score"].where(m["team_score"].notna(), team_score_from_odds)
    m["opponent_score"] = m["opponent_score"].where(m["opponent_score"].notna(), opp_score_from_odds)

    has_scores = m["team_score"].notna() & m["opponent_score"].notna()
    needs_status = has_scores & m["status"].isna()
    needs_short  = has_scores & m["status_short"].isna()
    m.loc[needs_status, "status"] = "FINAL"
    m.loc[needs_short,  "status_short"] = "Final"

    # limpia columnas auxiliares si se agregaron
    return m.drop(columns=["score_home","score_away"], errors="ignore")

def _process_live(bets: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return bets

    bets = _ensure_bets_columns(bets)
    bets = _prepare_bets(bets)

    # 1) Completar scores y estatus desde odds
    bets = _merge_scores(bets, odds)

    # 2) Flags de final
    has_final_status = bets.get("status", pd.Series([""] * len(bets), index=bets.index)).astype(str).str.upper().eq("FINAL")
    has_scores = bets["team_score"].notna() & bets["opponent_score"].notna()
    is_final_row = has_final_status | has_scores

    # 3) Result (si falta) a partir de scores
    calc_result = [
        _decide_result(a, b) if (pd.notna(a) and pd.notna(b)) else pd.NA
        for a, b in zip(bets["team_score"], bets["opponent_score"])
    ]
    bets["result"] = bets["result"].where(bets["result"].notna(), calc_result)

    # 4) Asegura status/short si ya hay scores
    needs_status = has_scores & bets["status"].isna()
    bets.loc[needs_status, "status"] = "FINAL"
    needs_short  = has_scores & bets["status_short"].isna()
    bets.loc[needs_short,  "status_short"] = "Final"

    # 5) Orden para calcular bankroll_after y cierre semanal
    bets["__sort_week"] = bets["week_order"].astype(int)
    bets["__sort_time"] = bets["schedule_date"]

    out_blocks = []

    # baseline por semana: último bankroll_week_final de la semana previa
    # si no existe, usa INITIAL_BANKROLL
    # Nota: No escribimos columna 'bankroll' (se elimina al final)
    # pero internamente la usamos como baseline de cada semana.
    # También soporta semanas futuras (sin scores aún).
    wk_final_by_order = {}  # week_order -> bankroll_week_final

    for season, df_season in bets.sort_values(["season","__sort_week","__sort_time"], kind="stable").groupby("season", sort=True):
        df_season = df_season.copy()

        # pre-cargar cierres de semanas ya presentes en el CSV (si existieran)
        tmp = df_season.dropna(subset=["bankroll_week_final"])
        for wkord, g in tmp.groupby("__sort_week"):
            wk_final_by_order[wkord] = float(g["bankroll_week_final"].dropna().iloc[-1])

        for wkord, df_week in df_season.groupby("__sort_week", sort=True):
            df_week = df_week.copy().sort_values(["__sort_time"], kind="stable")

            # baseline = cierre de la semana anterior, o INITIAL_BANKROLL si no hay
            prev_ord = int(wkord) - 1
            if prev_ord in wk_final_by_order:
                baseline = float(wk_final_by_order[prev_ord])
            else:
                baseline = float(INITIAL_BANKROLL)

            # calcula carrera de la semana
            running = baseline
            last_final_bank = pd.NA

            for idx, row in df_week.iterrows():
                settled = bool(is_final_row.loc[idx])

                # calcula profit si falta y está final
                if settled and pd.isna(row.get("profit")):
                    prof = _compute_profit(row.get("stake"), row.get("decimal_odds"), df_week.at[idx, "result"])
                    df_week.at[idx, "profit"] = prof

                if settled and pd.notna(df_week.at[idx, "profit"]):
                    running += float(df_week.at[idx, "profit"])
                    df_week.at[idx, "bankroll_after"] = running
                    last_final_bank = running
                else:
                    df_week.at[idx, "bankroll_after"] = pd.NA

                # refuerza status finales
                if settled and pd.isna(df_week.at[idx, "status"]):
                    df_week.at[idx, "status"] = "FINAL"
                if settled and pd.isna(df_week.at[idx, "status_short"]):
                    df_week.at[idx, "status_short"] = "Final"

            # Cierre semanal
            week_complete = df_week["result"].notna().all()
            df_week["week_is_final"] = bool(week_complete)
            df_week["bankroll_week_final"] = last_final_bank if week_complete else pd.NA

            if week_complete and pd.notna(last_final_bank):
                wk_final_by_order[wkord] = float(last_final_bank)

            out_blocks.append(df_week)

    final_df = (
        pd.concat(out_blocks, ignore_index=True)
        .sort_values(["season","__sort_week","__sort_time"], kind="stable")
        .drop(columns=["__sort_week","__sort_time"], errors="ignore")
    )

    # 6) Eliminar la columna 'bankroll' si existe (ya no se usa)
    final_df = final_df.drop(columns=["bankroll"], errors="ignore")

    return final_df

def main():
    if not LIVE_BETS_FILE.exists():
        print("⚠️ No existe data/live/bets.csv; nada que procesar.")
        return

    # Carga
    bets = pd.read_csv(LIVE_BETS_FILE, low_memory=False)
    odds = _load_live_odds()

    # Procesa solo LIVE
    out = _process_live(bets, odds)

    # Guarda
    out = out.sort_values(["week_order", "schedule_date"], kind="stable")
    LIVE_BETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LIVE_BETS_FILE, index=False)
    print("✅ LIVE actualizado: team_score, opponent_score, status, status_short, result, profit, bankroll_after, week_is_final, bankroll_week_final. 'bankroll' eliminado.")

if __name__ == "__main__":
    main()

