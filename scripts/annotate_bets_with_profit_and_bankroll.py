# scripts/annotate_bets_with_profit_and_bankroll.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

BETSPATH = Path("data/archive")  # raíz: season=YYYY/bets.csv


def _load_all_bets() -> pd.DataFrame:
    frames = []
    for p in sorted(BETSPATH.glob("season=*/bets.csv")):
        season = int(str(p.parent.name).split("=")[1])
        df = pd.read_csv(p, low_memory=False)
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def _decide_result(team_score, opp_score):
    if pd.isna(team_score) or pd.isna(opp_score):
        return pd.NA
    try:
        a, b = float(team_score), float(opp_score)
    except Exception:
        return pd.NA
    if a > b:
        return "WIN"
    elif a < b:
        return "LOSS"
    else:
        return "PUSH"


def _compute_profit(stake, decimal_odds, result):
    stake = _safe_num(stake)
    dec = _safe_num(decimal_odds)
    if pd.isna(stake) or pd.isna(dec) or pd.isna(result):
        return pd.NA
    if str(result) == "WIN":
        return stake * (dec - 1.0)
    if str(result) == "LOSS":
        return -stake
    if str(result) == "PUSH":
        return 0.0
    return pd.NA


def _normalize_week_order(df: pd.DataFrame) -> pd.Series:
    if "week_order" in df.columns:
        return _safe_num(df["week_order"]).fillna(999).astype(int)
    if "week" in df.columns:
        return _safe_num(df["week"]).fillna(999).astype(int)
    if "week_label" in df.columns:
        wk = df["week_label"].astype(str).str.extract(r"(\d+)")[0]
        return _safe_num(wk).fillna(999).astype(int)
    return pd.Series([999] * len(df), index=df.index, dtype=int)


def _write_back(df: pd.DataFrame):
    for season, g in df.groupby("season"):
        out = g.drop(columns=["__sort_week", "__sort_time"], errors="ignore").copy()
        out = out.sort_values(["week_order", "schedule_date"], kind="stable")
        f = BETSPATH / f"season={season}" / "bets.csv"
        out.to_csv(f, index=False)


def main():
    bets = _load_all_bets()
    if bets.empty:
        print("No bets found.")
        return

    # --- columnas requeridas / crear si faltan ---
    required_cols = [
        "stake", "profit", "decimal_odds", "ml", "schedule_date", "week_order",
        "team_score", "opponent_score", "status", "status_short",
        "result", "bankroll_after", "bankroll_week_final", "week_is_final", "bankroll"
    ]
    for c in required_cols:
        if c not in bets.columns:
            bets[c] = pd.NA

    # --- normalizaciones numéricas ---
    for col in ("stake", "profit", "decimal_odds", "ml"):
        bets[col] = _safe_num(bets[col])

    # schedule_date a datetime (UTC)
    bets["schedule_date"] = pd.to_datetime(bets["schedule_date"], errors="coerce", utc=True)

    # week_order si falta
    bets["week_order"] = _normalize_week_order(bets)

    # scores numéricos
    bets["team_score"] = _safe_num(bets["team_score"])
    bets["opponent_score"] = _safe_num(bets["opponent_score"])

    # settled si hay status FINAL o ambos scores
    has_final_status = bets.get("status", pd.Series([""] * len(bets), index=bets.index)).astype(str).str.upper().eq("FINAL")
    has_scores = bets["team_score"].notna() & bets["opponent_score"].notna()
    is_final_row = has_final_status | has_scores

    # result a partir de scores (si ya existía, se respeta; si no, se calcula)
    calc_result = [
        _decide_result(a, b) if (pd.notna(a) and pd.notna(b)) else pd.NA
        for a, b in zip(bets["team_score"], bets["opponent_score"])
    ]
    bets["result"] = bets["result"].where(bets["result"].notna(), calc_result)

    # Si hay scores (o status FINAL) y no hay status, marcar FINAL / Final
    needs_final_status = is_final_row & bets["status"].isna()
    bets.loc[needs_final_status, "status"] = "FINAL"
    bets.loc[needs_final_status, "status_short"] = "Final"
    # Si ya había status=FINAL pero status_short vacío, rellenar
    needs_short = has_final_status & bets["status_short"].isna()
    bets.loc[needs_short, "status_short"] = "Final"

    # Orden para procesar
    bets["__sort_week"] = bets["week_order"].astype(int)
    bets["__sort_time"] = bets["schedule_date"]

    out_blocks = []

    for season, df_season in bets.sort_values(["__sort_week", "__sort_time"], kind="stable").groupby("season", sort=True):
        df_season = df_season.copy()

        # bankroll de arranque de la temporada
        current_start = 1000.0

        for week, df_week in df_season.sort_values(["__sort_week", "__sort_time"], kind="stable").groupby("__sort_week", sort=True):
            df_week = df_week.copy().sort_values(["__sort_week", "__sort_time"], kind="stable")

            # bankroll inicial de ESTA semana
            df_week["bankroll"] = current_start

            running = float(current_start)
            last_final_bank = pd.NA

            for idx, row in df_week.iterrows():
                settled = bool(is_final_row.loc[idx])
                # Si está liquidada y no hay profit, calcularlo
                if settled and pd.isna(row.get("profit")):
                    prof = _compute_profit(row.get("stake"), row.get("decimal_odds"), df_week.at[idx, "result"])
                    df_week.at[idx, "profit"] = prof

                if settled and pd.notna(df_week.at[idx, "profit"]):
                    running += float(df_week.at[idx, "profit"])
                    df_week.at[idx, "bankroll_after"] = running
                    last_final_bank = running
                else:
                    df_week.at[idx, "bankroll_after"] = pd.NA

                # asegurar status final/short si hay scores aquí también
                if settled and pd.isna(df_week.at[idx, "status"]):
                    df_week.at[idx, "status"] = "FINAL"
                if settled and pd.isna(df_week.at[idx, "status_short"]):
                    df_week.at[idx, "status_short"] = "Final"

            week_complete = df_week["result"].notna().all()
            df_week["week_is_final"] = bool(week_complete)
            df_week["bankroll_week_final"] = last_final_bank if week_complete else pd.NA

            # Si la semana quedó completa y tenemos cierre, el start de la siguiente = cierre de esta
            if week_complete and pd.notna(last_final_bank):
                current_start = float(last_final_bank)

            out_blocks.append(df_week)

    final_df = (
        pd.concat(out_blocks, ignore_index=True)
        .sort_values(["season", "__sort_week", "__sort_time"], kind="stable")
    )

    _write_back(final_df)
    print("✅ Recalculado: result, profit, status/status_short, bankroll_after, bankroll_week_final, week_is_final y bankroll de inicio por semana.")


if __name__ == "__main__":
    main()
