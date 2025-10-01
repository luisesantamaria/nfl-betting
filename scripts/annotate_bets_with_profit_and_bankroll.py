# scripts/annotate_bets_with_profit_and_bankroll.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

LIVE_BETS_FILE = Path("data/live/bets.csv")
ARCHIVE_ROOT = Path("data/archive")  # opcional: season=YYYY/bets.csv


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
    result = str(result)
    if result == "WIN":
        return stake * (dec - 1.0)
    if result == "LOSS":
        return -stake
    if result == "PUSH":
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


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "stake", "profit", "decimal_odds", "ml", "schedule_date", "week_order",
        "team_score", "opponent_score", "status", "status_short",
        "result", "bankroll_after", "bankroll_week_final", "week_is_final", "bankroll",
        "season",
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _prepare_types(df: pd.DataFrame) -> pd.DataFrame:
    # numéricos básicos
    for col in ("stake", "profit", "decimal_odds", "ml"):
        df[col] = _safe_num(df[col])

    # fechas
    df["schedule_date"] = pd.to_datetime(df.get("schedule_date"), errors="coerce", utc=True)

    # week_order
    df["week_order"] = _normalize_week_order(df)

    # scores
    df["team_score"] = _safe_num(df["team_score"])
    df["opponent_score"] = _safe_num(df["opponent_score"])

    # season
    df["season"] = _safe_num(df["season"]).astype("Int64")

    return df


def _process_bets_dataframe(bets: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa un DataFrame (puede contener varias seasons) y devuelve el mismo DF con:
    result, profit (si faltaba), status/status_short, bankroll_after,
    week_is_final, bankroll_week_final y bankroll (inicio de semana).
    """
    if bets.empty:
        return bets

    bets = _ensure_columns(bets.copy())
    bets = _prepare_types(bets)

    # settled si hay status FINAL o ambos scores
    has_final_status = bets.get("status", pd.Series([""] * len(bets), index=bets.index)).astype(str).str.upper().eq("FINAL")
    has_scores = bets["team_score"].notna() & bets["opponent_score"].notna()
    is_final_row = has_final_status | has_scores

    # result (si falta) a partir de scores
    calc_result = [
        _decide_result(a, b) if (pd.notna(a) and pd.notna(b)) else pd.NA
        for a, b in zip(bets["team_score"], bets["opponent_score"])
    ]
    bets["result"] = bets["result"].where(bets["result"].notna(), calc_result)

    # status / status_short (si falta y hay fila final)
    needs_final_status = is_final_row & bets["status"].isna()
    bets.loc[needs_final_status, "status"] = "FINAL"
    bets.loc[needs_final_status, "status_short"] = "Final"
    needs_short = has_final_status & bets["status_short"].isna()
    bets.loc[needs_short, "status_short"] = "Final"

    # orden de proceso
    bets["__sort_week"] = bets["week_order"].astype(int)
    bets["__sort_time"] = bets["schedule_date"]

    out_blocks = []

    # Recorremos por temporada y por semana en orden temporal
    for season, df_season in bets.sort_values(["season", "__sort_week", "__sort_time"], kind="stable").groupby("season", sort=True):
        df_season = df_season.copy()

        # bankroll de arranque de la temporada (puedes cambiar si lo llevas en otro lado)
        current_start = 1000.0

        for week, df_week in df_season.sort_values(["__sort_week", "__sort_time"], kind="stable").groupby("__sort_week", sort=True):
            df_week = df_week.copy().sort_values(["__sort_week", "__sort_time"], kind="stable")

            # bankroll inicial de ESTA semana
            df_week["bankroll"] = current_start

            running = float(current_start)
            last_final_bank = pd.NA

            for idx, row in df_week.iterrows():
                settled = bool(is_final_row.loc[idx])

                # calcula profit si está liquidada y falta
                if settled and pd.isna(row.get("profit")):
                    prof = _compute_profit(row.get("stake"), row.get("decimal_odds"), df_week.at[idx, "result"])
                    df_week.at[idx, "profit"] = prof

                if settled and pd.notna(df_week.at[idx, "profit"]):
                    running += float(df_week.at[idx, "profit"])
                    df_week.at[idx, "bankroll_after"] = running
                    last_final_bank = running
                else:
                    df_week.at[idx, "bankroll_after"] = pd.NA

                # refuerza status finales si aplica
                if settled and pd.isna(df_week.at[idx, "status"]):
                    df_week.at[idx, "status"] = "FINAL"
                if settled and pd.isna(df_week.at[idx, "status_short"]):
                    df_week.at[idx, "status_short"] = "Final"

            week_complete = df_week["result"].notna().all()
            df_week["week_is_final"] = bool(week_complete)
            df_week["bankroll_week_final"] = last_final_bank if week_complete else pd.NA

            # siguiente semana arranca con el cierre de esta (si cerró)
            if week_complete and pd.notna(last_final_bank):
                current_start = float(last_final_bank)

            out_blocks.append(df_week)

    final_df = (
        pd.concat(out_blocks, ignore_index=True)
        .sort_values(["season", "__sort_week", "__sort_time"], kind="stable")
        .drop(columns=["__sort_week", "__sort_time"], errors="ignore")
    )
    return final_df


def _write_back_archive(df: pd.DataFrame):
    for season, g in df.groupby("season"):
        out = g.sort_values(["week_order", "schedule_date"], kind="stable").copy()
        (ARCHIVE_ROOT / f"season={int(season)}").mkdir(parents=True, exist_ok=True)
        f = ARCHIVE_ROOT / f"season={int(season)}" / "bets.csv"
        out.to_csv(f, index=False)


def _write_back_live(df: pd.DataFrame):
    out = df.sort_values(["week_order", "schedule_date"], kind="stable").copy()
    LIVE_BETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LIVE_BETS_FILE, index=False)


def main():
    processed_any = False

    # 1) Procesar LIVE si existe
    if LIVE_BETS_FILE.exists():
        live_df = pd.read_csv(LIVE_BETS_FILE, low_memory=False)
        live_df = _process_bets_dataframe(live_df)
        _write_back_live(live_df)
        print(f"✅ Actualizado LIVE: {LIVE_BETS_FILE}")
        processed_any = True
    else:
        print("ℹ️ No existe data/live/bets.csv; se omite LIVE.")

    # 2) Procesar ARCHIVE si existiera estructura por temporada
    archive_frames = []
    for p in sorted(ARCHIVE_ROOT.glob("season=*/bets.csv")):
        try:
            season = int(str(p.parent.name).split("=")[1])
        except Exception:
            continue
        df = pd.read_csv(p, low_memory=False)
        df["season"] = season
        archive_frames.append(df)

    if archive_frames:
        all_archive = pd.concat(archive_frames, ignore_index=True)
        all_archive = _process_bets_dataframe(all_archive)
        _write_back_archive(all_archive)
        print("✅ Actualizados ARCHIVE: data/archive/season=*/bets.csv")
        processed_any = True
    else:
        print("ℹ️ No hay archivos en data/archive/season=*/bets.csv; se omite ARCHIVE.")

    if not processed_any:
        print("⚠️ No se encontraron bets para procesar.")


if __name__ == "__main__":
    main()

