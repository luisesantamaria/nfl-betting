# scripts/annotate_bets_with_profit_and_bankroll.py
from __future__ import annotations

import pandas as pd
from pathlib import Path

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

    # --- normalizaciones ---
    for col in ("stake", "profit", "decimal_odds", "ml"):
        if col in bets.columns:
            bets[col] = _safe_num(bets[col])

    if "schedule_date" in bets.columns:
        bets["schedule_date"] = pd.to_datetime(bets["schedule_date"], errors="coerce", utc=True)
    else:
        bets["schedule_date"] = pd.NaT

    if "week_order" not in bets.columns:
        if "week" in bets.columns:
            bets["week_order"] = _safe_num(bets["week"]).fillna(999).astype(int)
        elif "week_label" in bets.columns:
            wk = bets["week_label"].astype(str).str.extract(r"(\d+)")[0]
            bets["week_order"] = _safe_num(wk).fillna(999).astype(int)
        else:
            bets["week_order"] = 999

    for col in [
        "team_score", "opponent_score", "status", "status_short",
        "result", "bankroll_after", "bankroll_week_final",
        "week_is_final", "bankroll"
    ]:
        if col not in bets.columns:
            bets[col] = pd.NA

    # scores numéricos
    bets["team_score"] = _safe_num(bets["team_score"])
    bets["opponent_score"] = _safe_num(bets["opponent_score"])

    # settled por status FINAL o presencia de ambos scores
    has_final_status = bets.get("status", pd.Series([""] * len(bets))).astype(str).str.upper().eq("FINAL")
    has_scores = bets["team_score"].notna() & bets["opponent_score"].notna()
    is_final_row = has_final_status | has_scores

    # result siempre que haya scores
    bets["result"] = [
        _decide_result(a, b) if (pd.notna(a) and pd.notna(b)) else pd.NA
        for a, b in zip(bets["team_score"], bets["opponent_score"])
    ]

    # orden para procesar
    bets["__sort_week"] = bets["week_order"].astype(int)
    bets["__sort_time"] = bets["schedule_date"]

    out_blocks = []

    for season, df_season in bets.sort_values(["__sort_week", "__sort_time"], kind="stable").groupby("season", sort=True):
        df_season = df_season.copy()

        # bankroll de arranque de la temporada
        current_start = 1000.0

        for week, df_week in df_season.sort_values(["__sort_week", "__sort_time"], kind="stable").groupby("__sort_week", sort=True):
            df_week = df_week.copy().sort_values(["__sort_week", "__sort_time"], kind="stable")

            # bankroll inicial de ESTA semana (lo que cerró la anterior si cerró)
            df_week["bankroll"] = current_start

            running = float(current_start)
            last_final_bank = pd.NA

            for idx, row in df_week.iterrows():
                settled = bool(is_final_row.loc[idx])
                if settled and pd.notna(row.get("profit")):
                    running += float(row["profit"])
                    df_week.at[idx, "bankroll_after"] = running
                    last_final_bank = running
                else:
                    df_week.at[idx, "bankroll_after"] = pd.NA

            week_complete = df_week["result"].notna().all()
            df_week["week_is_final"] = bool(week_complete)
            df_week["bankroll_week_final"] = last_final_bank if week_complete else pd.NA

            # si la semana quedó completa, el start de la siguiente semana = cierre de esta
            if week_complete and pd.notna(last_final_bank):
                current_start = float(last_final_bank)

            out_blocks.append(df_week)

    final_df = (
        pd.concat(out_blocks, ignore_index=True)
        .sort_values(["season", "__sort_week", "__sort_time"], kind="stable")
    )

    _write_back(final_df)
    print("✅ Recalculado: result, bankroll_after, bankroll_week_final, week_is_final, y bankroll de inicio por semana.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
