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
    # Guardar devuelta por temporada
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

    # Normalizaciones numéricas básicas
    for col in ("stake", "profit", "decimal_odds", "ml"):
        if col in bets.columns:
            bets[col] = _safe_num(bets[col])

    # schedule_date para ordenar dentro de semana
    if "schedule_date" in bets.columns:
        bets["schedule_date"] = pd.to_datetime(bets["schedule_date"], errors="coerce", utc=True)
    else:
        bets["schedule_date"] = pd.NaT

    # Asegurar week_order
    if "week_order" not in bets.columns:
        if "week" in bets.columns:
            bets["week_order"] = _safe_num(bets["week"]).fillna(999).astype(int)
        elif "week_label" in bets.columns:
            wk = bets["week_label"].astype(str).str.extract(r"(\d+)")[0]
            bets["week_order"] = _safe_num(wk).fillna(999).astype(int)
        else:
            bets["week_order"] = 999

    # Asegurar columnas que pueden faltar
    for col in ["team_score", "opponent_score", "status", "status_short",
                "result", "bankroll_after", "bankroll_week_final", "week_is_final", "bankroll"]:
        if col not in bets.columns:
            bets[col] = pd.NA

    # Scores como numérico (si existen)
    team_score = _safe_num(bets.get("team_score", pd.Series([pd.NA] * len(bets))))
    opp_score  = _safe_num(bets.get("opponent_score", pd.Series([pd.NA] * len(bets))))
    bets["team_score"] = team_score
    bets["opponent_score"] = opp_score

    # is_final: usa status==FINAL si existe; si no, ambos scores presentes
    if "status" in bets.columns:
        is_final = bets["status"].astype(str).str.upper().eq("FINAL")
    else:
        is_final = team_score.notna() & opp_score.notna()

    # Recalcular siempre el result cuando hay scores
    bets["result"] = [
        _decide_result(a, b) if (pd.notna(a) and pd.notna(b)) else pd.NA
        for a, b in zip(team_score, opp_score)
    ]

    # Orden global para procesar secuencialmente por (season, week, kickoff)
    bets["__sort_week"] = bets["week_order"].astype(int)
    bets["__sort_time"] = bets["schedule_date"]

    out = []
    # Mapa: semana -> bankroll final (solo si la semana quedó COMPLETA)
    for season, g_season in bets.sort_values(["__sort_week", "__sort_time"], kind="stable").groupby("season", sort=True):
        g_season = g_season.copy()
        prev_week_final = {}  # {week_order: bankroll_final}

        for wk, g_week in g_season.sort_values(["__sort_week", "__sort_time"], kind="stable").groupby("__sort_week", sort=True):
            g_week = g_week.copy().sort_values(["__sort_week", "__sort_time"], kind="stable")

            # Bankroll inicial de esta semana = bankroll final de semana anterior si cerró; si no, 1000
            start_bank = prev_week_final.get(wk - 1, 1000.0)
            g_week["bankroll"] = float(start_bank)

            running = float(start_bank)
            last_final_bank = pd.NA

            # Avanzar por cada apuesta en el orden cronológico
            for i, row in g_week.iterrows():
                settled = bool(is_final.loc[i]) or (
                    pd.notna(row.get("team_score")) and pd.notna(row.get("opponent_score"))
                )
                if settled and pd.notna(row.get("profit")):
                    running = running + float(row["profit"])
                    g_week.at[i, "bankroll_after"] = running
                    last_final_bank = running
                else:
                    g_week.at[i, "bankroll_after"] = pd.NA

            # ¿La semana está completamente finalizada?
            all_final = g_week["result"].notna().all()
            g_week["week_is_final"] = bool(all_final)
            g_week["bankroll_week_final"] = last_final_bank if all_final else pd.NA

            # Solo si la semana quedó completa, ese será el bankroll inicial de la semana siguiente
            if all_final and pd.notna(last_final_bank):
                prev_week_final[wk] = float(last_final_bank)

            out.append(g_week)

    final_df = pd.concat(out, ignore_index=True).sort_values(
        ["season", "__sort_week", "__sort_time"], kind="stable"
    )

    _write_back(final_df)
    print("✅ annotate_bets_with_profit_and_bankroll: recalculado result y bankroll_after/semana.")


if __name__ == "__main__":
    main()
