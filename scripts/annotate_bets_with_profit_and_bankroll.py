# scripts/annotate_bets_with_profit_and_bankroll.py
from __future__ import annotations

import math
import pandas as pd
from pathlib import Path

BETSPATH = Path("data/archive")  # raíz donde están season=YYYY/bets.csv


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

    # Normalizaciones
    for col in ("stake", "profit", "decimal_odds", "ml"):
        if col in bets.columns:
            bets[col] = _safe_num(bets[col])

    # schedule_date para ordenar dentro de la semana
    if "schedule_date" in bets.columns:
        bets["schedule_date"] = pd.to_datetime(bets["schedule_date"], errors="coerce", utc=True)
    else:
        bets["schedule_date"] = pd.NaT

    # Asegurar week_order (si no existe, intentar derivarlo de week o week_label)
    if "week_order" not in bets.columns:
        if "week" in bets.columns:
            bets["week_order"] = _safe_num(bets["week"]).fillna(999).astype(int)
        elif "week_label" in bets.columns:
            wk = bets["week_label"].astype(str).str.extract(r"(\d+)")[0]
            bets["week_order"] = _safe_num(wk).fillna(999).astype(int)
        else:
            bets["week_order"] = 999

    # Garantizar columnas objetivo
    for col in ["result", "bankroll_after", "bankroll_week_final", "week_is_final"]:
        if col not in bets.columns:
            bets[col] = pd.NA

    # Siempre recalcular result a partir de los scores si el partido está finalizado
    # (o si ambos scores están disponibles).
    if "status" in bets.columns:
        st = bets["status"].astype(str).str.upper()
        is_final = st.eq("FINAL")
    else:
        # si no hay status, considera final cuando hay 2 scores
        is_final = bets["team_score"].notna() & bets["opponent_score"].notna()

    # Result: WIN/LOSS/PUSH cuando hay scores
    bets["result"] = [
        _decide_result(a, b) if f else pd.NA
        for a, b, f in zip(bets.get("team_score"), bets.get("opponent_score"), (is_final | (bets["team_score"].notna() & bets["opponent_score"].notna())))
    ]

    # Orden “total” para procesar secuencialmente por temporada y semana
    bets["__sort_week"] = bets["week_order"].astype(int)
    # Si no hay fecha, ponemos NaT; al ordenar, NaT va al final
    bets["__sort_time"] = bets["schedule_date"]

    # Recalcular bankroll_after y flags por (season, week)
    out = []
    for season, g_season in bets.sort_values(["__sort_week", "__sort_time"], kind="stable").groupby("season", sort=True):
        g_season = g_season.copy()

        # Construir índice de bankroll final para semana previa
        # (lo recalculamos aquí mismo para no depender de versiones previas)
        prev_week_final = {}

        # ordenamos por semana y por hora dentro de semana
        for wk, g_week in g_season.sort_values(["__sort_week", "__sort_time"], kind="stable").groupby("__sort_week", sort=True):
            g_week = g_week.copy().sort_values(["__sort_week", "__sort_time"], kind="stable")

            # Bankroll inicial de esta semana: bankroll final de la semana anterior si existe; si no, 1000
            start_bank = prev_week_final.get(wk - 1, 1000.0)

            # Setear 'bankroll' (columna base) como el bankroll de inicio de semana para todas las filas
            g_week["bankroll"] = start_bank

            running = float(start_bank)
            last_final_bank = pd.NA
            any_final = False

            # Recorremos cronológicamente
            for i, row in g_week.iterrows():
                settled = bool(is_final.loc[i]) or (
                    pd.notna(row.get("team_score")) and pd.notna(row.get("opponent_score"))
                )

                # Si está final, su bankroll_after = running + profit
                if settled and pd.notna(row.get("profit")):
                    any_final = True
                    running = float(running) + float(row["profit"])
                    g_week.at[i, "bankroll_after"] = running
                    last_final_bank = running
                else:
                    # Si no está final, lo dejamos en NA (no queremos arrastrar “supuestos”)
                    g_week.at[i, "bankroll_after"] = pd.NA

            # Flags de semana
            g_week["week_is_final"] = bool(any_final and g_week["result"].notna().all())
            g_week["bankroll_week_final"] = last_final_bank

            # Guardar para que la próxima semana arranque desde aquí
            if pd.notna(last_final_bank):
                prev_week_final[wk] = float(last_final_bank)

            out.append(g_week)

        # fin for wk
    # fin for season

    final_df = pd.concat(out, ignore_index=True).sort_values(
        ["season", "__sort_week", "__sort_time"], kind="stable"
    )

    _write_back(final_df)
    print("✅ annotate_bets_with_profit_and_bankroll: recalculado result y bankroll_after.")


if __name__ == "__main__":
    main()
