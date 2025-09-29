# tabs/bets.py
import math
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_ledger, load_bets_this_week
from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.utils import ORDER_INDEX, week_label_to_num, norm_abbr
from nfl_dash.components import bet_card as render_bet_card


# -----------------------------
# Enriquecer bets con scoreboard ESPN (múltiples semanas)
# -----------------------------
def _enrich_bets_with_espn_all_weeks(bets_df: pd.DataFrame, season: int) -> pd.DataFrame:
    if bets_df.empty:
        return bets_df.copy()

    df = bets_df.copy()

    # Normalizar semana
    if "week" not in df.columns:
        if "week_label" in df.columns:
            df["week"] = df["week_label"].apply(week_label_to_num)
        else:
            df["week"] = pd.NA

    # Columnas de salida esperadas por bet_card
    for c in ("score_home", "score_away", "short", "status"):
        if c not in df.columns:
            df[c] = pd.NA

    # Por cada semana presente en las bets, pedimos el scoreboard
    weeks = (
        pd.to_numeric(df["week"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    for wk in weeks:
        sb = fetch_espn_scoreboard_df(season=int(season), week=int(wk))
        if sb is None or sb.empty:
            continue

        sb = sb.copy()
        sb["home_abbr"] = sb["home_team"].astype(str).map(norm_abbr)
        sb["away_abbr"] = sb["away_team"].astype(str).map(norm_abbr)

        # Índice por (home, away) y (away, home)
        idx = {}
        for r in sb.itertuples(index=False):
            key1 = (r.home_abbr, r.away_abbr)
            key2 = (r.away_abbr, r.home_abbr)
            idx[key1] = r
            idx[key2] = r

        sel = df["week"].eq(wk)
        for i, r in df[sel].iterrows():
            team = norm_abbr(str(r.get("team", "")))
            opp  = norm_abbr(str(r.get("opponent", "")))
            home = norm_abbr(str(r.get("home_team", "")))
            away = norm_abbr(str(r.get("away_team", "")))
            side = str(r.get("side", "")).lower()

            # Resolver home/away de la apuesta
            h = a = ""
            if home and away:
                h, a = home, away
            elif team and opp:
                if side == "home":
                    h, a = team, opp
                elif side == "away":
                    h, a = opp, team
                else:
                    # Desconocido: intentamos igualmente ambas direcciones
                    h, a = team, opp

            srow = idx.get((h, a))
            if srow:
                df.at[i, "score_home"] = srow.home_score
                df.at[i, "score_away"] = srow.away_score
                df.at[i, "short"]       = srow.short
                df.at[i, "status"]      = str(srow.state).upper()

    return df


def render(season: int):
    st.subheader("Bets")

    # 1) Ledger de la temporada (archivo de archive)
    bets_all_raw = load_ledger(season)

    # 2) Fallback para temporada actual si no hay ledger todavía:
    #    usa las apuestas de la semana vigente (para no dejar la pestaña vacía)
    if bets_all_raw.empty:
        wk = load_bets_this_week(season)
        if not wk.empty:
            bets_all_raw = wk.copy()

    if bets_all_raw.empty:
        st.caption("No bets file for this season.")
        return

    # Enriquecer TODAS las semanas con ESPN (scores/estado)
    bets_all = _enrich_bets_with_espn_all_weeks(bets_all_raw, season)

    # Ordenar por semana y kickoff si existe
    view = bets_all.copy()
    if "week_label" in view.columns:
        view["week_label"] = view["week_label"].astype(str)
        view["__order"] = view["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    elif "week" in view.columns:
        view["__order"] = pd.to_numeric(view["week"], errors="coerce").fillna(999).astype(int)
        view["week_label"] = view["week"].apply(lambda x: f"Week {int(x)}" if pd.notna(x) else "Week 999")
    else:
        view["__order"] = 999
        view["week_label"] = "Week 999"

    sort_cols = ["__order"]
    if "schedule_date" in view.columns:
        view["schedule_date"] = pd.to_datetime(view["schedule_date"], errors="coerce")
        sort_cols.append("schedule_date")
    view = view.sort_values(sort_cols, kind="stable")

    # Render por semanas con encabezado
    for order_val in sorted(view["__order"].unique()):
        chunk = view[view["__order"] == order_val]
        if chunk.empty:
            continue
        header = str(chunk.iloc[0]["week_label"])
        st.markdown(f"### {header}")

        cards = list(chunk.itertuples(index=False))
        idx = 0
        cols_per_row = 4
        rows = math.ceil(len(cards) / cols_per_row)
        for _ in range(rows):
            col_objs = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if idx < len(cards):
                    with col_objs[j]:
                        render_bet_card(pd.Series(cards[idx]._asdict()))
                    idx += 1)
