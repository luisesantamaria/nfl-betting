# tabs/overview.py
import math
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import ORDER_INDEX, add_week_order, norm_abbr
from nfl_dash.charts import chart_last8_profit, chart_bankroll
from nfl_dash.components import bet_card
from nfl_dash.live_scores import fetch_espn_scoreboard_df


# -------------------------------------------------
# Helpers ESPN
# -------------------------------------------------
def _fetch_week_scores(season: int, week: int) -> pd.DataFrame:
    """
    Scoreboard ESPN -> agrega home_abbr / away_abbr para emparejar por siglas.
    """
    try:
        df = fetch_espn_scoreboard_df(season=season, week=int(week)).copy()
    except Exception:
        return pd.DataFrame(
            columns=[
                "season", "week", "start_time", "state", "short",
                "home_team", "home_score", "away_team", "away_score",
            ]
        )

    if df.empty:
        return df

    # Normalizamos nombres a siglas para facilitar match
    df["home_abbr"] = df["home_team"].astype(str).map(norm_abbr)
    df["away_abbr"] = df["away_team"].astype(str).map(norm_abbr)

    # Orden y tipos
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df["state"] = df.get("state", pd.Series(dtype=str)).astype(str).str.lower()

    return df


def _enrich_bets_with_espn(bets_df: pd.DataFrame, season: int, *, debug: bool = False) -> pd.DataFrame:
    """
    Empareja cada bet con el juego ESPN por equipos (siglas).
    - Agrega: home_team, away_team, home_score, away_score, state, status_short, start_time
    """
    if bets_df.empty:
        return bets_df

    v = bets_df.copy()

    # Definimos la semana objetivo desde el DataFrame (máxima semana es "esta semana")
    week_final = None
    if "week" in v.columns:
        try:
            week_final = int(pd.to_numeric(v["week"], errors="coerce").dropna().max())
        except Exception:
            week_final = None

    if week_final is None:
        # fallback: si no hay "week" usamos el mayor week_order
        if "week_order" in v.columns:
            week_final = int(pd.to_numeric(v["week_order"], errors="coerce").dropna().max())
        else:
            week_final = 1

    es = _fetch_week_scores(season=season, week=week_final)
    if es.empty:
        return v

    # Para depuración/inspección
    debug_scores = es[["home_abbr", "away_abbr", "home_score", "away_score", "start_time", "state", "short"]].copy()

    # Creamos clave de emparejamiento por conjunto {home,away}
    es["_key_set"] = es.apply(
        lambda r: frozenset({str(r.get("home_abbr", "")), str(r.get("away_abbr", ""))}),
        axis=1,
    )
    es_idx = es.set_index("_key_set")

    # Preparamos columnas de salida
    for col in ["home_team", "away_team", "home_score", "away_score", "state", "status_short", "start_time"]:
        if col not in v.columns:
            v[col] = pd.NA

    # Normalizamos team/opponent a siglas
    v["__team_abbr"] = v.get("team", "").astype(str).map(norm_abbr)
    v["__opp_abbr"]  = v.get("opponent", "").astype(str).map(norm_abbr)
    v["_key_set"] = v.apply(lambda r: frozenset({r["__team_abbr"], r["__opp_abbr"]}), axis=1)

    # Hacemos el match
    rows = []
    for i, row in v.iterrows():
        k = row["_key_set"]
        if k in es_idx.index:
            srow = es_idx.loc[k]
            # si hay múltiples (no es lo usual), tomamos el primero
            if isinstance(srow, pd.DataFrame):
                srow = srow.iloc[0]

            v.at[i, "home_team"]    = srow.get("home_abbr")
            v.at[i, "away_team"]    = srow.get("away_abbr")
            v.at[i, "home_score"]   = srow.get("home_score")
            v.at[i, "away_score"]   = srow.get("away_score")
            v.at[i, "state"]        = srow.get("state")
            v.at[i, "status_short"] = srow.get("short")
            v.at[i, "start_time"]   = srow.get("start_time")

            rows.append({
                "bet_idx": i,
                "home_abbr": srow.get("home_abbr"),
                "away_abbr": srow.get("away_abbr"),
                "home_score": srow.get("home_score"),
                "away_score": srow.get("away_score"),
                "start_time": srow.get("start_time"),
                "state": srow.get("state"),
            })

    if debug:
        with st.expander("Debug · ESPN matching (Overview)", expanded=False):
            st.markdown(f"**(season={season}, week={week_final}) — ESPN scoreboard**")
            st.dataframe(debug_scores, use_container_width=True)
            if rows:
                st.markdown("**Candidatos (emparejados por equipo)**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.caption("No hubo emparejamientos por equipo en esta vista.")

    # Limpieza columnas auxiliares
    v = v.drop(columns=["__team_abbr", "__opp_abbr", "_key_set"], errors="ignore")
    return v


# -------------------------------------------------
# Render principal
# -------------------------------------------------
def render(season: int):
    st.subheader("Overview")

    # Carga de datos base
    pnl = load_pnl_weekly(season)
    bets_week = load_bets_this_week(season)

    # Bets de esta semana (si hay)
    if not bets_week.empty:
        # Enriquecer con ESPN (scores + estado)
        debug = st.checkbox("Debug ESPN (Overview)", value=False)
        try:
            view = _enrich_bets_with_espn(bets_week, season=season, debug=debug)
        except Exception as e:
            st.error("Overview: fallo enriqueciendo bets con ESPN")
            st.exception(e)
            view = bets_week.copy()
    else:
        view = pd.DataFrame()

    if not view.empty:
        # Ordenar por (week_label -> ORDER_INDEX) y luego por horario si está
        if "week_label" in view.columns:
            view["week_label"] = view["week_label"].astype(str)
            view["__order"] = view["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
        else:
            view["__order"] = 999
        sort_cols = ["__order"]
        if "schedule_date" in view.columns:
            view["schedule_date"] = pd.to_datetime(view["schedule_date"], errors="coerce", utc=True)
            sort_cols.append("schedule_date")
        elif "start_time" in view.columns:
            # si no hay schedule_date, usamos start_time de ESPN
            view["start_time"] = pd.to_datetime(view["start_time"], errors="coerce", utc=True)
            sort_cols.append("start_time")

        view = view.sort_values(sort_cols).drop(columns="__order", errors="ignore")

        st.markdown("**This Week’s Bets**")

        # Pintar cards en 4 columnas
        cards = list(view.itertuples(index=False))
        idx = 0
        cols_per_row = 4
        rows = math.ceil(len(cards) / cols_per_row)
        for _ in range(rows):
            col_objs = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if idx < len(cards):
                    with col_objs[j]:
                        # pasamos la serie al card renderer
                        bet_card(pd.Series(cards[idx]._asdict()))
                    idx += 1

        st.divider()

    # === Season Overview ===
    st.markdown("**Season Overview**")
    if pnl.empty:
        st.caption("No `pnl_weekly_{year}.csv` found for this season.")
        return

    # KPI con bankroll inicial=1000 y acumulado por profits
    profits_series = pd.to_numeric(pnl.get("profit", pd.Series([0] * len(pnl))), errors="coerce").fillna(0.0)
    total_profit   = float(profits_series.sum())
    initial_bankroll = 1000.0
    final_bankroll   = initial_bankroll + total_profit
    total_stake      = float(pd.to_numeric(pnl.get("stake", pd.Series([0]*len(pnl))), errors="coerce").fillna(0.0).sum())
    yield_pct        = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Initial", f"${initial_bankroll:,.2f}")
    k2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-initial_bankroll):,.2f}")
    k3.metric("Total Profit", f"${total_profit:,.2f}")
    k4.metric("Yield",        f"{yield_pct:.2f}%")

    # Alturas dinámicas: si NO hay bets arriba, agrandamos ambos charts
    if view.empty:
        H_BANK = 380
        H_PROF = 380
    else:
        H_BANK = 200
        H_PROF = 200

    # Bankroll a partir de 1000 (profit acumulado)
    bank_df = pd.DataFrame({
        "week_label": pnl["week_label"].astype(str),
        "bankroll":   initial_bankroll + profits_series.cumsum()
    })
    bank_df = add_week_order(bank_df).sort_values("__order")

    cA, cB = st.columns(2)
    with cA:
        st.altair_chart(chart_bankroll(bank_df, height=H_BANK), use_container_width=True)
    with cB:
        st.altair_chart(chart_last8_profit(pnl, profits_series, last=8, height=H_PROF), use_container_width=True)
