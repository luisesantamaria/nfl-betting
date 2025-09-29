# tabs/overview.py
import math
import pandas as pd
import altair as alt
import streamlit as st

from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import (
    kpis_from_pnl,
    season_stage,
    ORDER_INDEX,
    week_label_to_num,
    norm_abbr,
)
from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.components import bet_card as render_bet_card
from nfl_dash.charts import chart_last8_profit


# ============
# Helpers ESPN
# ============
def _enrich_bets_with_espn_this_week(bets_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Enriquece SOLO la semana visible (bets_df) con scores/estado desde ESPN."""
    if bets_df.empty:
        return bets_df

    v = bets_df.copy()
    # normaliza semana
    if "week" not in v.columns:
        if "week_label" in v.columns:
            v["week"] = v["week_label"].apply(week_label_to_num)
        else:
            v["week"] = pd.NA

    # columnas que usa bet_card
    for c in ("score_home", "score_away", "short", "status"):
        if c not in v.columns:
            v[c] = pd.NA

    wk = pd.to_numeric(v["week"], errors="coerce").dropna().astype(int).unique()
    if len(wk) == 0:
        return v
    week = int(wk[0])

    sb = fetch_espn_scoreboard_df(season=int(season), week=week)
    if sb is None or sb.empty:
        return v

    sb = sb.copy()
    sb["home_abbr"] = sb["home_team"].astype(str).map(norm_abbr)
    sb["away_abbr"] = sb["away_team"].astype(str).map(norm_abbr)

    # índice (home,away) y espejo
    idx = {}
    for r in sb.itertuples(index=False):
        key1 = (r.home_abbr, r.away_abbr)
        key2 = (r.away_abbr, r.home_abbr)
        idx[key1] = r
        idx[key2] = r

    for i, r in v.iterrows():
        team = norm_abbr(str(r.get("team", "")))
        opp  = norm_abbr(str(r.get("opponent", "")))
        home = norm_abbr(str(r.get("home_team", "")))
        away = norm_abbr(str(r.get("away_team", "")))
        side = str(r.get("side", "")).lower()

        h = a = ""
        if home and away:
            h, a = home, away
        elif team and opp:
            if side == "home":
                h, a = team, opp
            elif side == "away":
                h, a = opp, team
            else:
                h, a = team, opp

        srow = idx.get((h, a))
        if srow:
            v.at[i, "score_home"] = srow.home_score
            v.at[i, "score_away"] = srow.away_score
            v.at[i, "short"]       = srow.short
            v.at[i, "status"]      = str(srow.state).upper()

    return v


# ===================
# Bankroll line chart
# ===================
def _bankroll_chart_from_pnl(pnl: pd.DataFrame, height: int = 260) -> alt.Chart:
    """
    Línea de bankroll que:
    - agrega un punto inicial sintético en $1000 pegado al eje Y
    - muestra etiquetas del eje X solo para semanas reales
    - ajusta el dominio Y alrededor de los datos (no desde 0)
    """
    if pnl.empty or "bankroll" not in pnl.columns:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()

    df = pnl.copy()
    df["week_label"] = df["week_label"].astype(str)
    df["__order"] = df["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    df = df.sort_values("__order")

    # serie real (sin start)
    real_orders = df["__order"].tolist()
    real_labels = df["week_label"].tolist()

    # punto inicial sintético en $1000 (pegado al eje Y)
    start_order = real_orders[0] - 0.01
    start_row = pd.DataFrame(
        {
            "week_label": [real_labels[0]],  # etiqueta de la 1ª semana (no se muestra)
            "bankroll": [1000.0],
            "__order": [start_order],
        }
    )

    plot_df = pd.concat(
        [start_row[["week_label", "bankroll", "__order"]], df[["week_label", "bankroll", "__order"]]],
        ignore_index=True,
    )

    # dominio Y ajustado con padding leve
    vals = pd.to_numeric(plot_df["bankroll"], errors="coerce").dropna()
    vmin, vmax = float(vals.min()), float(vals.max())
    pad = max(1.0, 0.01 * max(vmax - vmin, 1.0))
    y_domain = [vmin - pad, vmax + pad]

    # mapeo de order->label para el eje X (solo semanas reales)
    orders_labels = {int(o): lbl for o, lbl in zip(real_orders, real_labels)}
    js_map = "{" + ",".join([f"{int(o)}:'{lbl}'" for o, lbl in orders_labels.items()]) + "}"
    label_expr = f"({js_map})[datum.value]"

    base = alt.Chart(plot_df, height=height).encode(
        x=alt.X(
            "__order:Q",
            axis=alt.Axis(
                title=None,
                values=real_orders,     # solo ticks en semanas reales
                labelExpr=label_expr,   # etiquetas = semanas
                labelAngle=0,
            ),
            scale=alt.Scale(nice=False, zero=False),
        ),
        y=alt.Y(
            "bankroll:Q",
            title="Bankroll ($)",
            scale=alt.Scale(domain=y_domain, nice=False, zero=False),
        ),
        tooltip=[alt.Tooltip("week_label:N", title="Week"),
                 alt.Tooltip("bankroll:Q", title="Bankroll", format="$.2f")],
    )

    # estética más limpia
    line = base.mark_line(point=True, strokeWidth=2.2)
    return line.properties(width="container")


# =======
# Render
# =======
def render(season: int):
    st.subheader("Overview")

    # Datos base
    pnl = load_pnl_weekly(season)
    stage = season_stage(season, pnl)
    bets_week = load_bets_this_week(season) if stage == "in_season" else pd.DataFrame()

    # Bets de esta semana (si hay) + ESPN scores
    if not bets_week.empty:
        bets_week = _enrich_bets_with_espn_this_week(bets_week, season)

        st.markdown("**This Week’s Bets**")
        cards = list(bets_week.itertuples(index=False))
        idx = 0
        cols_per_row = 4
        rows = math.ceil(len(cards) / cols_per_row)
        for _ in range(rows):
            col_objs = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if idx < len(cards):
                    with col_objs[j]:
                        render_bet_card(pd.Series(cards[idx]._asdict()))
                    idx += 1
        st.divider()

    # Season Overview
    st.markdown("**Season Overview**")
    if pnl.empty:
        st.caption("No `pnl.csv` found for this season.")
        return

    # KPIs
    # Initial fijo en 1000; Final desde pnl o acumulando profit si no hay bankroll
    initial_bankroll = 1000.0
    if "bankroll" in pnl.columns and pnl["bankroll"].notna().any():
        final_bankroll = float(pd.to_numeric(pnl["bankroll"], errors="coerce").dropna().iloc[-1])
    else:
        total_profit = float(pd.to_numeric(pnl.get("profit", pd.Series([0])), errors="coerce").fillna(0).sum())
        final_bankroll = initial_bankroll + total_profit
    profit_total = final_bankroll - initial_bankroll

    k1, k2 = st.columns(2)
    with k1:
        st.metric("Initial", f"${initial_bankroll:,.2f}")
    with k2:
        st.metric("Final", f"${final_bankroll:,.2f}", f"{profit_total:,.2f}")

    # Gráficos lado a lado: (izq) bankroll desde 1000 pegado a eje Y, (der) last-8 profits
    profits_series = pd.to_numeric(pnl.get("profit", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    cA, cB = st.columns(2)
    with cA:
        st.altair_chart(_bankroll_chart_from_pnl(pnl, height=260), use_container_width=True)
    with cB:
        st.altair_chart(chart_last8_profit(pnl, profits_series, last=8, height=260), use_container_width=True)
