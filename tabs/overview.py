# tabs/overview.py
import math
import pandas as pd
import streamlit as st
import altair as alt

from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import kpis_from_pnl, add_week_order, season_stage, norm_abbr, week_label_from_num
from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.charts import chart_last8_profit
from nfl_dash.components import bet_card


# ----------------------------
# Enriquecer bets con ESPN (scores/estado) sin debug
# ----------------------------
def _dominant_week(bets: pd.DataFrame) -> int | None:
    if "week" in bets.columns and pd.to_numeric(bets["week"], errors="coerce").notna().any():
        return int(pd.to_numeric(bets["week"], errors="coerce").dropna().mode().iloc[0])
    if "week_label" in bets.columns:
        vals = bets["week_label"].astype(str).str.extract(r"(\d+)")[0]
        vals = pd.to_numeric(vals, errors="coerce").dropna()
        if len(vals):
            return int(vals.mode().iloc[0])
    return None


def _espn_index_for_week(season: int, week: int) -> dict[tuple[str, str], dict]:
    """
    Devuelve un diccionario {(home_abbr, away_abbr) -> fila_dict} y también (away,home) -> fila_dict
    """
    try:
        es = fetch_espn_scoreboard_df(season=season, week=week)
    except Exception:
        es = pd.DataFrame()

    if es.empty:
        return {}

    es = es.copy()
    es["home_abbr"] = es["home_team"].astype(str).map(norm_abbr)
    es["away_abbr"] = es["away_team"].astype(str).map(norm_abbr)

    idx = {}
    for r in es.to_dict("records"):
        ha, aa = r["home_abbr"], r["away_abbr"]
        idx[(ha, aa)] = r
        idx[(aa, ha)] = r  # par invertido para emparejar por team/opponent + side
    return idx


def _enrich_bets_with_espn(bets: pd.DataFrame, season: int) -> pd.DataFrame:
    if bets.empty:
        return bets

    wk = _dominant_week(bets)
    if wk is None:
        return bets

    idx = _espn_index_for_week(season, wk)
    if not idx:
        return bets

    v = bets.copy()
    v["team"] = v.get("team", "").astype(str).map(norm_abbr)
    v["opponent"] = v.get("opponent", "").astype(str).map(norm_abbr)
    v["side"] = v.get("side", "").astype(str).str.lower()

    # Derivar home/away desde team/opponent + side
    home = []
    away = []
    for r in v.itertuples(index=False):
        t = getattr(r, "team", "")
        o = getattr(r, "opponent", "")
        s = getattr(r, "side", "")
        if s == "home":
            home.append(t); away.append(o)
        elif s == "away":
            home.append(o); away.append(t)
        else:
            # No hay side claro: intentamos adivinar con el índice
            if (t, o) in idx:
                ha = t if idx[(t, o)]["home_abbr"] == t else o
                aa = o if ha == t else t
                home.append(ha); away.append(aa)
            elif (o, t) in idx:
                ha = o if idx[(o, t)]["home_abbr"] == o else t
                aa = t if ha == o else o
                home.append(ha); away.append(aa)
            else:
                home.append(t); away.append(o)

    v["home_team"] = home
    v["away_team"] = away

    # Rellenar scores/estado si existe match en ESPN
    add_cols = {
        "home_score": [],
        "away_score": [],
        "state": [],
        "status_short": [],
        "start_time": [],
    }
    for r in v.itertuples(index=False):
        key = (getattr(r, "home_team"), getattr(r, "away_team"))
        srow = idx.get(key)
        if not srow:
            for k in add_cols: add_cols[k].append(pd.NA)
            continue
        add_cols["home_score"].append(srow.get("home_score"))
        add_cols["away_score"].append(srow.get("away_score"))
        add_cols["state"].append(str(srow.get("state") or "").lower())
        add_cols["status_short"].append(srow.get("short"))
        add_cols["start_time"].append(srow.get("start_time"))

    for k, vals in add_cols.items():
        v[k] = vals

    # Tipos
    if "start_time" in v.columns:
        v["start_time"] = pd.to_datetime(v["start_time"], errors="coerce", utc=True)
    for c in ("home_score", "away_score"):
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce")

    return v


# ----------------------------
# Bankroll chart con ancla 1000 al inicio
# ----------------------------
def _bankroll_chart_with_anchor(pnl: pd.DataFrame, height: int = 220):
    """
    Construye un dataframe con un punto 'Start' = 1000 y luego (Week N, bankroll) en orden.
    """
    pts = []
    pts.append({"x_label": "Start", "order": 0, "bankroll": 1000.0})

    if not pnl.empty and "bankroll" in pnl.columns:
        tmp = add_week_order(pnl[["week_label", "bankroll"]].copy())
        tmp = tmp.sort_values("__order")
        ord_base = 1
        for r in tmp.itertuples(index=False):
            pts.append({
                "x_label": str(getattr(r, "week_label")),
                "order": ord_base,
                "bankroll": float(getattr(r, "bankroll") or 0.0),
            })
            ord_base += 1

    df = pd.DataFrame(pts)
    domain = df.sort_values("order")["x_label"].tolist()

    return (
        alt.Chart(df, height=height)
        .mark_line(point=True)
        .encode(
            x=alt.X("x_label:N", sort=domain, title=None),
            y=alt.Y("bankroll:Q", title="Bankroll ($)"),
            tooltip=[alt.Tooltip("x_label:N", title="Week"),
                     alt.Tooltip("bankroll:Q", title="Bankroll", format="$.2f")],
        )
        .properties(width="container")
    )


# ----------------------------
# Render principal
# ----------------------------
def render(season: int):
    st.subheader("Overview")

    # Datos base
    pnl = load_pnl_weekly(season)
    stage = season_stage(season, pnl)
    bets_week_raw = load_bets_this_week(season) if stage == "in_season" else pd.DataFrame()

    # This Week’s Bets (estilo cards)
    if not bets_week_raw.empty:
        # Enriquecer con ESPN (scores/estado)
        try:
            view = _enrich_bets_with_espn(bets_week_raw, season=season)
        except Exception:
            view = bets_week_raw.copy()

        st.markdown("**This Week’s Bets**")

        # Orden visual por semana/kickoff si lo hay
        if "week_label" in view.columns:
            view["week_label"] = view["week_label"].astype(str)
            view["__order"] = view["week_label"].apply(lambda s: int(s.split()[-1]) if s.startswith("Week ") and s.split()[-1].isdigit() else 999)
            sort_cols = ["__order"]
            if "schedule_date" in view.columns:
                view["schedule_date"] = pd.to_datetime(view["schedule_date"], errors="coerce")
                sort_cols.append("schedule_date")
            view = view.sort_values(sort_cols).drop(columns="__order", errors="ignore")

        # Render de tarjetas (4 por fila)
        cards = list(view.itertuples(index=False))
        idx = 0
        cols_per_row = 4
        rows = math.ceil(len(cards) / cols_per_row)
        for _ in range(rows):
            col_objs = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if idx < len(cards):
                    with col_objs[j]:
                        bet_card(pd.Series(cards[idx]._asdict()))
                    idx += 1
        st.divider()

    # Season overview
    st.markdown("**Season Overview**")
    if pnl.empty:
        st.caption("No `pnl.csv` found for this season.")
        return

    # KPIs (forzamos initial=1000)
    _, _, total_profit, total_stake, yield_pct, profits, stakes = kpis_from_pnl(pnl)
    initial_bankroll = 1000.0
    final_bankroll = float(pnl["bankroll"].dropna().iloc[-1]) if "bankroll" in pnl.columns and pnl["bankroll"].notna().any() else initial_bankroll
    delta_bankroll = final_bankroll - initial_bankroll

    k1, k2 = st.columns(2)
    k1.metric("Initial", f"${initial_bankroll:,.2f}")
    k2.metric("Final", f"${final_bankroll:,.2f}", f"{delta_bankroll:,.2f}")

    # Gráficas (Bankroll con ancla 1000 + últimos profits)
    H_BANK = 300
    H_PROF = 300

    bank_chart = _bankroll_chart_with_anchor(pnl, height=H_BANK)

    cA, cB = st.columns(2)
    with cA:
        st.altair_chart(bank_chart, use_container_width=True)
    with cB:
        st.altair_chart(chart_last8_profit(pnl, profits, last=8, height=H_PROF), use_container_width=True)
