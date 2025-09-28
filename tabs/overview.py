# tabs/overview.py
import math
import pandas as pd
import streamlit as st
import altair as alt

from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import (
    kpis_from_pnl,
    add_week_order,
    season_stage,
    norm_abbr,
)
from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.charts import chart_last8_profit
from nfl_dash.components import bet_card


# ----------------------------
# Enriquecer bets con ESPN (scores/estado)
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
    """ {(home_abbr, away_abbr) -> fila_dict} y también (away,home) -> fila_dict """
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
        idx[(aa, ha)] = r
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

    home, away = [], []
    for r in v.itertuples(index=False):
        t = getattr(r, "team", "")
        o = getattr(r, "opponent", "")
        s = getattr(r, "side", "")
        if s == "home":
            home.append(t); away.append(o)
        elif s == "away":
            home.append(o); away.append(t)
        else:
            if (t, o) in idx:
                ha = idx[(t, o)]["home_abbr"]
                aa = idx[(t, o)]["away_abbr"]
                home.append(ha); away.append(aa)
            elif (o, t) in idx:
                ha = idx[(o, t)]["home_abbr"]
                aa = idx[(o, t)]["away_abbr"]
                home.append(ha); away.append(aa)
            else:
                home.append(t); away.append(o)

    v["home_team"] = home
    v["away_team"] = away

    add_cols = { "home_score": [], "away_score": [], "state": [], "status_short": [], "start_time": [] }
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

    if "start_time" in v.columns:
        v["start_time"] = pd.to_datetime(v["start_time"], errors="coerce", utc=True)
    for c in ("home_score", "away_score"):
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce")

    return v


# ----------------------------
# Bankroll chart pegado al eje Y y arrancando en 1000
# ----------------------------
def _bankroll_chart_snug(pnl: pd.DataFrame, height: int = 220) -> alt.Chart:
    """
    - Eje X cuantitativo 0..N-1 -> primer punto x=0 cae sobre el eje Y.
    - Se inserta un punto inicial (x=0, y=1000) y se oculta su etiqueta (queda vacío).
    - Las etiquetas visibles empiezan en Week 3, Week 4, ...
    - Eje Y con dominio ajustado (no arranca en 0) y pad mínimo.
    """
    # Base sin estar vacía
    core = add_week_order(pnl[["week_label", "bankroll"]].copy()) if (not pnl.empty and "bankroll" in pnl.columns) else pd.DataFrame(columns=["week_label", "bankroll", "__order"])
    if core.empty:
        base = pd.DataFrame({"week_label": [], "bankroll": [], "__order": []})
    else:
        core = core.sort_values("__order")
        base = core[["week_label", "bankroll"]].copy()

    # Insertar punto inicial 1000
    init_row = pd.DataFrame({"week_label": ["Initial"], "bankroll": [1000.0]})
    df = pd.concat([init_row, base], ignore_index=True)
    df["idx"] = range(len(df))  # 0..N-1

    # Etiquetas: ocultar la de idx 0 (punto Initial)
    labels = [""] + base["week_label"].astype(str).tolist()
    n = len(labels)
    label_array = "[" + ",".join([f"'{s}'" for s in labels]) + "]"
    label_expr = f"datum.value === 0 ? '' : {label_array}[datum.value]"

    # Y-scale ajustado al rango real
    vals = pd.to_numeric(df["bankroll"], errors="coerce").dropna()
    if len(vals):
        vmin, vmax = float(vals.min()), float(vals.max())
        pad = max(1.0, 0.01 * max(vmax - vmin, 1.0))
        y_scale = alt.Scale(domain=[vmin - pad, vmax + pad], nice=False, zero=False)
    else:
        y_scale = alt.Scale(nice=True, zero=False)

    return (
        alt.Chart(df, height=height)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "idx:Q",
                title=None,
                scale=alt.Scale(domain=[0, max(0, n - 1)], nice=False, zero=False),
                axis=alt.Axis(values=list(range(n)), labelExpr=label_expr),
            ),
            y=alt.Y("bankroll:Q", title="Bankroll ($)", scale=y_scale),
            tooltip=[alt.Tooltip("bankroll:Q", format="$.2f")],
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

    # This Week’s Bets (cards con score/estado)
    if not bets_week_raw.empty:
        try:
            view = _enrich_bets_with_espn(bets_week_raw, season=season)
        except Exception:
            view = bets_week_raw.copy()

        st.markdown("**This Week’s Bets**")

        if "week_label" in view.columns:
            view["week_label"] = view["week_label"].astype(str)
            view["__order"] = view["week_label"].apply(
                lambda s: int(s.split()[-1]) if s.startswith("Week ") and s.split()[-1].isdigit() else 999
            )
            sort_cols = ["__order"]
            if "schedule_date" in view.columns:
                view["schedule_date"] = pd.to_datetime(view["schedule_date"], errors="coerce")
                sort_cols.append("schedule_date")
            view = view.sort_values(sort_cols).drop(columns="__order", errors="ignore")

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

    # KPIs (Initial fijo en 1000)
    _, _, total_profit, total_stake, yield_pct, profits, stakes = kpis_from_pnl(pnl)
    initial_bankroll = 1000.0
    final_bankroll = (
        float(pnl["bankroll"].dropna().iloc[-1])
        if "bankroll" in pnl.columns and pnl["bankroll"].notna().any()
        else initial_bankroll
    )
    delta_bankroll = final_bankroll - initial_bankroll

    k1, k2 = st.columns(2)
    k1.metric("Initial", f"${initial_bankroll:,.2f}")
    k2.metric("Final", f"${final_bankroll:,.2f}", f"{delta_bankroll:,.2f}")

    # Charts
    H_BANK = 300
    H_PROF = 300

    bank_chart = _bankroll_chart_snug(pnl, height=H_BANK)

    cA, cB = st.columns(2)
    with cA:
        st.altair_chart(bank_chart, use_container_width=True)
    with cB:
        st.altair_chart(chart_last8_profit(pnl, profits, last=8, height=H_PROF), use_container_width=True)
