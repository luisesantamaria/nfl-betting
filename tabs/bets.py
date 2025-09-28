# tabs/bets.py
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_ledger
from nfl_dash.utils import ORDER_INDEX, norm_abbr, week_label_from_num


LIVE_BETS_PATH = Path("data/live/bets.csv")


def _load_current_season_bets_from_live(season: int) -> pd.DataFrame:
    """Fallback: lee data/live/bets.csv y filtra por temporada."""
    if not LIVE_BETS_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(LIVE_BETS_PATH, low_memory=False)

    # Filtro por season
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        df = df[df["season"] == season].copy()

    if df.empty:
        return df

    # Normalizaciones
    for col in ("decimal_odds", "ml", "stake", "profit"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for c in ("team", "opponent"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)

    # week_label
    if "week_label" not in df.columns:
        if "week" in df.columns:
            df["week"] = pd.to_numeric(df["week"], errors="coerce")
            df["week_label"] = df["week"].apply(week_label_from_num).astype(str)
        else:
            df["week_label"] = "Week 999"

    # Derivar home/away + scores para que bet_card pinte escudos y marcadores
    # Si side está vacío no alteramos (bet_card usa team/opponent como respaldo)
    side = df.get("side")
    if side is not None:
        df["side"] = df["side"].astype(str).str.lower()
        has_scores = "team_score" in df.columns and "opponent_score" in df.columns

        def _home_team(r):
            if r["side"] == "home":
                return r.get("team")
            elif r["side"] == "away":
                return r.get("opponent")
            return r.get("home_team", pd.NA)

        def _away_team(r):
            if r["side"] == "home":
                return r.get("opponent")
            elif r["side"] == "away":
                return r.get("team")
            return r.get("away_team", pd.NA)

        df["home_team"] = df.apply(_home_team, axis=1).astype(str).map(norm_abbr)
        df["away_team"] = df.apply(_away_team, axis=1).astype(str).map(norm_abbr)

        if has_scores:
            def _score_home(r):
                if r["side"] == "home":
                    return r.get("team_score")
                elif r["side"] == "away":
                    return r.get("opponent_score")
                return r.get("score_home", pd.NA)

            def _score_away(r):
                if r["side"] == "home":
                    return r.get("opponent_score")
                elif r["side"] == "away":
                    return r.get("team_score")
                return r.get("score_away", pd.NA)

            df["score_home"] = df.apply(_score_home, axis=1)
            df["score_away"] = df.apply(_score_away, axis=1)

    # Orden suave por kickoff si existe
    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
        df = df.sort_values("schedule_date")

    return df


def render(season: int):
    st.subheader("Bets")

    # 1) Archivo de temporadas archivadas
    bets_all_raw = load_ledger(season)

    # 2) Fallback a live (temporada actual)
    if bets_all_raw.empty:
        bets_all_raw = _load_current_season_bets_from_live(season)

    # 3) Si sigue vacío, no hay nada
    if bets_all_raw.empty:
        st.caption("No bets file for this season.")
        return

    # 4) Si tienes un enriquecedor para temporadas pasadas, puedes seguir usándolo.
    #    Para live ya trajimos scores/teams arriba, así que no es obligatorio.
    try:
        from nfl_dash.odds_scores import enrich_bets_with_scores
        # Sólo enriquece si faltan columnas de score_home/score_away (típico en archive).
        need_enrich = not {"score_home", "score_away"}.issubset(set(bets_all_raw.columns))
        bets_all = enrich_bets_with_scores(bets_all_raw, season) if need_enrich else bets_all_raw
    except Exception:
        # Si no existe el módulo o falla, usa lo que tenemos
        bets_all = bets_all_raw

    view = bets_all.copy()

    # Week label + orden
    if "week_label" not in view.columns:
        if "week" in view.columns:
            view["week_label"] = view["week"].apply(week_label_from_num).astype(str)
        else:
            view["week_label"] = "Week 999"

    view["week_label"] = view["week_label"].astype(str)
    view["__order"] = view["week_label"].map(ORDER_INDEX).fillna(999).astype(int)

    # Orden global por semana y dentro de semana por kickoff si existe
    sort_cols = ["__order"]
    if "schedule_date" in view.columns:
        sort_cols.append("schedule_date")
    view = view.sort_values(sort_cols, kind="stable").drop(columns="__order")

    # Import perezoso del renderer
    from nfl_dash.components import bet_card as render_bet_card

    # 5) Agrupar por semana y pintar secciones
    weeks_sorted = (
        view["week_label"].dropna().astype(str).unique().tolist()
    )
    weeks_sorted.sort(key=lambda s: ORDER_INDEX.get(s, 999))

    for wk_label in weeks_sorted:
        st.markdown(f"### {wk_label}")

        g = view[view["week_label"].astype(str).eq(wk_label)]
        cards = list(g.itertuples(index=False))

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

        # Separador entre semanas
        st.divider()
