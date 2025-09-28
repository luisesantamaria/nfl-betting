# tabs/bets.py
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_ledger
from nfl_dash.utils import ORDER_INDEX, norm_abbr, week_label_from_num

# Rutas locales
LIVE_DIR = Path("data/live")
LIVE_BETS_PATH = LIVE_DIR / "bets.csv"
LIVE_ODDS_PATH = LIVE_DIR / "odds.csv"

# ------- util: carga odds con scores (archive -> live) -------
def _load_odds_scores(season: int) -> pd.DataFrame:
    """
    Lee odds con marcadores/estado.
    Prioridad: data/archive/season=YYYY/odds.csv -> data/live/odds.csv
    Devuelve columnas normalizadas:
      home_abbr, away_abbr, home_score, away_score, state ('pre'|'in'|'post'), start_time (opcional)
    """
    candidates = [
        Path(f"data/archive/season={season}/odds.csv"),
        LIVE_ODDS_PATH,
    ]
    df = pd.DataFrame()
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            break
    if df.empty:
        return df

    out = pd.DataFrame()

    # mapear nombres posibles a lo que necesitamos
    cols = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    # equipos
    h_team_col = col("home_abbr", "home_team", "home", "home_name")
    a_team_col = col("away_abbr", "away_team", "away", "away_name")
    if not h_team_col or not a_team_col:
        return pd.DataFrame()

    out["home_abbr"] = df[h_team_col].astype(str).map(norm_abbr)
    out["away_abbr"] = df[a_team_col].astype(str).map(norm_abbr)

    # marcadores
    h_sc_col = col("home_score", "score_home", "home_pts")
    a_sc_col = col("away_score", "score_away", "away_pts")
    if h_sc_col in df and a_sc_col in df:
        out["home_score"] = pd.to_numeric(df[h_sc_col], errors="coerce")
        out["away_score"] = pd.to_numeric(df[a_sc_col], errors="coerce")
    else:
        out["home_score"] = pd.NA
        out["away_score"] = pd.NA

    # estado
    st_col = col("state", "status", "game_state", "game_status")
    if st_col:
        s = df[st_col].astype(str).str.lower()
        # normalizamos a pre/in/post
        s = (
            s.replace({"final": "post", "finished": "post", "ended": "post",
                       "live": "in", "in progress": "in", "halftime": "in",
                       "pre": "pre", "scheduled": "pre"})
            .where(~s.isin(["post", "in", "pre"]), s)
        )
        out["state"] = s
    else:
        out["state"] = pd.NA

    # start_time opcional (para ordenar)
    dt_col = col("start_time", "kickoff", "schedule_date", "date")
    if dt_col:
        out["start_time"] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    else:
        out["start_time"] = pd.NaT

    # claves de emparejamiento
    out["_key_set"] = out.apply(lambda r: frozenset([r["home_abbr"], r["away_abbr"]]), axis=1)
    out["_key_order"] = out["home_abbr"] + "_" + out["away_abbr"]

    # index para búsqueda rápida
    set_idx = out.drop_duplicates("_key_set").set_index("_key_set")
    order_idx = out.drop_duplicates("_key_order").set_index("_key_order")
    # guardamos en attrs para reuso (para no recrearlos en cada llamada)
    out.attrs["set_idx"] = set_idx
    out.attrs["order_idx"] = order_idx
    return out

# ------- util: carga bets live cuando no hay archive -------
def _load_current_season_bets_from_live(season: int) -> pd.DataFrame:
    if not LIVE_BETS_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(LIVE_BETS_PATH, low_memory=False)

    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        df = df[df["season"] == season].copy()
    if df.empty:
        return df

    for col in ("decimal_odds", "ml", "stake", "profit"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for c in ("team", "opponent"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)

    # week label
    if "week_label" not in df.columns:
        if "week" in df.columns:
            df["week"] = pd.to_numeric(df["week"], errors="coerce")
            df["week_label"] = df["week"].apply(week_label_from_num).astype(str)
        else:
            df["week_label"] = "Week 999"

    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")

    return df

# ------- util: enriquecer bets con odds (scores + estado) -------
def _enrich_with_odds_scores(bets: pd.DataFrame, season: int) -> pd.DataFrame:
    if bets.empty:
        return bets

    odds = _load_odds_scores(season)
    if odds.empty:
        return bets

    set_idx = odds.attrs["set_idx"]
    order_idx = odds.attrs["order_idx"]

    v = bets.copy()

    # normalizar team/opponent/side
    for c in ("team", "opponent", "home_team", "away_team"):
        if c in v.columns:
            v[c] = v[c].astype(str).map(norm_abbr)

    if "side" in v.columns:
        v["side"] = v["side"].astype(str).str.lower()

    # inferir home/away si es posible
    def _pair_home(r):
        side = r.get("side", "")
        t = r.get("team", "")
        o = r.get("opponent", "")
        if side == "home":
            return t, o
        if side == "away":
            return o, t
        # si ya existen columnas home_team/away_team, úsalas
        ht = r.get("home_team", "")
        at = r.get("away_team", "")
        if ht and at:
            return ht, at
        # sin orden, devolvemos None (usaremos clave set)
        return None, None

    # columnas a rellenar
    for col in ("home_team", "away_team", "score_home", "score_away", "state"):
        if col not in v.columns:
            v[col] = pd.NA

    # match fila por fila
    for i, r in v.iterrows():
        home, away = _pair_home(r)
        # match por orden si lo sabemos
        if home and away:
            k = f"{home}_{away}"
            if k in order_idx.index:
                row = order_idx.loc[k]
                v.at[i, "home_team"] = row["home_abbr"]
                v.at[i, "away_team"] = row["away_abbr"]
                v.at[i, "score_home"] = row.get("home_score", pd.NA)
                v.at[i, "score_away"] = row.get("away_score", pd.NA)
                v.at[i, "state"] = row.get("state", pd.NA)
                continue

        # si no sabemos el orden o no hubo match, probamos por set
        t = r.get("team", "")
        o = r.get("opponent", "")
        if t and o:
            kset = frozenset([t, o])
            if kset in set_idx.index:
                row = set_idx.loc[kset]
                v.at[i, "home_team"] = row["home_abbr"]
                v.at[i, "away_team"] = row["away_abbr"]
                v.at[i, "score_home"] = row.get("home_score", pd.NA)
                v.at[i, "score_away"] = row.get("away_score", pd.NA)
                v.at[i, "state"] = row.get("state", pd.NA)

    return v

# ---------------- render principal ----------------
def render(season: int):
    st.subheader("Bets")

    # 1) Archivo de bets archivado
    bets_all_raw = load_ledger(season)

    # 2) Fallback live si no hay archive
    if bets_all_raw.empty:
        bets_all_raw = _load_current_season_bets_from_live(season)

    if bets_all_raw.empty:
        st.caption("No bets file for this season.")
        return

    # 3) Normalizaciones mínimas
    view = bets_all_raw.copy()

    if "week_label" not in view.columns:
        if "week" in view.columns:
            view["week"] = pd.to_numeric(view["week"], errors="coerce")
            view["week_label"] = view["week"].apply(week_label_from_num).astype(str)
        else:
            view["week_label"] = "Week 999"

    if "schedule_date" in view.columns:
        view["schedule_date"] = pd.to_datetime(view["schedule_date"], errors="coerce")

    for c in ("team", "opponent"):
        if c in view.columns:
            view[c] = view[c].astype(str).map(norm_abbr)

    # 4) Enriquecer SIEMPRE con odds.csv (scores + estado)
    view = _enrich_with_odds_scores(view, season)

    # 5) Orden por semana y dentro por kickoff si hay
    view["week_label"] = view["week_label"].astype(str)
    view["__order"] = view["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    sort_cols = ["__order"]
    if "schedule_date" in view.columns:
        sort_cols.append("schedule_date")
    view = view.sort_values(sort_cols, kind="stable").drop(columns="__order")

    # 6) Render cards agrupadas por semana
    from nfl_dash.components import bet_card as render_bet_card

    weeks_sorted = view["week_label"].dropna().astype(str).unique().tolist()
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

        st.divider()
