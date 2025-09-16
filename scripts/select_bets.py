#!/usr/bin/env python3
# scripts/select_bets.py
import os, io, gzip, time, re, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# ML
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# =========================
# Paths / Config
# =========================
LIVE_ODDS_PATH   = os.path.join("data", "live", "odds.csv")       # odds actuales (acumula semanas)
LIVE_STATS_PATH  = os.path.join("data", "live", "stats.csv")      # pregame YTD actual
ARCHIVE_DIR      = os.path.join("data", "archive")                # stats/odds históricos por temporada
BETS_OUT_PATH    = os.path.join("data", "live", "bets.csv")

# Columnas solicitadas
BETS_COLS = [
    "season","week","week_label","schedule_date","side","team","opponent",
    "decimal_odds","ml","market_prob_nv","model_prob","ev","edge","won",
    "week_order","game_id","stake","profit","bankroll"
]

# Equipo: normalizaciones
TEAM_FIX = {
    "STL":"LA","LAR":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV","WSH":"WAS","JAC":"JAX",
    "GNB":"GB","KAN":"KC","NWE":"NE","NOR":"NO","SFO":"SF","TAM":"TB"
}

def norm_team(x: str) -> str:
    s = str(x).upper().strip()
    return TEAM_FIX.get(s, s)

ORDER_LABELS = [f"Week {i}" for i in range(1,19)] + ["Wild Card","Divisional","Conference","Super Bowl"]
ORDER_INDEX  = {lab:i for i,lab in enumerate(ORDER_LABELS)}

def week_label_from_num(n:int) -> str:
    return f"Week {n}" if 1<=n<=18 else {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")

def add_week_order(df):
    out = df.copy()
    out["week_label"] = out["week_label"].astype(str)
    out["week_order"] = out["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    return out

def american_to_decimal(m):
    if m is None or (isinstance(m,float) and np.isnan(m)): return np.nan
    try:
        m = float(m)
    except Exception:
        return np.nan
    return 1 + (100/abs(m) if m<0 else m/100)

def decimal_to_american(d):
    try:
        d = float(d)
    except Exception:
        return np.nan
    if d <= 1.0 or np.isnan(d): return np.nan
    return round((d - 1.0) * 100.0, 0) if d >= 2.0 else round(-100.0 / (d - 1.0), 0)

# =========================
# Carga de datos (stats + odds)
# =========================
def detect_target_season() -> int:
    # Detecta temporada a partir de odds.csv (preferido) o stats.csv
    for p in (LIVE_ODDS_PATH, LIVE_STATS_PATH):
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, nrows=100)
                s = pd.to_numeric(df.get("season"), errors="coerce").dropna()
                if not s.empty:
                    return int(s.mode().iloc[0])
            except Exception:
                pass
    # fallback: año actual (ET)
    return datetime.now().year

def load_pregame_all(target_season: int) -> pd.DataFrame:
    """
    Concatena stats pregame:
      - últimos 4 años previos desde archive si existen (season={YYYY}/stats.csv)
      - temporada actual desde data/live/stats.csv (si existe)
    """
    frames = []
    for y in range(target_season-4, target_season+1):
        # live sólo para temporada actual
        if y == target_season and os.path.exists(LIVE_STATS_PATH):
            try:
                df = pd.read_csv(LIVE_STATS_PATH, low_memory=False)
                df["season"] = int(y)
                frames.append(df)
            except Exception:
                pass
        # archive para historicos
        p = os.path.join(ARCHIVE_DIR, f"season={y}", "stats.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, low_memory=False)
                df["season"] = int(y)
                frames.append(df)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    pre = pd.concat(frames, ignore_index=True)
    # Normalizaciones mínimas
    if "team" in pre.columns:
        pre["team"] = pre["team"].map(norm_team)
    if "week" in pre.columns:
        pre["week"] = pd.to_numeric(pre["week"], errors="coerce").astype("Int64")
    if "game_date" in pre.columns:
        pre["game_date"] = pd.to_datetime(pre["game_date"], errors="coerce")
    # Dejar 1 fila por (season, week, team) con la última (YTD ya viene)
    keep = ["season","week","team"]
    pre = (pre.sort_values(keep + (["game_date"] if "game_date" in pre.columns else []))
             .drop_duplicates(keep, keep="last"))
    return pre

def load_odds_season_current(target_season: int) -> pd.DataFrame:
    """
    Toma odds acumuladas de la temporada actual (data/live/odds.csv).
    Este archivo ya trae spreads, totals, moneylines (si hay), y al cerrar juegos, score_home/score_away/home_win.
    """
    if not os.path.exists(LIVE_ODDS_PATH):
        return pd.DataFrame()
    df = pd.read_csv(LIVE_ODDS_PATH, low_memory=False)
    if df.empty:
        return df
    # tipos
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce").astype("Int64")
    df["week"]   = pd.to_numeric(df.get("week"), errors="coerce").astype("Int64")
    df["home_team"] = df.get("home_team","").astype(str).map(norm_team)
    df["away_team"] = df.get("away_team","").astype(str).map(norm_team)
    # Semana/label/date
    if "week_label" not in df.columns and "week" in df.columns:
        df["week_label"] = df["week"].apply(lambda w: week_label_from_num(int(w)) if pd.notna(w) else None)
    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce", utc=True)
    # Home line desde spread_home si falta
    if "home_line" not in df.columns:
        if "spread_home" in df.columns:
            df["home_line"] = pd.to_numeric(df["spread_home"], errors="coerce")
        else:
            df["home_line"] = np.nan
    # Moneylines y probs de mercado (si existen) o estimadas a partir de decimales
    for side in ("home","away"):
        if f"decimal_{side}" not in df.columns and f"ml_{side}" in df.columns:
            df[f"decimal_{side}"] = df[f"ml_{side}"].apply(american_to_decimal)
    if "decimal_home" in df.columns and "decimal_away" in df.columns:
        ph = 1.0 / pd.to_numeric(df["decimal_home"], errors="coerce")
        pa = 1.0 / pd.to_numeric(df["decimal_away"], errors="coerce")
        s = ph + pa
        df["market_prob_home_nv"] = (ph/s).clip(1e-6, 1-1e-6)
        df["market_prob_away_nv"] = (pa/s).clip(1e-6, 1-1e-6)
    return df

def build_master_from_repo(odds_df: pd.DataFrame, pre_df: pd.DataFrame) -> pd.DataFrame:
    """
    Emula tu 'build_master_from_kaggle': normaliza teams, asegura home_line y une pregame (home_/away_ prefijo).
    Sólo usamos columnas existentes; no inventamos nada.
    """
    if odds_df.empty or pre_df.empty:
        return pd.DataFrame()

    out = odds_df.copy()
    for c in ("home_team","away_team","team_favorite_id"):
        if c in out.columns:
            out[c] = out[c].map(norm_team)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype(int)
    out["week"]   = pd.to_numeric(out["week"],   errors="coerce").astype(int)
    if "week_label" not in out.columns:
        out["week_label"] = out["week"].apply(week_label_from_num)

    # home_line si falta
    if "home_line" not in out.columns or out["home_line"].isna().any():
        def compute_home_line(r):
            s  = r.get("spread_favorite", np.nan)
            tf = str(r.get("team_favorite_id", ""))
            if pd.isna(s) or tf == "" or tf == "nan":
                return r.get("home_line", np.nan)
            if tf == r["home_team"]:
                return float(s)
            if tf == r["away_team"]:
                return float(-s)
            return r.get("home_line", np.nan)
        out["home_line"] = out.apply(compute_home_line, axis=1)

    pre = pre_df.copy()
    pre["team"]   = pre["team"].map(norm_team)
    pre["season"] = pd.to_numeric(pre["season"], errors="coerce").astype(int)
    pre["week"]   = pd.to_numeric(pre["week"],   errors="coerce").astype(int)
    pre = (pre.sort_values(["season","week","team"])
             .drop_duplicates(["season","week","team"], keep="last"))

    def add_pref(df, pref, key):
        dd = df.copy()
        dd = dd.add_prefix(pref)
        dd = dd.rename(columns={pref+"team": key, pref+"season":"season", pref+"week":"week"})
        return dd

    home_pre = add_pref(pre.rename(columns={"team":"home_team"}), "home_", key="home_team")
    away_pre = add_pref(pre.rename(columns={"team":"away_team"}), "away_", key="away_team")

    m = (out.merge(home_pre, on=["season","week","home_team"], how="left")
            .merge(away_pre, on=["season","week","away_team"], how="left"))

    if "schedule_date" in m.columns:
        m["schedule_date"] = pd.to_datetime(m["schedule_date"], errors="coerce", utc=True)
    m = m.sort_values(["schedule_date","season","week","home_team"]).reset_index(drop=True)
    return m

# =========================
# Modelo (idéntico a tu notebook)
# =========================
def train_and_predict(master_hist: pd.DataFrame, master_cur: pd.DataFrame, target_season: int):
    assert isinstance(master_cur, pd.DataFrame)
    if master_hist is None or master_hist.empty:
        # Sin históricos suficientes con odds → entrenamos con lo disponible anterior a target (si existe)
        df = master_cur.copy()
    else:
        df = pd.concat([master_hist.copy(), master_cur.copy()], ignore_index=True)

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"]   = pd.to_numeric(df["week"],   errors="coerce").astype("Int64")

    # home_line + features de mercado
    def compute_home_line_row(r):
        s  = r.get("spread_favorite", np.nan)
        tf = str(r.get("team_favorite_id", ""))
        if pd.isna(s) or tf == "" or tf == "nan":
            return r.get("home_line", np.nan)
        if tf == r["home_team"]:
            return float(s)
        if tf == r["away_team"]:
            return float(-s)
        return r.get("home_line", np.nan)

    if "home_line" not in df.columns or df["home_line"].isna().any():
        df["home_line"] = df.apply(compute_home_line_row, axis=1)

    df["abs_spread"] = pd.to_numeric(df["home_line"], errors="coerce").astype(float).abs()
    df["fav_home"]   = (pd.to_numeric(df["home_line"], errors="coerce") < 0).astype(int)
    df["ou"]         = pd.to_numeric(df.get("over_under_line", np.nan), errors="coerce").astype(float)

    df["spread_x_ou"]   = df["abs_spread"] * df["ou"]
    df["fav_x_spread"]  = df["fav_home"]   * df["abs_spread"]
    df["home_line_sq"]  = df["home_line"]  * df["home_line"]

    # columnas pregame (todas *_pre*)
    pre_cols = [c for c in df.columns if re.search(r'(?:_pre(_ewm)?|_pre_l8)$', c)]
    mkt_cols = [c for c in ["home_line","abs_spread","fav_home","ou","spread_x_ou","fav_x_spread","home_line_sq"] if c in df.columns]
    feat_cols = [c for c in (pre_cols + mkt_cols) if c in df.columns]

    hist_years = sorted(df.loc[df["season"] < target_season, "season"].dropna().unique().tolist())
    if len(hist_years) < 2:
        # si no hay 2 años previos completos, usa todo < target como train y target como test, sin validación separada
        train_years = hist_years[:-1] if len(hist_years) > 0 else []
        val_year    = hist_years[-1] if len(hist_years) > 0 else target_season
    else:
        val_year    = hist_years[-1]
        train_years = hist_years[:-1]
    test_year = target_season

    base = df.dropna(subset=["home_line"]).copy()
    train_df = base[base["season"].isin(train_years)].copy()
    val_df   = base[base["season"].eq(val_year)].copy()
    test_df  = base[base["season"].eq(test_year)].copy()

    # Requeridos por tu notebook
    y_train = train_df["home_win"].astype(int).values if "home_win" in train_df.columns else np.zeros(len(train_df), dtype=int)
    y_val   = val_df["home_win"].astype(int).values   if "home_win" in val_df.columns else np.zeros(len(val_df), dtype=int)

    X_train = train_df[feat_cols].copy()
    X_val   = val_df[feat_cols].copy()
    X_test  = test_df[feat_cols].copy()

    train_meds = X_train.median(numeric_only=True) if len(X_train) else pd.Series(dtype=float)
    X_train = X_train.fillna(train_meds)
    X_val   = X_val.fillna(train_meds)
    X_test  = X_test.fillna(train_meds)

    monotonic_cst = [(-1 if c == "home_line" else 0) for c in feat_cols]

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_leaf_nodes=21,
        min_samples_leaf=60,
        l2_regularization=2.0,
        max_iter=320,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        monotonic_cst=monotonic_cst,
        random_state=42
    )
    if len(X_train):
        hgb.fit(X_train, y_train)
        p_tr = hgb.predict_proba(X_train)[:,1] if len(X_train) else np.array([])
        p_va = hgb.predict_proba(X_val)[:,1]   if len(X_val)   else np.array([])
        p_te = hgb.predict_proba(X_test)[:,1]  if len(X_test)  else np.array([])
    else:
        # sin train → usa prior de spread luego
        p_tr = np.array([]); p_va = np.array([]); p_te = np.array([])

    # Binning por spread
    def cut_bins(x):
        bin_edges = [-21, -10.5, -6.5, -3.5, -0.5, 0.5, 3.5, 6.5, 10.5, 21]
        return pd.cut(x, bins=bin_edges, include_lowest=True)

    val_bins  = cut_bins(val_df["home_line"]) if len(val_df) else pd.Series(dtype="category")
    test_bins = cut_bins(test_df["home_line"]) if len(test_df) else pd.Series(dtype="category")

    # Isotonic global + por bin (si hay datos)
    iso_global = IsotonicRegression(out_of_bounds="clip")
    if len(p_va):
        iso_global.fit(p_va, y_val)
    else:
        # fallback trivial si no hay validación
        iso_global.fit(np.array([0.25,0.75]), np.array([0,1]))

    bin_min = 120
    iso_by_bin = {}
    if len(p_va):
        for b in val_bins.unique():
            idx = (val_bins == b)
            if idx.sum() >= bin_min:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_va[idx], y_val[idx])
                iso_by_bin[b] = iso

    def apply_bin_calibration(p_raw, bins_series):
        if len(p_raw) == 0:
            return p_raw
        out = np.empty_like(p_raw, dtype=float)
        if not iso_by_bin:
            return np.clip(iso_global.transform(p_raw), 1e-6, 1-1e-6)
        for b in bins_series.unique():
            m = (bins_series == b).values
            if b in iso_by_bin:
                out[m] = iso_by_bin[b].transform(p_raw[m])
            else:
                out[m] = iso_global.transform(p_raw[m])
        return np.clip(out, 1e-6, 1-1e-6)

    p_tr_cal = iso_global.transform(p_tr) if len(p_tr) else p_tr
    p_va_cal = apply_bin_calibration(p_va, val_bins) if len(p_va) else p_va
    p_te_cal = apply_bin_calibration(p_te, test_bins) if len(p_te) else p_te

    # Prior por spread (LR)
    prior_lr = LogisticRegression(C=2.0, solver="liblinear", max_iter=200)
    if len(train_df):
        prior_lr.fit(train_df[["home_line"]], y_train)
        p_tr_sp = prior_lr.predict_proba(train_df[["home_line"]])[:,1] if len(train_df) else np.array([])
        p_va_sp = prior_lr.predict_proba(val_df[["home_line"]])[:,1]   if len(val_df)   else np.array([])
        p_te_sp = prior_lr.predict_proba(test_df[["home_line"]])[:,1]  if len(test_df)  else np.array([])
    else:
        # prior de spread sin entrenamiento → usa sigmoide aproximada vía coef fijo (fallback simple)
        def p_from_spread(s):
            try:
                s = float(s)
            except Exception:
                return 0.5
            # heurística suave (NO altera tu pipeline; sólo evita crashear si no hay historial)
            return 1.0 / (1.0 + np.exp(0.28 * s))
        p_tr_sp = np.array([])
        p_va_sp = np.array([])
        p_te_sp = test_df["home_line"].apply(p_from_spread).values if len(test_df) else np.array([])

    def meta_matrix(df_in, p_hgb_cal, p_spread):
        M = pd.DataFrame({
            "p_hgb":   p_hgb_cal if len(p_hgb_cal) else np.zeros(len(df_in)),
            "p_sp":    p_spread  if len(p_spread)  else np.zeros(len(df_in)),
            "home_line": df_in["home_line"].values,
            "abs_spread": df_in["abs_spread"].values,
            "ou": df_in["ou"].values
        })
        return M.fillna(M.median(numeric_only=True))

    meta_tr   = meta_matrix(train_df, p_tr_cal, p_tr_sp) if len(train_df) else pd.DataFrame()
    meta_val  = meta_matrix(val_df,   p_va_cal, p_va_sp) if len(val_df)   else pd.DataFrame()
    meta_test = meta_matrix(test_df,  p_te_cal, p_te_sp) if len(test_df)  else pd.DataFrame()

    meta_lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=300)
    if len(meta_val):
        meta_lr.fit(meta_val, y_val)
        p_te_meta = np.clip(meta_lr.predict_proba(meta_test)[:,1], 1e-6, 1-1e-6) if len(meta_test) else np.array([])
    else:
        # Si no hay val, usa prior de spread calibrado globalmente
        p_te_meta = np.clip(p_te_sp if len(p_te_sp) else p_te_cal, 1e-6, 1-1e-6) if len(meta_test) else np.array([])

    # Salida: probabilidades para temporada target
    test_preds = test_df[["season","week","week_label","home_team","away_team"]].copy()
    if "home_win" in test_df.columns:
        test_preds["home_win"] = test_df["home_win"].values
    # Siguiendo nombres de tu notebook:
    test_preds["p_home_win_lr_cal"] = p_te_cal if len(p_te_cal) else (p_te_sp if len(p_te_sp) else np.array([]))
    test_preds["p_home_win_meta"]   = p_te_meta if len(p_te_meta) else (p_te_sp if len(p_te_sp) else np.array([]))
    return test_preds

# =========================
# EV + estrategia + staking (idéntico a tu notebook)
# =========================
CFG = dict(
    INITIAL_BANKROLL = 1000.0,

    KELLY_FRACTION   = 0.25,
    KELLY_PCT_CAP    = 0.05,
    ABS_STAKE_CAP    = 300.0,
    WEEKLY_CAP_PCT   = 0.50,

    ODDS_MIN         = 1.20,
    ODDS_MAX         = 3.40,
    CONF_MIN         = 0.070,
    EDGE_TAU         = 0.045,
    EDGE_TAU_MIN     = 0.040,
    EV_BASE_MIN      = 0.010,
    EV_SLOPE         = 0.011,
    SKIP_W1_W2       = True,

    DEVIG_SINGLE_SIDE = "raw",

    KELLY_EDGE_WEIGHT = dict(EDGE_REF=0.12, MIN_SCALE=0.60, MAX_SCALE=1.00),
    MAX_BETS_PER_WEEK = 9,
    MAX_BIG_DOGS_PER_WEEK = 1,

    BANDS = dict(
        FAV = dict(odds_lt=1.60,                 tau=0.040, conf=0.065, ev_slope=0.008),
        MID = dict(odds_ge=1.60, odds_lt=2.40,   tau=0.045, conf=0.070, ev_slope=0.010),
        DOG = dict(odds_ge=2.40, odds_le=3.40,   tau=0.055, conf=0.085, ev_slope=0.015,
                   extra_if_ge3_10=dict(tau=0.060, conf=0.090))
    ),
)
MIN_PROB = 1e-6

def make_pair(a, b):
    a = str(a); b = str(b)
    return a + "_" + b if a < b else b + "_" + a

def make_game_id(df):
    t1 = df["team"].astype(str); opp = df["opponent"].astype(str)
    return df["week_label"].astype(str) + " | " + np.where(t1 < opp, t1+"_"+opp, opp+"_"+t1)

def implied_from_decimal(d):
    d = float(d); return np.clip(1.0/max(d,1e-9), MIN_PROB, 1-MIN_PROB)

def devig_per_game(ev_df: pd.DataFrame, single_side_mode: str) -> pd.DataFrame:
    ev = ev_df.copy()
    if "market_prob_nv" not in ev.columns:
        ev["market_prob_nv"] = ev["decimal_odds"].apply(implied_from_decimal)

    tmp = ev.copy()
    tmp["p_raw"] = 1.0 / tmp["decimal_odds"].clip(1e-9)
    sides_per_game = tmp.groupby("game_id")["side"].transform("nunique")
    sum_p = tmp.groupby("game_id")["p_raw"].transform("sum")

    mask_pairs = sides_per_game >= 2
    ev.loc[mask_pairs, "market_prob_nv"] = (tmp.loc[mask_pairs, "p_raw"] / sum_p[mask_pairs]).clip(1e-6, 1-1e-6)

    if single_side_mode == "raw":
        pass
    else:
        pass

    ev["market_prob_nv"] = ev["market_prob_nv"].clip(1e-6, 1-1e-6)
    return ev

def sanitize_ev(df: pd.DataFrame, cfg) -> pd.DataFrame:
    need = ["season","week","week_label","schedule_date","side","team","opponent","decimal_odds","model_prob"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"EV table missing columns: {missing}")

    ev = df.copy()
    ev["schedule_date"] = pd.to_datetime(ev["schedule_date"], errors="coerce", utc=True)
    ev["model_prob"] = ev["model_prob"].astype(float).clip(MIN_PROB, 1-MIN_PROB)
    ev["decimal_odds"] = ev["decimal_odds"].astype(float)

    ev = add_week_order(ev)
    ev["game_id"] = make_game_id(ev)

    ev = ev[ev["decimal_odds"].between(cfg["ODDS_MIN"], cfg["ODDS_MAX"])]
    ev = ev[np.abs(ev["model_prob"] - 0.5) >= cfg["CONF_MIN"]]
    if cfg["SKIP_W1_W2"]:
        ev = ev[~ev["week_label"].isin(["Week 1","Week 2"])]

    return ev.sort_values(["week_order","schedule_date","game_id"]).reset_index(drop=True)

def ev_floor(decimal_odds, ev_base_min, ev_slope):
    d = float(decimal_odds)
    return ev_base_min + max(0.0, d - 2.0) * ev_slope

def ev_slope_for_row(row, cfg):
    d = float(row["decimal_odds"])
    b = cfg["BANDS"]
    if d < b["FAV"]["odds_lt"]:
        return b["FAV"]["ev_slope"]
    if d < b["MID"]["odds_lt"]:
        return b["MID"]["ev_slope"]
    return b["DOG"]["ev_slope"]

def apply_band_rules(df: pd.DataFrame, cfg) -> pd.DataFrame:
    b = cfg.get("BANDS", {})
    if not b:
        return df
    x = df.copy()
    d = x["decimal_odds"].astype(float)
    conf = np.abs(x["model_prob"] - 0.5)
    edge = x["edge"].astype(float)
    keep = pd.Series(True, index=x.index)

    if "FAV" in b:
        m = d < b["FAV"]["odds_lt"]
        keep &= (~m) | ((edge >= b["FAV"]["tau"]) & (conf >= b["FAV"]["conf"]))
    if "MID" in b:
        m = (d >= b["MID"]["odds_ge"]) & (d < b["MID"]["odds_lt"])
        keep &= (~m) | ((edge >= b["MID"]["tau"]) & (conf >= b["MID"]["conf"]))
    if "DOG" in b:
        m = (d >= b["DOG"]["odds_ge"]) & (d <= b["DOG"]["odds_le"])
        keep &= (~m) | ((edge >= b["DOG"]["tau"]) & (conf >= b["DOG"]["conf"]))
        if "extra_if_ge3_10" in b["DOG"]:
            m2 = d >= 3.10
            keep &= (~m2) | ((edge >= b["DOG"]["extra_if_ge3_10"]["tau"])
                             & (conf >= b["DOG"]["extra_if_ge3_10"]["conf"]))

    return x[keep].reset_index(drop=True)

def pick_per_game_best(ev: pd.DataFrame, cfg) -> pd.DataFrame:
    tau = max(cfg["EDGE_TAU"], cfg["EDGE_TAU_MIN"])

    def ok_ev(row):
        slope = ev_slope_for_row(row, cfg)
        return row["ev"] >= ev_floor(row["decimal_odds"], cfg["EV_BASE_MIN"], slope)

    c = ev[ev["edge"] >= tau].copy()
    if c.empty: return c
    c = c[c.apply(ok_ev, axis=1)]

    c = (c.sort_values(["week_order","schedule_date","game_id","edge"],
                       ascending=[True,True,True,False])
           .drop_duplicates("game_id"))

    c = apply_band_rules(c, cfg)

    if cfg.get("MAX_BIG_DOGS_PER_WEEK"):
        out = []
        for wk, g in c.groupby("week_label", sort=False):
            m_big = g["decimal_odds"] >= 3.10
            if m_big.sum() > cfg["MAX_BETS_PER_WEEK"]:
                g_big = g[m_big].sort_values("edge", ascending=False).head(cfg["MAX_BIG_DOGS_PER_WEEK"])
                g_rest = g[~m_big]
                g = pd.concat([g_rest, g_big], ignore_index=True)
            out.append(g)
        c = pd.concat(out, ignore_index=True)

    if cfg.get("MAX_BETS_PER_WEEK"):
        out = []
        for wk, g in c.groupby("week_label", sort=False):
            g = g.sort_values(["ev","edge"], ascending=[False,False]).head(cfg["MAX_BETS_PER_WEEK"])
            out.append(g)
        c = pd.concat(out, ignore_index=True)

    return c.reset_index(drop=True)

def kelly_fraction_scaled(edge, cfg):
    wd = cfg.get("KELLY_EDGE_WEIGHT")
    if not wd:
        return cfg["KELLY_FRACTION"]
    scale = np.clip(edge / wd["EDGE_REF"], wd["MIN_SCALE"], wd["MAX_SCALE"])
    return cfg["KELLY_FRACTION"] * float(scale)

def kelly_stake(bk_week0, p, d, edge, cfg):
    b = max(d - 1.0, 1e-9)
    f_star = (b*p - (1-p)) / b
    f_base = max(0.0, f_star)
    f_adj  = kelly_fraction_scaled(edge, cfg)
    stake = bk_week0 * f_base * f_adj
    stake = min(stake, bk_week0*cfg["KELLY_PCT_CAP"], cfg["ABS_STAKE_CAP"])
    return float(np.floor(stake*100)/100.0)

def stake_week_with_cap(group_df, bk_week0, cfg):
    raw = []
    for _, r in group_df.iterrows():
        st = kelly_stake(bk_week0, float(r["model_prob"]), float(r["decimal_odds"]), float(r["edge"]), cfg)
        raw.append(st)
    cap = bk_week0 * cfg["WEEKLY_CAP_PCT"]
    total = sum(raw)
    scale = min(1.0, cap / total) if total > 0 else 1.0
    return [float(np.floor(s*scale*100)/100.0) for s in raw]

def simulate_weekly_kelly(bets_in: pd.DataFrame, cfg):
    data = bets_in.copy()
    if "won" not in data.columns:
        data["won"] = np.nan
    data = data.sort_values(["week_order","schedule_date","game_id"]).reset_index(drop=True)

    # Sólo computamos stake/profit/bankroll cuando 'won' está definido; si no, stake=0 hasta que haya resultado.
    # Pero para mostrar apuestas de esta semana, calculamos stake con el bankroll al inicio de cada semana.
    # Para medir bankroll progresivo usamos sólo semanas con resultado disponible.
    bk = cfg["INITIAL_BANKROLL"]
    stakes, profits, bk_view = [], [], []
    prev_week = None
    wk_open_bankroll = bk

    for _, r in data.iterrows():
        wk = r["week_label"]
        if wk != prev_week:
            wk_open_bankroll = bk
            prev_week = wk

        # stake propuesto (aunque 'won' sea NaN, lo guardamos para UI)
        st = kelly_stake(wk_open_bankroll, float(r["model_prob"]), float(r["decimal_odds"]), float(r["edge"]), cfg)
        stakes.append(st)

        if pd.isna(r["won"]):
            prof = 0.0
            bk_after = bk  # sin cerrar
        else:
            prof = (float(r["decimal_odds"])-1.0)*st if int(r["won"])==1 else -st
            bk_after = float(np.round(bk + prof, 2))
            bk = bk_after  # avanza bankroll al cerrar apuesta
        profits.append(float(np.round(prof, 2)))
        bk_view.append(float(np.round(bk_after, 2)))

    out = data.copy()
    out["stake"] = np.round(stakes, 2)
    out["profit"] = np.round(profits, 2)
    out["bankroll"] = np.round(bk_view, 2)
    return out

def backfill_won_if_missing(ev_df: pd.DataFrame) -> pd.DataFrame:
    ev = ev_df.copy()
    if "won" in ev.columns:
        return ev
    if not os.path.exists(LIVE_ODDS_PATH):
        return ev

    tgt = pd.read_csv(LIVE_ODDS_PATH, low_memory=False)
    need = ["season","week","home_team","away_team"]
    miss = [c for c in need if c not in tgt.columns]
    if miss:
        return ev

    if "home_win" not in tgt.columns:
        if {"score_home","score_away"}.issubset(tgt.columns):
            tgt["home_win"] = (pd.to_numeric(tgt["score_home"], errors="coerce")
                               > pd.to_numeric(tgt["score_away"], errors="coerce")).astype(int)
        else:
            return ev

    def pair(a,b): 
        a=str(a); b=str(b); 
        a=norm_team(a); b=norm_team(b); 
        return a+"_"+b if a<b else b+"_"+a

    tgt["pair"] = tgt.apply(lambda r: pair(r["home_team"], r["away_team"]), axis=1)
    tgt_min = tgt[["season","week","pair","home_win"]].drop_duplicates()

    ev["pair"] = ev.apply(lambda r: pair(r["team"], r["opponent"]), axis=1)
    ev = ev.merge(tgt_min, on=["season","week","pair"], how="left", validate="m:1")
    ev["won"] = np.where(ev["side"].str.lower().eq("home"), ev["home_win"], 1 - ev["home_win"])
    ev = ev.drop(columns=["home_win","pair"])
    return ev

# =========================
# Main
# =========================
def main():
    target_season = int(os.environ.get("TARGET_SEASON", detect_target_season()))

    # 1) Cargar stats (hist + actual)
    pre_all = load_pregame_all(target_season)
    if pre_all.empty:
        raise FileNotFoundError("No pregame stats found in archive/live.")

    # 2) Cargar odds temporada actual
    odds_cur = load_odds_season_current(target_season)
    if odds_cur.empty:
        raise FileNotFoundError("No live odds found at data/live/odds.csv.")

    # 3) Build master para temporada target
    master_cur = build_master_from_repo(odds_cur, pre_all)

    # 4) Master histórico (si hay odds históricos en archive) — opcional
    frames_hist = []
    for y in range(target_season-4, target_season):
        p_odds = os.path.join(ARCHIVE_DIR, f"season={y}", "odds.csv")
        p_stat = os.path.join(ARCHIVE_DIR, f"season={y}", "stats.csv")
        if os.path.exists(p_odds) and os.path.exists(p_stat):
            try:
                oh = pd.read_csv(p_odds, low_memory=False)
                sh = pd.read_csv(p_stat, low_memory=False)
                sh["season"] = int(y)
                oh = oh[oh.get("season", y) == y].copy()
                mh = build_master_from_repo(oh, sh)
                if not mh.empty:
                    frames_hist.append(mh)
            except Exception:
                pass
    master_hist = pd.concat(frames_hist, ignore_index=True) if frames_hist else pd.DataFrame()

    # 5) Entrenar + predecir (idéntico a tu notebook)
    test_preds = train_and_predict(master_hist, master_cur, target_season)

    # 6) EV sobre temporada target (idéntico a tu notebook)
    # Merge odds actuales con preds
    cols_key = ["season","week","home_team","away_team"]
    need_cols = cols_key + ["week_label","schedule_date",
                            "decimal_home","decimal_away","ml_home","ml_away",
                            "market_prob_home_nv","market_prob_away_nv","home_line","spread_favorite","over_under_line"]
    m24 = master_cur.copy()
    for c in ["decimal_home","decimal_away","ml_home","ml_away"]:
        if c not in m24.columns:
            m24[c] = np.nan
    merge_df = m24[[c for c in need_cols if c in m24.columns]].copy()
    if "week_label" not in merge_df.columns:
        merge_df["week_label"] = merge_df["week"].apply(week_label_from_num)

    pred = test_preds[["season","week","home_team","away_team","p_home_win_meta"]].copy()
    dfm  = merge_df.merge(pred, on=cols_key, how="inner")

    homes = pd.DataFrame({
        "season": dfm["season"], "week": dfm["week"], "week_label": dfm["week_label"], "schedule_date": dfm["schedule_date"],
        "side":"home", "team": dfm["home_team"].map(norm_team), "opponent": dfm["away_team"].map(norm_team),
        "decimal_odds": pd.to_numeric(dfm["decimal_home"], errors="coerce"), "ml": dfm["ml_home"],
        "market_prob_nv": dfm.get("market_prob_home_nv", np.nan),
        "model_prob": dfm["p_home_win_meta"]
    })
    aways = pd.DataFrame({
        "season": dfm["season"], "week": dfm["week"], "week_label": dfm["week_label"], "schedule_date": dfm["schedule_date"],
        "side":"away", "team": dfm["away_team"].map(norm_team), "opponent": dfm["home_team"].map(norm_team),
        "decimal_odds": pd.to_numeric(dfm["decimal_away"], errors="coerce"), "ml": dfm["ml_away"],
        "market_prob_nv": dfm.get("market_prob_away_nv", np.nan),
        "model_prob": 1.0 - dfm["p_home_win_meta"]
    })
    ev_predictions = pd.concat([homes, aways], ignore_index=True)

    ev_predictions = ev_predictions[pd.to_numeric(ev_predictions["decimal_odds"], errors="coerce").notna()].copy()
    ev_predictions["decimal_odds"] = ev_predictions["decimal_odds"].astype(float)

    ev_predictions["ev"]   = ev_predictions["model_prob"]*(ev_predictions["decimal_odds"]-1) - (1-ev_predictions["model_prob"])
    ev_predictions["edge"] = ev_predictions["model_prob"] - ev_predictions["market_prob_nv"].astype(float)

    ev_predictions["schedule_date"] = pd.to_datetime(ev_predictions["schedule_date"], errors="coerce", utc=True)
    ev_predictions = ev_predictions.sort_values(["schedule_date","week","team","side"]).reset_index(drop=True)

    # 7) Sanitizar, devig, selección, staking
    ev_base  = sanitize_ev(ev_predictions, CFG)
    ev_base  = backfill_won_if_missing(ev_base)
    ev_devig = devig_per_game(ev_base.assign(game_id=make_game_id(ev_base)), single_side_mode=CFG["DEVIG_SINGLE_SIDE"])

    p = ev_devig["model_prob"].astype(float)
    d = ev_devig["decimal_odds"].astype(float)
    mkt = ev_devig["market_prob_nv"].astype(float)
    ev_devig["edge"] = p - mkt
    ev_devig["ev"]   = p*(d-1) - (1-p)

    cand = pick_per_game_best(ev_devig, CFG)

    # 8) Simulamos bankroll desde inicio de temporada (bankroll inicial = 1000)
    bets_all = simulate_weekly_kelly(cand, CFG)

    # 9) Sólo publicar de Week 3+ (regla pedida)
    bets_all = bets_all[bets_all["week_label"].isin(ORDER_LABELS[2:])].copy()  # quita Week 1 & 2

    # Ajuste de tipos, ML american si falta
    if "ml" in bets_all.columns:
        miss = bets_all["ml"].isna() | (bets_all["ml"].astype(str).str.strip()=="")
        if miss.any():
            bets_all.loc[miss, "ml"] = bets_all.loc[miss, "decimal_odds"].apply(decimal_to_american)

    # Reorden final de columnas
    for col in BETS_COLS:
        if col not in bets_all.columns:
            bets_all[col] = pd.NA
    bets_out = bets_all[BETS_COLS].copy()

    # 10) Escribir CSV
    os.makedirs(os.path.dirname(BETS_OUT_PATH), exist_ok=True)
    bets_out.to_csv(BETS_OUT_PATH, index=False)
    print(f"[select_bets] wrote {BETS_OUT_PATH} | rows={len(bets_out)} at {datetime.now(timezone.utc).isoformat()}")

if __name__ == "__main__":
    main()
