#!/usr/bin/env python3
"""
select_bets.py (pregame staking only)

- Entrena el modelo (HGB + calibración + meta LR) igual al notebook.
- Usa stats pregame históricos (4 temporadas previas) + temporada actual.
- Une con odds actuales (data/live/odds.csv) solo para temporada target.
- Calcula EV/edge, aplica estrategia con filtros más estrictos y planifica stake pregame
  (Kelly fraccional + caps). NUNCA usa 'won' ni resultados. Bankroll SIEMPRE = INITIAL_BANKROLL.
- Exporta apuestas seleccionadas y stake a data/live/bets.csv
"""

import os, re, warnings
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

warnings.filterwarnings("ignore")

# ----------------------------
# Debug helper
# ----------------------------
DEBUG = True
def dprint(*args):
    if DEBUG:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[DBG {ts}]", *args, flush=True)

# ----------------------------
# Constantes y paths
# ----------------------------
ET = ZoneInfo("America/New_York")
LIVE_ODDS_PATH = os.path.join("data", "live", "odds.csv")
LIVE_STATS_PATH = os.path.join("data", "live", "stats.csv")
ARCHIVE_DIR = os.path.join("data", "archive")
BETS_OUT_PATH = os.path.join("data", "live", "bets.csv")

TEAM_FIX = {
    "STL":"LA","LAR":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV","WSH":"WAS","JAC":"JAX",
    "GNB":"GB","KAN":"KC","NWE":"NE","NOR":"NO","SFO":"SF","TAM":"TB"
}

ORDER_LABELS = [f"Week {i}" for i in range(1,19)] + ["Wild Card","Divisional","Conference","Super Bowl"]
ORDER_INDEX  = {lab:i for i,lab in enumerate(ORDER_LABELS)}
MIN_PROB = 1e-6

# --------- Filtros MÁS estrictos + tope semanal menor ---------
CFG = dict(
    INITIAL_BANKROLL = 1000.0,

    # Kelly & caps
    KELLY_FRACTION   = 0.25,
    KELLY_PCT_CAP    = 0.05,
    ABS_STAKE_CAP    = 300.0,
    WEEKLY_CAP_PCT   = 0.50,     # cap semanal = 50% del bankroll inicial (por semana)

    # Filtros globales
    ODDS_MIN         = 1.20,
    ODDS_MAX         = 3.40,
    CONF_MIN         = 0.090,    # antes 0.070
    EDGE_TAU         = 0.060,    # antes 0.045
    EDGE_TAU_MIN     = 0.040,
    EV_BASE_MIN      = 0.012,    # un pelín más alto
    EV_SLOPE         = 0.012,    # un pelín más alto
    SKIP_W1_W2       = True,

    DEVIG_SINGLE_SIDE = "raw",

    KELLY_EDGE_WEIGHT = dict(EDGE_REF=0.12, MIN_SCALE=0.60, MAX_SCALE=1.00),

    MAX_BETS_PER_WEEK = 6,       # baja el tope semanal (antes 9)
    MAX_BIG_DOGS_PER_WEEK = 1,

    # Bandas más exigentes
    BANDS = dict(
        FAV = dict(odds_lt=1.60,                 tau=0.050, conf=0.075, ev_slope=0.010),
        MID = dict(odds_ge=1.60, odds_lt=2.40,   tau=0.055, conf=0.080, ev_slope=0.012),
        DOG = dict(odds_ge=2.40, odds_le=3.40,   tau=0.065, conf=0.095, ev_slope=0.017,
                   extra_if_ge3_10=dict(tau=0.070, conf=0.100))
    ),
)

# ----------------------------
# Helpers básicos
# ----------------------------
def getenv_int(name: str, default_val: int) -> int:
    v = os.environ.get(name, None)
    try:
        s = str(v).strip()
        return int(s) if s != "" else default_val
    except (TypeError, ValueError):
        return default_val

def getenv_int_or_none(name: str):
    v = os.environ.get(name, None)
    try:
        s = str(v).strip()
        return int(s) if s != "" else None
    except (TypeError, ValueError):
        return None

def detect_target_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 3 else now.year - 1

def norm_team(s: str) -> str:
    s = str(s).upper().strip()
    return TEAM_FIX.get(s, s)

def week_label_from_num(n:int) -> str:
    return f"Week {n}" if 1<=n<=18 else {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")

def add_week_order(df):
    out = df.copy()
    out["week_label"] = out["week_label"].astype(str)
    out["week_order"] = out["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    return out

def make_pair(a, b):
    a = str(a); b = str(b)
    return a + "_" + b if a < b else b + "_" + a

def make_game_id_row(week_label: str, team: str, opp: str) -> str:
    a, b = str(team), str(opp)
    pair = a + "_" + b if a < b else b + "_" + a
    return f"{week_label} | {pair}"

def implied_from_decimal(d):
    d = float(d); return np.clip(1.0/max(d,1e-9), MIN_PROB, 1-MIN_PROB)

def american_to_decimal(m):
    if pd.isna(m): return np.nan
    m = float(m)
    return 1 + (100/abs(m) if m < 0 else m/100)

def decimal_to_american(d):
    if pd.isna(d): return np.nan
    d = float(d)
    return round((d - 1) * 100, 0) if d >= 2.0 else round(-100 / (d - 1), 0)

# ----------------------------
# Carga datasets
# ----------------------------
def load_current_odds() -> pd.DataFrame:
    dprint("LOAD odds live path:", LIVE_ODDS_PATH, "| exists:", os.path.exists(LIVE_ODDS_PATH))
    if not os.path.exists(LIVE_ODDS_PATH):
        raise FileNotFoundError(f"{LIVE_ODDS_PATH} not found.")
    df = pd.read_csv(LIVE_ODDS_PATH, low_memory=False)
    dprint("odds live raw shape:", df.shape, "| cols:", list(df.columns)[:12], "...")
    for c in ("season","week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce", utc=True)
    for c in ("home_team","away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_team)
    if "home_win" not in df.columns and {"score_home","score_away"}.issubset(df.columns):
        # NO usamos 'home_win' luego, pero si existe la derivamos por consistencia
        df["home_win"] = (pd.to_numeric(df["score_home"], errors="coerce")
                          > pd.to_numeric(df["score_away"], errors="coerce")).astype("Int64")
    dprint("odds live normalized shape:", df.shape, "| seasons:", sorted(df["season"].dropna().unique().tolist()))
    return df

def load_pregame_stats_for_seasons(target_season: int) -> pd.DataFrame:
    frames = []
    seasons_seen = []
    for y in range(target_season-4, target_season):
        p = os.path.join(ARCHIVE_DIR, f"season={y}", "stats.csv")
        dprint("CHECK stats archive:", p, "| exists:", os.path.exists(p))
        if os.path.exists(p):
            tmp = pd.read_csv(p, low_memory=False)
            tmp["season"] = pd.to_numeric(tmp["season"], errors="coerce").astype("Int64")
            tmp["week"]   = pd.to_numeric(tmp["week"], errors="coerce").astype("Int64")
            tmp["team"]   = tmp["team"].astype(str).map(norm_team)
            frames.append(tmp); seasons_seen.append(y)
            dprint(f"stats season {y} shape:", tmp.shape, "| weeks sample:", tmp["week"].dropna().unique()[:6])
    dprint("CHECK live stats:", LIVE_STATS_PATH, "| exists:", os.path.exists(LIVE_STATS_PATH))
    if os.path.exists(LIVE_STATS_PATH):
        tmp = pd.read_csv(LIVE_STATS_PATH, low_memory=False)
        tmp["season"] = pd.to_numeric(tmp["season"], errors="coerce").astype("Int64")
        tmp["week"]   = pd.to_numeric(tmp["week"], errors="coerce").astype("Int64")
        tmp["team"]   = tmp["team"].astype(str).map(norm_team)
        frames.append(tmp)
        seasons_seen.append(int(pd.to_numeric(tmp["season"], errors="coerce").dropna().max()))
        dprint("stats live shape:", tmp.shape, "| weeks live sample:", tmp["week"].dropna().unique()[:6])

    if not frames:
        raise FileNotFoundError("No se encontraron stats pregame para entrenar/pred.")
    df = pd.concat(frames, ignore_index=True)
    before = df.shape
    df = (df.sort_values(["season","week","team"])
            .drop_duplicates(["season","week","team"], keep="last"))
    dprint("stats all concat shape:", before, "-> unique shape:", df.shape, "| seasons seen:", sorted(set(seasons_seen)))
    return df

# ----------------------------
# Merge odds + pregame a nivel juego
# ----------------------------
def ensure_home_line_from_odds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_line" not in out.columns:
        if "spread_home" in out.columns:
            out["home_line"] = pd.to_numeric(out["spread_home"], errors="coerce")
        else:
            dec_h = pd.to_numeric(out.get("decimal_home", np.nan), errors="coerce")
            dec_a = pd.to_numeric(out.get("decimal_away", np.nan), errors="coerce")
            out["home_line"] = np.where(dec_h.notna() & dec_a.notna(),
                                        np.where(dec_h < dec_a, -1.5, 1.5), np.nan)
    return out

def add_pref(df: pd.DataFrame, pref: str, key: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ("season","week", key)]
    ren = {c:f"{pref}{c}" for c in cols}
    return df.rename(columns=ren)

def build_master(odds_df: pd.DataFrame, pre_df: pd.DataFrame) -> pd.DataFrame:
    dprint("build_master: incoming odds shape:", odds_df.shape, "| pre shape:", pre_df.shape)
    out = odds_df.copy()
    out["home_team"] = out["home_team"].map(norm_team)
    out["away_team"] = out["away_team"].map(norm_team)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["week"]   = pd.to_numeric(out["week"],   errors="coerce").astype("Int64")
    if "week_label" not in out.columns:
        out["week_label"] = out["week"].apply(week_label_from_num)
    out = ensure_home_line_from_odds(out)

    pre = pre_df.copy()
    pre["team"]   = pre["team"].map(norm_team)
    pre["season"] = pd.to_numeric(pre["season"], errors="coerce").astype("Int64")
    pre["week"]   = pd.to_numeric(pre["week"],   errors="coerce").astype("Int64")
    pre = (pre.sort_values(["season","week","team"])
             .drop_duplicates(["season","week","team"], keep="last"))

    home_pre = add_pref(pre.rename(columns={"team":"home_team"}), "home_", key="home_team")
    away_pre = add_pref(pre.rename(columns={"team":"away_team"}), "away_", key="away_team")

    m = (out.merge(home_pre, on=["season","week","home_team"], how="left")
            .merge(away_pre, on=["season","week","away_team"], how="left"))

    if "schedule_date" in m.columns:
        m["schedule_date"] = pd.to_datetime(m["schedule_date"], errors="coerce", utc=True)
    m = m.sort_values(["schedule_date","season","week","home_team"]).reset_index(drop=True)
    dprint("build_master: merged master shape:", m.shape,
           "| nulls home_* sample:",
           int(m.filter(like="home_").isna().sum().sum()),
           "| nulls away_* sample:",
           int(m.filter(like="away_").isna().sum().sum()))
    return m

# ----------------------------
# Modelo (igual al notebook) + métricas
# ----------------------------
def run_model(master_all: pd.DataFrame, target_season: int):
    dprint("run_model: master_all shape:", master_all.shape,
           "| seasons present:", sorted(master_all["season"].dropna().unique().tolist()))
    df = master_all.copy()

    # Home line fallback
    def compute_home_line(r):
        s  = r.get("spread_favorite", np.nan)
        tf = str(r.get("team_favorite_id", ""))
        if pd.isna(s) or tf == "" or tf == "nan":
            return r.get("home_line", np.nan)
        if "home_team" in r and tf == r["home_team"]:
            return float(s)
        if "away_team" in r and tf == r["away_team"]:
            return float(-s)
        return r.get("home_line", np.nan)

    if "home_line" not in df.columns or df["home_line"].isna().any():
        df["home_line"] = df.apply(compute_home_line, axis=1)

    df["abs_spread"] = pd.to_numeric(df["home_line"], errors="coerce").astype(float).abs()
    df["fav_home"]   = (pd.to_numeric(df["home_line"], errors="coerce").astype(float) < 0).astype(int)
    df["ou"]         = pd.to_numeric(df.get("over_under_line", np.nan), errors="coerce").astype(float)
    df["spread_x_ou"]   = df["abs_spread"] * df["ou"]
    df["fav_x_spread"]  = df["fav_home"]   * df["abs_spread"]
    df["home_line_sq"]  = df["home_line"]  * df["home_line"]

    # features
    pre_cols = [c for c in df.columns if re.search(r'(?:_pre(_ewm)?|_pre_ytd|_pre_l8)$', c)]
    mkt_cols = [c for c in ["home_line","abs_spread","fav_home","ou","spread_x_ou","fav_x_spread","home_line_sq"]
                if c in df.columns]
    feat_cols = [c for c in (pre_cols + mkt_cols) if c in df.columns]
    dprint("run_model: #feat_cols:", len(feat_cols), "| example feats:", feat_cols[:10])

    # Split temporal: train = target-4..target-2; val = target-1; test = target
    hist_mask = df["season"] < target_season
    hist_years = sorted(df.loc[hist_mask, "season"].dropna().unique().tolist())
    dprint("run_model: hist_years available:", hist_years, "| target_season:", target_season)
    if len(hist_years) < 2:
        raise RuntimeError("Se requieren al menos 2 temporadas históricas para entrenar/validar.")

    if len(hist_years) >= 4:
        train_years = hist_years[-4:-1]
        val_year    = hist_years[-1]
    else:
        train_years = hist_years[:-1]
        val_year    = hist_years[-1]
    test_year  = target_season
    dprint("run_model: split -> train_years:", train_years, "| val_year:", val_year, "| test_year:", test_year)

    base = df.dropna(subset=["home_line"]).copy()
    dprint("run_model: base (con home_line) shape:", base.shape)

    train_df = base[base["season"].isin(train_years)].copy()
    val_df   = base[base["season"].eq(val_year)].copy()
    test_df  = base[base["season"].eq(test_year)].copy()
    dprint("run_model: train/val/test shapes:", train_df.shape, val_df.shape, test_df.shape)

    # y labels
    y_train = train_df["home_win"].dropna().astype(int).values
    y_val   = val_df["home_win"].dropna().astype(int).values
    dprint("run_model: y_train len:", len(y_train), "| y_val len:", len(y_val))

    # X
    X_train = train_df[feat_cols].copy()
    X_val   = val_df[feat_cols].copy()
    X_test  = test_df[feat_cols].copy()

    train_meds = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_meds); X_val = X_val.fillna(train_meds); X_test = X_test.fillna(train_meds)
    dprint("run_model: X shapes:", X_train.shape, X_val.shape, X_test.shape)

    # Monotonicidad (home_line negativa)
    monotonic_cst = [(-1 if c == "home_line" else 0) for c in feat_cols]

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.06, max_leaf_nodes=21, min_samples_leaf=60,
        l2_regularization=2.0, max_iter=320, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=40,
        monotonic_cst=monotonic_cst, random_state=42
    )
    dprint("run_model: fitting HGB...")
    hgb.fit(X_train, y_train)

    p_tr = hgb.predict_proba(X_train)[:,1]
    p_va = hgb.predict_proba(X_val)[:,1]
    p_te = hgb.predict_proba(X_test)[:,1] if len(X_test) else np.array([])
    dprint("run_model: raw preds lens -> train/val/test:", len(p_tr), len(p_va), len(p_te))

    # Calibración (global + por bins si hay soporte)
    bin_edges = [-21, -10.5, -6.5, -3.5, -0.5, 0.5, 3.5, 6.5, 10.5, 21]
    cut_bins = lambda x: pd.cut(x, bins=bin_edges, include_lowest=True)
    val_bins  = cut_bins(val_df["home_line"])
    test_bins = cut_bins(test_df["home_line"]) if len(test_df) else pd.Series(dtype="category")

    iso_global = IsotonicRegression(out_of_bounds="clip")
    iso_global.fit(p_va, y_val); dprint("run_model: fitted iso_global.")

    bin_min = 120
    iso_by_bin = {}
    for b in val_bins.unique():
        idx = (val_bins == b)
        if idx.sum() >= bin_min:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_va[idx], y_val[idx]); iso_by_bin[b] = iso
    dprint("run_model: bins with custom iso:", len(iso_by_bin))

    def apply_bin_calibration(p_raw, bins_series):
        if len(p_raw) == 0: return p_raw
        if not iso_by_bin: return np.clip(iso_global.transform(p_raw), 1e-6, 1-1e-6)
        out = np.empty_like(p_raw, dtype=float)
        for b in bins_series.unique():
            m = (bins_series == b).values
            out[m] = (iso_by_bin[b].transform(p_raw[m]) if b in iso_by_bin else iso_global.transform(p_raw[m]))
        return np.clip(out, 1e-6, 1-1e-6)

    p_tr_cal = iso_global.transform(p_tr)
    p_va_cal = apply_bin_calibration(p_va, val_bins)
    p_te_cal = apply_bin_calibration(p_te, test_bins)
    dprint("run_model: calibrated preds lens -> train/val/test:", len(p_tr_cal), len(p_va_cal), len(p_te_cal))

    # Prior por spread y meta-LR
    prior_lr = LogisticRegression(C=2.0, solver="liblinear", max_iter=200)
    prior_lr.fit(train_df[["home_line"]], y_train); dprint("run_model: fitted prior LR on home_line.")
    p_tr_sp = prior_lr.predict_proba(train_df[["home_line"]])[:,1]
    p_va_sp = prior_lr.predict_proba(val_df[["home_line"]])[:,1]
    p_te_sp = prior_lr.predict_proba(test_df[["home_line"]])[:,1] if len(test_df) else np.array([])

    def meta_matrix(df_in, p_hgb_cal, p_spread):
        M = pd.DataFrame({
            "p_hgb":   p_hgb_cal,
            "p_sp":    p_spread,
            "home_line": df_in["home_line"].values,
            "abs_spread": df_in["abs_spread"].values,
            "ou": df_in["ou"].values
        })
        return M.fillna(M.median(numeric_only=True))

    meta_tr   = meta_matrix(train_df, p_tr_cal, p_tr_sp)
    meta_val  = meta_matrix(val_df,   p_va_cal, p_va_sp)
    meta_test = meta_matrix(test_df,  p_te_cal, p_te_sp) if len(test_df) else pd.DataFrame()

    meta_lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=300)
    meta_lr.fit(meta_val, y_val)
    dprint("run_model: fitted meta LR. meta shapes ->", meta_tr.shape, meta_val.shape, meta_test.shape)

    p_te_meta = (np.clip(meta_lr.predict_proba(meta_test)[:,1], 1e-6, 1-1e-6)
                 if len(meta_test) else np.array([]))
    dprint("run_model: p_te_meta len:", len(p_te_meta))

    # Métricas comparables (entrenamiento/validación)
    acc_tr  = accuracy_score(y_train, (p_tr_cal >= 0.5).astype(int))
    auc_tr  = roc_auc_score(y_train, p_tr_cal)
    ll_tr   = log_loss(y_train, p_tr_cal)
    acc_va  = accuracy_score(y_val, (p_va_cal >= 0.5).astype(int))
    auc_va  = roc_auc_score(y_val, p_va_cal)
    ll_va   = log_loss(y_val, p_va_cal)

    dprint("\n[DBG Metrics]")
    dprint(f"train | ACC {acc_tr:.3f} | ROC_AUC {auc_tr:.3f} | LOGLOSS {ll_tr:.3f}")
    dprint(f"val   | ACC {acc_va:.3f} | ROC_AUC {auc_va:.3f} | LOGLOSS {ll_va:.3f}")
    dprint("test  | pregame only (no labels, no results)")

    # Preds para temporada target
    test_preds = test_df[["season","week","week_label","home_team","away_team"]].copy()
    if "home_win" in test_df.columns:
        # no usamos 'home_win', pero si existe la colocamos aparte para depurar
        pass
    test_preds["p_home_win_lr_cal"] = p_te_cal
    test_preds["p_home_win_meta"]   = p_te_meta
    dprint("run_model: test_preds shape:", test_preds.shape)

    return test_preds

# ----------------------------
# EV, selección y staking pregame
# ----------------------------
def sanitize_ev(df: pd.DataFrame, cfg) -> pd.DataFrame:
    need = ["season","week","week_label","schedule_date","side","team","opponent","decimal_odds","model_prob"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        dprint("sanitize_ev: MISSING COLUMNS ->", missing)
        raise ValueError(f"EV table missing columns: {missing}")

    ev = df.copy()
    ev["schedule_date"] = pd.to_datetime(ev["schedule_date"], errors="coerce", utc=True)
    ev["model_prob"] = pd.to_numeric(ev["model_prob"], errors="coerce").astype(float).clip(MIN_PROB, 1-MIN_PROB)
    ev["decimal_odds"] = pd.to_numeric(ev["decimal_odds"], errors="coerce").astype(float)

    ev = add_week_order(ev)
    ev["game_id"] = ev.apply(lambda r: make_game_id_row(r["week_label"], r["team"], r["opponent"]), axis=1)

    before = ev.shape
    ev = ev[ev["decimal_odds"].between(cfg["ODDS_MIN"], cfg["ODDS_MAX"])]
    ev = ev[np.abs(ev["model_prob"] - 0.5) >= cfg["CONF_MIN"]]
    if cfg["SKIP_W1_W2"]:
        ev = ev[~ev["week_label"].isin(["Week 1","Week 2"])]

    ev = ev.sort_values(["week_order","schedule_date","game_id"]).reset_index(drop=True)
    dprint("sanitize_ev:", before, "->", ev.shape, "| kept after filters")
    return ev

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

    ev["market_prob_nv"] = ev["market_prob_nv"].clip(1e-6, 1-1e-6)
    return ev

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

    out = x[keep].reset_index(drop=True)
    dprint("apply_band_rules: in", df.shape, "-> out", out.shape)
    return out

def pick_per_game_best(ev: pd.DataFrame, cfg) -> pd.DataFrame:
    tau = max(cfg["EDGE_TAU"], cfg["EDGE_TAU_MIN"])

    def ok_ev(row):
        slope = ev_slope_for_row(row, cfg)
        return row["ev"] >= ev_floor(row["decimal_odds"], cfg["EV_BASE_MIN"], slope)

    c = ev[ev["edge"] >= tau].copy()
    if c.empty:
        dprint("pick_per_game_best: no candidates after edge tau.")
        return c
    c = c[c.apply(ok_ev, axis=1)]

    c = (c.sort_values(["week_order","schedule_date","game_id","edge"],
                       ascending=[True,True,True,False])
           .drop_duplicates("game_id"))

    c = apply_band_rules(c, cfg)

    if cfg.get("MAX_BIG_DOGS_PER_WEEK"):
        out = []
        for wk, g in c.groupby("week_label", sort=False):
            m_big = g["decimal_odds"] >= 3.10
            if m_big.sum() > cfg["MAX_BIG_DOGS_PER_WEEK"]:
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

    dprint("pick_per_game_best: final candidates:", c.shape)
    return c.reset_index(drop=True)

# -------- Kelly fraccional + caps (PREGAME, sin resultados) ----------
def kelly_fraction_scaled(edge, cfg):
    wd = cfg.get("KELLY_EDGE_WEIGHT")
    if not wd: return cfg["KELLY_FRACTION"]
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

def plan_pregame_stakes(bets_in: pd.DataFrame, cfg):
    """
    Asigna stake PRE-GAME a todos los picks por semana, con bankroll FIJO = INITIAL_BANKROLL.
    No calcula ni usa 'won' ni resultados. 'profit' se deja como NaN. 'bankroll' muestra el BK0.
    """
    if bets_in.empty:
        dprint("plan_pregame_stakes: no bets; returning zeros.")
        return bets_in.assign(stake=0.0, profit=np.nan, bankroll=cfg["INITIAL_BANKROLL"]), pd.DataFrame(), {
            "bets_planned": 0, "total_planned_stake": 0.0, "bankroll_assumed": cfg["INITIAL_BANKROLL"]
        }

    data = add_week_order(bets_in).sort_values(["week_order","schedule_date","game_id"]).reset_index(drop=True)

    BK0 = cfg["INITIAL_BANKROLL"]
    stakes, bk_view = [], []

    for wk, g in data.groupby("week_label", sort=False):
        # calcular stake con BK0 y cap semanal
        if cfg["WEEKLY_CAP_PCT"] >= 0.999:
            g_stakes = [kelly_stake(BK0, float(r["model_prob"]), float(r["decimal_odds"]), float(r["edge"]), cfg)
                        for _, r in g.iterrows()]
        else:
            g_stakes = stake_week_with_cap(g, BK0, cfg)

        stakes.extend(g_stakes)
        bk_view.extend([BK0]*len(g))  # SIEMPRE el mismo bankroll mostrado (no se mueve)

    out = data.copy()
    out["stake"] = np.round(stakes, 2)
    out["profit"] = np.nan
    out["bankroll"] = float(BK0)

    # Resumen por semana (solo planificación)
    wk_summary = (out.groupby("week_label", sort=False)
                    .agg(stake=("stake","sum"))
                    .reset_index())
    wk_summary["bankroll"] = BK0
    wk_summary.rename(columns={"stake":"planned_stake"}, inplace=True)

    summary = {
        "bets_planned": int(len(out)),
        "total_planned_stake": float(np.round(out["stake"].sum(), 2)),
        "bankroll_assumed": float(BK0)
    }
    dprint("plan_pregame_stakes: summary ->", summary)
    return out, wk_summary, summary

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(os.path.dirname(BETS_OUT_PATH), exist_ok=True)

    target_season = getenv_int("TARGET_SEASON", detect_target_season())
    target_week_override = getenv_int_or_none("TARGET_WEEK")
    dprint("ENV -> TARGET_SEASON:", os.environ.get("TARGET_SEASON"),
           "TARGET_WEEK:", os.environ.get("TARGET_WEEK"))
    dprint("Resolved -> target_season:", target_season, "| target_week_override:", target_week_override)

    # Carga
    odds_cur = load_current_odds()
    stats_all = load_pregame_stats_for_seasons(target_season)

    # Odds solo de temporada target (+ semana si la pasas por env)
    odds_cur = odds_cur[odds_cur["season"].eq(target_season)].copy()
    if target_week_override is not None:
        odds_cur = odds_cur[odds_cur["week"].eq(target_week_override)].copy()
    dprint("odds_cur (filtered to target) shape:", odds_cur.shape,
           "| weeks:", sorted(odds_cur["week"].dropna().unique().tolist()) if len(odds_cur) else [])

    # Master target
    master_target = build_master(odds_cur, stats_all)
    dprint("master_target shape:", master_target.shape)

    # Cargar odds históricas de 4 temporadas previas (para features y labels históricas)
    hist_frames = []
    for y in range(target_season-4, target_season):
        p = os.path.join(ARCHIVE_DIR, f"season={y}", "odds.csv")
        dprint("CHECK odds archive:", p, "| exists:", os.path.exists(p))
        if os.path.exists(p):
            tmp = pd.read_csv(p, low_memory=False)
            for c in ("season","week"):
                if c in tmp.columns:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce").astype("Int64")
            if "schedule_date" in tmp.columns:
                tmp["schedule_date"] = pd.to_datetime(tmp["schedule_date"], errors="coerce", utc=True)
            for c in ("home_team","away_team"):
                if c in tmp.columns:
                    tmp[c] = tmp[c].astype(str).map(norm_team)
            if "home_win" not in tmp.columns and {"score_home","score_away"}.issubset(tmp.columns):
                tmp["home_win"] = (pd.to_numeric(tmp["score_home"], errors="coerce")
                                  > pd.to_numeric(tmp["score_away"], errors="coerce")).astype("Int64")
            hist_frames.append(tmp)
            dprint(f"odds season {y} shape:", tmp.shape)

    master_hist = pd.DataFrame()
    if hist_frames:
        odds_hist = pd.concat(hist_frames, ignore_index=True)
        dprint("odds_hist concat shape:", odds_hist.shape,
               "| seasons:", sorted(odds_hist["season"].dropna().unique().tolist()))
        master_hist = build_master(odds_hist, stats_all)
        dprint("master_hist shape:", master_hist.shape)
    else:
        dprint("No historical odds found in archive; proceeding with target only.")

    master_all = pd.concat([master_hist, master_target], ignore_index=True) if not master_hist.empty else master_target.copy()
    dprint("master_all shape:", master_all.shape,
           "| seasons in master_all:", sorted(master_all["season"].dropna().unique().tolist()))

    # Modelo + métricas
    test_preds = run_model(master_all, target_season)

    # Unir probs con odds target
    cols_key = ["season","week","home_team","away_team"]
    need_cols = cols_key + ["week_label","schedule_date",
                            "decimal_home","decimal_away","ml_home","ml_away",
                            "market_prob_home_nv","market_prob_away_nv","home_line","spread_favorite","over_under_line"]
    merge_df = master_target[[c for c in need_cols if c in master_target.columns]].copy()
    if "week_label" not in merge_df.columns:
        merge_df["week_label"] = merge_df["week"].apply(week_label_from_num)
    dprint("merge_df (odds target) shape:", merge_df.shape)

    pred = test_preds[["season","week","home_team","away_team","p_home_win_meta"]].copy()
    dfm  = merge_df.merge(pred, on=cols_key, how="inner")
    dprint("dfm (odds+preds) shape:", dfm.shape)

    # EV home/away
    homes = pd.DataFrame({
        "season": dfm["season"], "week": dfm["week"], "week_label": dfm["week_label"], "schedule_date": dfm["schedule_date"],
        "side":"home", "team": dfm["home_team"], "opponent": dfm["away_team"],
        "decimal_odds": dfm.get("decimal_home", np.nan), "ml": dfm.get("ml_home", np.nan),
        "market_prob_nv": dfm.get("market_prob_home_nv", np.nan),
        "model_prob": dfm["p_home_win_meta"]
    })
    aways = pd.DataFrame({
        "season": dfm["season"], "week": dfm["week"], "week_label": dfm["week_label"], "schedule_date": dfm["schedule_date"],
        "side":"away", "team": dfm["away_team"], "opponent": dfm["home_team"],
        "decimal_odds": dfm.get("decimal_away", np.nan), "ml": dfm.get("ml_away", np.nan),
        "market_prob_nv": dfm.get("market_prob_away_nv", np.nan),
        "model_prob": 1.0 - dfm["p_home_win_meta"]
    })
    ev_predictions = pd.concat([homes, aways], ignore_index=True)
    dprint("ev_predictions (home+away) shape:", ev_predictions.shape)

    # Completar ML si falta
    if "ml" in ev_predictions.columns and "decimal_odds" in ev_predictions.columns:
        m = ev_predictions["ml"].isna() & ev_predictions["decimal_odds"].notna()
        ev_predictions.loc[m, "ml"] = ev_predictions.loc[m, "decimal_odds"].apply(decimal_to_american)

    ev_predictions = ev_predictions[pd.to_numeric(ev_predictions["decimal_odds"], errors="coerce").notna()].copy()
    ev_predictions["decimal_odds"] = ev_predictions["decimal_odds"].astype(float)
    dprint("ev_predictions after dropna(decimal_odds) shape:", ev_predictions.shape)

    # EV / edge
    ev_predictions["ev"]   = ev_predictions["model_prob"]*(ev_predictions["decimal_odds"]-1) - (1-ev_predictions["model_prob"])
    ev_predictions = ev_predictions.sort_values(["schedule_date","week","team","side"]).reset_index(drop=True)

    # Limpieza/sanitizado y devig
    ev_base = sanitize_ev(ev_predictions, CFG)
    ev_base["game_id"] = ev_base.apply(lambda r: make_game_id_row(r["week_label"], r["team"], r["opponent"]), axis=1)
    ev_devig = devig_per_game(ev_base.copy(), single_side_mode=CFG["DEVIG_SINGLE_SIDE"])
    dprint("ev_devig shape:", ev_devig.shape)

    p = ev_devig["model_prob"].astype(float)
    d = ev_devig["decimal_odds"].astype(float)
    mkt = ev_devig["market_prob_nv"].astype(float)
    ev_devig["edge"] = p - mkt
    ev_devig["ev"]   = p*(d-1) - (1-p)

    # Embudo de verificación
    total_games = ev_predictions.assign(game_id=ev_predictions.apply(
                        lambda r: make_game_id_row(week_label_from_num(int(r["week"])) if "week_label" not in ev_predictions.columns else r["week_label"],
                                                   r["team"] if "team" in ev_predictions.columns else r["home_team"],
                                                   r["opponent"] if "opponent" in ev_predictions.columns else r["away_team"]),
                        axis=1)).drop_duplicates("game_id").shape[0]
    after_basic = ev_base.drop_duplicates("game_id").shape[0]
    pre_top = ev_devig.shape[0]
    dprint(f"FUNNEL -> games_total={total_games} | after_basic_filters={after_basic} | rows_after_devig={pre_top}")

    cand = pick_per_game_best(ev_devig, CFG)
    dprint("cand (picked bets) shape:", cand.shape)

    # --------- PLAN PREGAME (sin resultados) ----------
    planned_bets, weekly_plan, S = plan_pregame_stakes(cand, CFG)
    dprint("weekly_plan shape:", weekly_plan.shape if isinstance(weekly_plan, pd.DataFrame) else None,
           "| planning summary:", S)

    # Orden/columnas finales
    planned_bets = add_week_order(planned_bets)
    final_cols = [
        "season","week","week_label","schedule_date","side","team","opponent",
        "decimal_odds","ml","market_prob_nv","model_prob","ev","edge",
        "week_order","game_id","stake","profit","bankroll"
    ]
    for c in final_cols:
        if c not in planned_bets.columns:
            planned_bets[c] = pd.NA
    planned_bets = planned_bets[final_cols].copy()

    planned_bets.to_csv(BETS_OUT_PATH, index=False)
    print(f"[select_bets] wrote {BETS_OUT_PATH} | rows={len(planned_bets)} | season={target_season}")

if __name__ == "__main__":
    main()
