#!/usr/bin/env python3
"""
select_bets.py (pregame staking only) — v2 (relax + LH single-side + audits)

- Modelo igual que antes (HGB + calibración + meta LR) — NO tocado.
- Usa stats pregame (4 temporadas previas) + temporada actual.
- Une con odds actuales (data/live/odds.csv) para la semana VIGENTE detectada.
- EV/edge + selección con filtros y relajación escalonada (mín 5, máx 8).
- Kelly fraccional + caps semanales (usa BK0 por semana pero NO exporta bankroll).
- Auditorías detalladas (edge/conf/EV, near-miss) SIEMPRE impresas.
"""

import os, re, warnings, json
from datetime import datetime, timedelta, timezone
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
PNL_PATH = os.path.join("data", "live", "pnl.csv")

TEAM_FIX = {
    "STL":"LA","LAR":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV","WSH":"WAS","JAC":"JAX",
    "GNB":"GB","KAN":"KC","NWE":"NE","NOR":"NO","SFO":"SF","TAM":"TB"
}

ORDER_LABELS = [f"Week {i}" for i in range(1,19)] + ["Wild Card","Divisional","Conference","Super Bowl"]
ORDER_INDEX  = {lab:i for i,lab in enumerate(ORDER_LABELS)}
MIN_PROB = 1e-6

# --------- Filtros + límites (ajustados) ---------
CFG = dict(
    INITIAL_BANKROLL = 1000.0,

    # Kelly & caps (igual que antes)
    KELLY_FRACTION   = 0.25,
    KELLY_PCT_CAP    = 0.05,
    ABS_STAKE_CAP    = 300.0,
    WEEKLY_CAP_PCT   = 0.50,

    # Rango de cuotas (ajustado ligeramente para abrir espacio)
    ODDS_MIN         = 1.20,
    ODDS_MAX         = 3.60,

    # Filtros base (edge/conf/EV) — base algo más amable
    CONF_MIN         = 0.085,     # antes 0.090
    EDGE_TAU         = 0.050,     # antes 0.060
    EV_BASE_MIN      = 0.010,     # antes 0.012

    # Relajación deseada
    MIN_BETS_PER_WEEK = 5,
    MAX_BETS_PER_WEEK = 8,

    SKIP_W1_W2       = True,
    DEVIG_SINGLE_SIDE = "lh",     # << NUEVO: modo LH para single-side
    LH_FALLBACK_HOLD = 0.045,     # hold fallback si no hay pares suficientes
    LH_MIN_POS_EV    = 0.002,     # EV mínimo absoluto al final de toda relajación (nunca 0)

    # Bandas (EV slope adaptativo)
    BANDS = dict(
        FAV = dict(odds_lt=1.60,                 tau=0.048, conf=0.075, ev_slope=0.009),
        MID = dict(odds_ge=1.60, odds_lt=2.40,   tau=0.052, conf=0.080, ev_slope=0.011),
        DOG = dict(odds_ge=2.40, odds_le=3.60,   tau=0.062, conf=0.095, ev_slope=0.016,
                   extra_if_ge3_10=dict(tau=0.068, conf=0.100))
    ),
)

# Pasos de relajación (más agresivos, pero con suelo EV > LH_MIN_POS_EV)
RELAX_STEPS = [
    dict(EDGE_TAU=0.050, CONF_MIN=0.080, EV_BASE_MIN=0.008),
    dict(EDGE_TAU=0.0475, CONF_MIN=0.075, EV_BASE_MIN=0.006),
    dict(EDGE_TAU=0.045, CONF_MIN=0.075, EV_BASE_MIN=0.006),
    dict(EDGE_TAU=0.040, CONF_MIN=0.070, EV_BASE_MIN=0.004),
    dict(EDGE_TAU=0.0375, CONF_MIN=0.065, EV_BASE_MIN=0.0035),
]

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

def implied_from_decimal(d):
    d = float(d); return np.clip(1.0/max(d,1e-9), MIN_PROB, 1-MIN_PROB)

def decimal_to_american(d):
    if pd.isna(d): return np.nan
    d = float(d)
    return round((d - 1) * 100, 0) if d >= 2.0 else round(-100 / (d - 1), 0)

def make_game_id_row(week_label: str, team: str, opp: str) -> str:
    a, b = str(team), str(opp)
    pair = a + "_" + b if a < b else b + "_" + a
    return f"{week_label} | {pair}"

# ----------------------------
# Semana vigente (elige una sola semana)
# ----------------------------
def detect_vigente_week_label(odds_df: pd.DataFrame) -> str:
    now = datetime.now(timezone.utc)
    if "schedule_date" in odds_df.columns:
        odds_df = odds_df.copy()
        odds_df["schedule_date"] = pd.to_datetime(odds_df["schedule_date"], errors="coerce", utc=True)
        horizon = now - timedelta(hours=18)
        future = odds_df[odds_df["schedule_date"] >= horizon]
        if not future.empty:
            wk = int(future["week"].dropna().min())
            wk_label = week_label_from_num(wk)
            dprint("Semana vigente detectada por fechas (>= now-18h):", wk_label)
            return wk_label
    wk = int(pd.to_numeric(odds_df["week"], errors="coerce").dropna().max())
    wk_label = week_label_from_num(wk)
    dprint("Semana vigente por fallback (max week en odds):", wk_label)
    return wk_label

# ----------------------------
# Self-learn (robusto)
# ----------------------------
def self_learn_adjusters():
    """
    Lee data/live/pnl.csv si existe y estima:
    - league_calib_delta: pequeño ajuste de calibración (shrink/expand) en single-side
    - hold_fallback: hold de liga observado (para DEVIG single-side)
    Silencioso y robusto: si no hay datos, usa defaults.
    """
    league_calib_delta = 0.0
    hold_fallback = CFG["LH_FALLBACK_HOLD"]

    try:
        if os.path.exists(PNL_PATH):
            df = pd.read_csv(PNL_PATH, low_memory=False)
            # tolerante con columnas
            if "unit_pnl" in df.columns:
                # media recortada simple
                s = pd.to_numeric(df["unit_pnl"], errors="coerce").dropna()
                if len(s) >= 5:
                    mu = s.clip(lower=s.quantile(0.1), upper=s.quantile(0.9)).mean()
                    # mapear PnL a pequeño delta de calibración [-0.02, +0.03]
                    league_calib_delta = float(np.clip(mu/100.0, -0.02, 0.03))
            if "observed_overround" in df.columns:
                h = pd.to_numeric(df["observed_overround"], errors="coerce").dropna()
                if len(h) >= 5:
                    hold_fallback = float(np.clip(h.median(), 0.02, 0.08))
    except Exception as e:
        dprint("WARN self-learn: error leyendo pnl.csv:", repr(e))

    dprint("Self-learn adjusters -> league:", {"league_calib_delta": league_calib_delta, "league_unit_pnl": df["unit_pnl"].mean() if 'df' in locals() and "unit_pnl" in df.columns else None, "hold_fallback": hold_fallback})
    return dict(league_calib_delta=league_calib_delta, hold_fallback=hold_fallback)

# ----------------------------
# Carga datasets
# ----------------------------
def load_current_odds() -> pd.DataFrame:
    dprint("LOAD odds live path:", LIVE_ODDS_PATH, "| exists:", os.path.exists(LIVE_ODDS_PATH))
    if not os.path.exists(LIVE_ODDS_PATH):
        raise FileNotFoundError(f"{LIVE_ODDS_PATH} not found.")
    df = pd.read_csv(LIVE_ODDS_PATH, low_memory=False)
    for c in ("season","week"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce", utc=True)
    for c in ("home_team","away_team"):
        if c in df.columns: df[c] = df[c].astype(str).map(norm_team)
    if "home_win" not in df.columns and {"score_home","score_away"}.issubset(df.columns):
        df["home_win"] = (pd.to_numeric(df["score_home"], errors="coerce")
                          > pd.to_numeric(df["score_away"], errors="coerce")).astype("Int64")
    return df

def load_pregame_stats_for_seasons(target_season: int) -> pd.DataFrame:
    frames = []
    for y in range(target_season-4, target_season+1):
        # prior seasons en archive + live para la actual
        if y < target_season:
            p = os.path.join(ARCHIVE_DIR, f"season={y}", "stats.csv")
        else:
            p = LIVE_STATS_PATH
        if os.path.exists(p):
            tmp = pd.read_csv(p, low_memory=False)
            tmp["season"] = pd.to_numeric(tmp["season"], errors="coerce").astype("Int64")
            tmp["week"]   = pd.to_numeric(tmp["week"], errors="coerce").astype("Int64")
            tmp["team"]   = tmp["team"].astype(str).map(norm_team)
            frames.append(tmp)
    if not frames:
        raise FileNotFoundError("No se encontraron stats pregame.")
    df = pd.concat(frames, ignore_index=True)
    df = (df.sort_values(["season","week","team"])
            .drop_duplicates(["season","week","team"], keep="last"))
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
    return m

# ----------------------------
# Modelo + métricas (NO tocado en lógica)
# ----------------------------
def run_model(master_all: pd.DataFrame, target_season: int):
    df = master_all.copy()

    # Fallback home_line
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

    pre_cols = [c for c in df.columns if re.search(r'(?:_pre(_ewm)?|_pre_ytd|_pre_l8)$', c)]
    mkt_cols = [c for c in ["home_line","abs_spread","fav_home","ou","spread_x_ou","fav_x_spread","home_line_sq"] if c in df.columns]
    feat_cols = [c for c in (pre_cols + mkt_cols) if c in df.columns]

    # Split
    hist_mask = df["season"] < target_season
    hist_years = sorted(df.loc[hist_mask, "season"].dropna().unique().tolist())
    if len(hist_years) < 2:
        raise RuntimeError("Se requieren al menos 2 temporadas históricas para entrenar/validar.")

    if len(hist_years) >= 4:
        train_years = hist_years[-4:-1]; val_year = hist_years[-1]
    else:
        train_years = hist_years[:-1];   val_year = hist_years[-1]
    test_year  = target_season

    base = df.dropna(subset=["home_line"]).copy()
    train_df = base[base["season"].isin(train_years)].copy()
    val_df   = base[base["season"].eq(val_year)].copy()
    test_df  = base[base["season"].eq(test_year)].copy()

    # Targets
    for part, part_df in [("Train", train_df), ("Val", val_df)]:
        y = pd.to_numeric(part_df["home_win"], errors="coerce")
        if y.isna().any():
            part_df["home_win"] = y.fillna(0).astype(int)

    y_train = train_df["home_win"].astype(int).values
    y_val   = val_df["home_win"].astype(int).values

    X_train = train_df[feat_cols].copy()
    X_val   = val_df[feat_cols].copy()
    X_test  = test_df[feat_cols].copy()

    # Imputación
    train_meds = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_meds); X_val = X_val.fillna(train_meds); X_test = X_test.fillna(train_meds)

    dprint(f"Features usadas: {len(feat_cols)} | train_rows={len(X_train)} | val_rows={len(X_val)} | test_rows={len(X_test)}")

    # Balance check
    for name, y in [("Train", y_train), ("Val", y_val)]:
        vals, cnts = np.unique(y, return_counts=True)
        dprint(f"{name} y unique:", (vals, cnts))

    # Modelo base
    monotonic_cst = [(-1 if c == "home_line" else 0) for c in feat_cols]
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.06, max_leaf_nodes=21, min_samples_leaf=60,
        l2_regularization=2.0, max_iter=320, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=40,
        monotonic_cst=monotonic_cst, random_state=42
    )
    hgb.fit(X_train, y_train)

    p_tr = hgb.predict_proba(X_train)[:,1]
    p_va = hgb.predict_proba(X_val)[:,1]
    p_te = hgb.predict_proba(X_test)[:,1] if len(X_test) else np.array([])

    # Métricas previas a calibración
    def safe_metrics(p, y, tag):
        if len(p)==0:
            print(f"Model Results:\ntrain | ACC --- | ROC_AUC --- | LOGLOSS ---\nval   | ACC --- | ROC_AUC --- | LOGLOSS ---")
            return
        acc = accuracy_score(y, (p>=0.5).astype(int))
        auc = roc_auc_score(y, p)
        ll  = log_loss(y, p, labels=[0,1])
        print("Model Results:")
        print(f"train | ACC {accuracy_score(y_train,(p_tr>=0.5).astype(int)):.3f} | ROC_AUC {roc_auc_score(y_train,p_tr):.3f} | LOGLOSS {log_loss(y_train,p_tr,labels=[0,1]):.3f}")
        print(f"val   | ACC {acc:.3f} | ROC_AUC {auc:.3f} | LOGLOSS {ll:.3f}")

    dprint("Metrics p-nans:", int(np.isnan(p_tr).sum()), "| p[min,med,max]:", np.min(p_tr), np.median(p_tr), np.max(p_tr))
    dprint("Metrics p-nans:", int(np.isnan(p_va).sum()), "| p[min,med,max]:", np.min(p_va), np.median(p_va), np.max(p_va))
    safe_metrics(p_va, y_val, "val")

    # Calibración
    bin_edges = [-21, -10.5, -6.5, -3.5, -0.5, 0.5, 3.5, 6.5, 10.5, 21]
    cut_bins = lambda x: pd.cut(x, bins=bin_edges, include_lowest=True)
    val_bins  = cut_bins(val_df["home_line"])
    test_bins = cut_bins(test_df["home_line"]) if len(test_df) else pd.Series(dtype="category")

    iso_global = IsotonicRegression(out_of_bounds="clip")
    iso_global.fit(p_va, y_val)

    bin_min = 120
    iso_by_bin = {}
    for b in val_bins.unique():
        idx = (val_bins == b)
        if idx.sum() >= bin_min:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_va[idx], y_val[idx]); iso_by_bin[b] = iso

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

    prior_lr = LogisticRegression(C=2.0, solver="liblinear", max_iter=200)
    prior_lr.fit(train_df[["home_line"]], y_train)
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

    p_te_meta = (np.clip(meta_lr.predict_proba(meta_test)[:,1], 1e-6, 1-1e-6)
                 if len(meta_test) else np.array([]))

    # Preds test/target
    test_df = test_df.copy()
    test_df["week_label"] = test_df["week"].apply(week_label_from_num)
    test_preds = test_df[["season","week","week_label","home_team","away_team"]].copy()
    test_preds["p_home_win_lr_cal"] = p_te_cal
    test_preds["p_home_win_meta"]   = p_te_meta
    return test_preds

# ----------------------------
# EV helpers
# ----------------------------
def sanitize_ev(df: pd.DataFrame, cfg) -> pd.DataFrame:
    need = ["season","week","week_label","schedule_date","side","team","opponent","decimal_odds","model_prob"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"EV table missing columns: {missing}")

    ev = df.copy()
    ev["schedule_date"] = pd.to_datetime(ev["schedule_date"], errors="coerce", utc=True)
    ev["model_prob"] = pd.to_numeric(ev["model_prob"], errors="coerce").astype(float).clip(MIN_PROB, 1-MIN_PROB)
    ev["decimal_odds"] = pd.to_numeric(ev["decimal_odds"], errors="coerce").astype(float)

    # Odds range
    before = len(ev)*2 if False else len(ev)
    dprint("Filtro odds range antes:", len(ev)*2 if False else len(ev))
    ev = ev[ev["decimal_odds"].between(cfg["ODDS_MIN"], cfg["ODDS_MAX"])]
    dprint("Filtro odds range después:", len(ev))

    # Confianza
    dprint("Filtro confianza antes:", len(ev))
    conf = (ev["model_prob"] - 0.5).abs()
    ev = ev[conf >= cfg["CONF_MIN"]]
    dprint("Filtro confianza después:", len(ev))

    # Skip W1 W2
    if cfg["SKIP_W1_W2"]:
        ev = ev[~ev["week_label"].isin(["Week 1","Week 2"])]

    ev = ev.sort_values(["week_label","schedule_date","team","side"]).reset_index(drop=True)
    return ev

def ev_floor(decimal_odds, ev_base_min, ev_slope):
    d = float(decimal_odds)
    return max(ev_base_min, ev_base_min + max(0.0, d - 2.0) * ev_slope)

def ev_slope_for_row(row, cfg):
    d = float(row["decimal_odds"])
    b = cfg["BANDS"]
    if d < b["FAV"]["odds_lt"]: return b["FAV"]["ev_slope"]
    if d < b["MID"]["odds_lt"]: return b["MID"]["ev_slope"]
    return b["DOG"]["ev_slope"]

# ----------------------------
# DEVIG (con LH single-side)
# ----------------------------
def devig_per_game(ev_df: pd.DataFrame, single_side_mode: str, league_delta: float, hold_fallback: float) -> pd.DataFrame:
    ev = ev_df.copy()
    if "market_prob_nv" not in ev.columns:
        ev["market_prob_nv"] = ev["decimal_odds"].apply(implied_from_decimal)

    tmp = ev.copy()
    tmp["p_raw"] = 1.0 / tmp["decimal_odds"].clip(1e-9)
    sides_per_game = tmp.groupby("game_id")["side"].transform("nunique")
    sum_p = tmp.groupby("game_id")["p_raw"].transform("sum")

    # PARES (ambos lados presentes): normalización exacta
    mask_pairs = sides_per_game >= 2
    dprint("DEVIG pares antes:", int(mask_pairs.sum()), "de", len(tmp))
    ev.loc[mask_pairs, "market_prob_nv"] = (tmp.loc[mask_pairs, "p_raw"] / sum_p[mask_pairs]).clip(1e-6, 1-1e-6)

    # SINGLE-SIDE (solo un lado): modo LH
    if single_side_mode.lower() == "lh":
        # Estimar overround O_est por semana a partir de pares
        wk_overround = {}
        for wk, g in tmp[mask_pairs].groupby("week_label"):
            over = (g.groupby("game_id")["p_raw"].sum() - 1.0).clip(lower=0.0)
            if not over.empty:
                wk_overround[wk] = float(np.clip(over.median(), 0.02, 0.08))
        # Aplicar a singles
        singles = ~mask_pairs
        if singles.any():
            rows = ev[singles].index
            for idx in rows:
                r = ev.loc[idx]
                wk  = str(r.get("week_label",""))
                p_r = float(1.0 / max(r["decimal_odds"], 1e-9))
                O   = wk_overround.get(wk, float(np.clip(hold_fallback, 0.02, 0.08)))
                # LH: fair ≈ p_raw /(1 + O/2)  (aprox repartir vig mitad/mithad)
                p_fair = p_r / (1.0 + O/2.0)
                # pequeño shrink hacia 0.5 guiado por liga (autocorrección muy leve)
                shrink = league_delta
                p_adj = (1.0 - abs(shrink)) * p_fair + (abs(shrink)) * 0.5
                ev.at[idx, "market_prob_nv"] = float(np.clip(p_adj, 1e-6, 1-1e-6))
    else:
        # modo legacy "raw": dejar p_raw sin normalización (no recomendado)
        pass

    nan_after = int(ev["market_prob_nv"].isna().sum())
    dprint("DEVIG aplicado; NaNs market_prob_nv:", nan_after)
    return ev

# ----------------------------
# Selección
# ----------------------------
def apply_band_rules(df: pd.DataFrame, cfg) -> pd.DataFrame:
    b = cfg.get("BANDS", {})
    if not b: return df
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
            keep &= (~m2) | ((edge >= b["DOG"]["extra_if_ge3_10"]["tau"]) & (conf >= b["DOG"]["extra_if_ge3_10"]["conf"]))
    return x[keep].reset_index(drop=True)

def pick_per_game_best(ev: pd.DataFrame, cfg, allow_two_sides: bool=False) -> pd.DataFrame:
    # edge filter
    dprint("Filtro edge antes:", len(ev))
    c = ev[ev["edge"] >= cfg["EDGE_TAU"]].copy()
    dprint("Filtro edge después:", len(c))

    # EV floor adaptativo
    dprint("Filtro EV floor antes:", len(c))
    def ok_ev(row):
        slope = ev_slope_for_row(row, cfg)
        floor = ev_floor(row["decimal_odds"], cfg["EV_BASE_MIN"], slope)
        return row["ev"] >= max(floor, cfg["LH_MIN_POS_EV"])  # nunca abajo de un EV>0 mínimo
    c = c[c.apply(ok_ev, axis=1)]
    dprint("Filtro EV floor después:", len(c))

    # Un pick por juego (salvo que se permita 2 lados como último recurso)
    if not allow_two_sides:
        c = (c.sort_values(["week_label","schedule_date","game_id","edge","ev"], ascending=[True,True,True,False,False])
               .drop_duplicates("game_id"))
    else:
        # permitir dos lados si ambos superan filtros (poco común)
        pass

    before = len(c)
    c = apply_band_rules(c, cfg)
    dropped = before - len(c)
    dprint(f"Band rules drop: {dropped} de {before}")
    return c.reset_index(drop=True)

# ---------- Kelly fraccional + caps ----------
def kelly_fraction_scaled(edge, cfg):
    # mantener igual (simple)
    EDGE_REF=0.12
    MIN_SCALE=0.60
    MAX_SCALE=1.00
    scale = np.clip(edge / EDGE_REF, MIN_SCALE, MAX_SCALE)
    return cfg["KELLY_FRACTION"] * float(scale)

def kelly_stake(bk_week0, p, d, edge, cfg):
    b = max(d - 1.0, 1e-9)
    f_star = (b*p - (1-p)) / b
    f_base = max(0.0, f_star)
    f_adj  = kelly_fraction_scaled(edge, cfg)
    stake = bk_week0 * f_base * f_adj
    stake = min(stake, bk_week0*cfg["KELLY_PCT_CAP"], cfg["ABS_STAKE_CAP"])
    return float(np.floor(stake*100)/100.0)

# ----------------------------
# Bankroll (no se exporta)
# ----------------------------
ORDER_MIN = {wl:i for i, wl in enumerate(ORDER_LABELS)}
def build_bk0_for_week(existing: pd.DataFrame, week_label: str, cfg):
    if existing is None or existing.empty or "bankroll_week_final" not in existing.columns:
        return cfg["INITIAL_BANKROLL"]
    ex = existing.copy()
    if "week_label" not in ex.columns and "week" in ex.columns:
        ex["week_label"] = ex["week"].apply(week_label_from_num)
    ex["week_order"] = ex["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    my_ord = ORDER_INDEX.get(week_label, 999)
    ex_prev = ex[ex["week_order"] < my_ord]
    if "bankroll_week_final" in ex_prev.columns and not ex_prev["bankroll_week_final"].dropna().empty:
        return float(ex_prev["bankroll_week_final"].dropna().iloc[-1])
    return cfg["INITIAL_BANKROLL"]

# ----------------------------
# MAIN
# ----------------------------
def main():
    os.makedirs(os.path.dirname(BETS_OUT_PATH), exist_ok=True)

    target_season = getenv_int("TARGET_SEASON", detect_target_season())
    target_week_override = getenv_int_or_none("TARGET_WEEK")
    dprint("Resolved -> target_season:", target_season, "| TARGET_WEEK override:", target_week_override)

    odds_cur_all = load_current_odds()
    odds_cur = odds_cur_all[odds_cur_all["season"].eq(target_season)].copy()

    # Semanas disponibles en odds
    weeks_present = sorted(pd.to_numeric(odds_cur["week"], errors="coerce").dropna().unique().tolist())
    dprint("Semanas presentes en odds (season target):", weeks_present)

    # Detectar semana vigente
    wk_label_auto = detect_vigente_week_label(odds_cur)
    wk_label = week_label_from_num(target_week_override) if target_week_override else wk_label_auto
    dprint("Semana elegida para picks:", wk_label)

    # Filtrar a una sola semana
    odds_cur = odds_cur[odds_cur["week_label"].astype(str).eq(wk_label)].copy()

    # Cargar stats (4 prev + actual) y master
    stats_all = load_pregame_stats_for_seasons(target_season)

    master_target = build_master(odds_cur.copy(), stats_all)
    dprint("Master target rows:", len(master_target))
    have_cols = [c for c in master_target.columns if re.search(r'_(pre(_ewm)?|pre_ytd|pre_l8)$', c)]
    dprint("Audit: columnas pregame presentes:", len(have_cols), "OK")
    miss_h = (master_target[[c for c in master_target.columns if c.startswith("home_") and c.endswith(("_pre_ytd","_pre_ewm","_pre_l8"))]].isna().all(axis=1)).sum()
    miss_a = (master_target[[c for c in master_target.columns if c.startswith("away_") and c.endswith(("_pre_ytd","_pre_ewm","_pre_l8"))]].isna().all(axis=1)).sum()
    dprint("Audit: filas sin pregame (home)=", miss_h, "/ (away)=", miss_a, "de", len(master_target))

    # Hist odds (4 temporadas previas)
    hist_frames = []
    for y in range(target_season-4, target_season):
        p = os.path.join(ARCHIVE_DIR, f"season={y}", "odds.csv")
        if os.path.exists(p):
            tmp = pd.read_csv(p, low_memory=False)
            for c in ("season","week"):
                if c in tmp.columns: tmp[c] = pd.to_numeric(tmp[c], errors="coerce").astype("Int64")
            if "schedule_date" in tmp.columns:
                tmp["schedule_date"] = pd.to_datetime(tmp["schedule_date"], errors="coerce", utc=True)
            for c in ("home_team","away_team"):
                if c in tmp.columns: tmp[c] = tmp[c].astype(str).map(norm_team)
            if "home_win" not in tmp.columns and {"score_home","score_away"}.issubset(tmp.columns):
                tmp["home_win"] = (pd.to_numeric(tmp["score_home"], errors="coerce")
                                  > pd.to_numeric(tmp["score_away"], errors="coerce")).astype("Int64")
            hist_frames.append(tmp)
    master_hist = pd.DataFrame()
    if hist_frames:
        odds_hist = pd.concat(hist_frames, ignore_index=True)
        master_hist = build_master(odds_hist, stats_all)

    master_all = pd.concat([master_hist, master_target], ignore_index=True) if not master_hist.empty else master_target.copy()
    dprint("Master ALL rows (hist+target):", len(master_all))

    # Self-learn (robusto)
    sl = self_learn_adjusters()
    league_delta = float(sl.get("league_calib_delta", 0.0))
    hold_fallback = float(sl.get("hold_fallback", CFG["LH_FALLBACK_HOLD"]))

    # Modelo
    test_preds = run_model(master_all, target_season)

    # Odds + preds solo semana vigente
    cols_key = ["season","week","home_team","away_team"]
    need_cols = cols_key + ["week_label","schedule_date",
                            "decimal_home","decimal_away","ml_home","ml_away",
                            "market_prob_home_nv","market_prob_away_nv","home_line","spread_favorite","over_under_line"]
    merge_df = master_target[[c for c in need_cols if c in master_target.columns]].copy()
    if merge_df.empty:
        raise RuntimeError(f"No hay odds para la semana vigente {wk_label}.")
    dprint("Post-merge odds+pred rows (semana vigente):", len(merge_df))

    if "week_label" not in merge_df.columns:
        merge_df["week_label"] = merge_df["week"].apply(week_label_from_num)

    pred = test_preds[["season","week","home_team","away_team","p_home_win_meta"]].copy()
    dfm  = merge_df.merge(pred, on=cols_key, how="inner")

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

    # Completar ML si falta
    if "ml" in ev_predictions.columns and "decimal_odds" in ev_predictions.columns:
        m = ev_predictions["ml"].isna() & ev_predictions["decimal_odds"].notna()
        ev_predictions.loc[m, "ml"] = ev_predictions.loc[m, "decimal_odds"].apply(decimal_to_american)

    # Sanitizado + DEVIG
    ev_base = sanitize_ev(ev_predictions, CFG)
    dprint(f"EV base rows para {wk_label} tras sanitize:", len(ev_base))
    ev_base["game_id"] = ev_base.apply(lambda r: make_game_id_row(r["week_label"], r["team"], r["opponent"]), axis=1)
    ev_devig = devig_per_game(ev_base.copy(),
                              single_side_mode=CFG["DEVIG_SINGLE_SIDE"],
                              league_delta=league_delta,
                              hold_fallback=hold_fallback)

    # edge (modelo vs mercado) y EV recálculo
    p = ev_devig["model_prob"].astype(float)
    d = ev_devig["decimal_odds"].astype(float)
    mkt = ev_devig["market_prob_nv"].astype(float)
    ev_devig["edge"] = p - mkt
    ev_devig["ev"]   = p*(d-1) - (1-p)

    # === AUDIT DIAGNÓSTICO (siempre activado) =================
    try:
        aud = ev_devig.copy()
        if "schedule_date" in aud.columns:
            aud["schedule_date"] = pd.to_datetime(aud["schedule_date"], errors="coerce", utc=True)
        else:
            aud["schedule_date"] = pd.NaT

        required_cols = ["decimal_odds","model_prob","market_prob_nv","edge","ev","week_label","team","opponent","side","game_id"]
        for c in required_cols:
            if c not in aud.columns:
                aud[c] = np.nan

        aud["conf"] = (aud["model_prob"].astype(float) - 0.5).abs()

        tau         = float(CFG.get("EDGE_TAU", 0.055))
        conf_min    = float(CFG.get("CONF_MIN", 0.085))
        ev_base_min = float(CFG.get("EV_BASE_MIN", 0.010))

        def _ev_floor_row(r):
            return ev_floor(float(r["decimal_odds"]), ev_base_min, ev_slope_for_row(r, CFG))
        aud["EV_floor"] = aud.apply(_ev_floor_row, axis=1)

        aud["fail_edge"] = aud["edge"].astype(float) < tau
        aud["fail_conf"] = aud["conf"].astype(float) < conf_min
        aud["fail_ev"]   = aud["ev"].astype(float)   < np.maximum(aud["EV_floor"].astype(float), CFG["LH_MIN_POS_EV"])

        aud_disp = (aud[[
            "week_label","schedule_date","game_id","side","team","opponent",
            "decimal_odds","model_prob","market_prob_nv","edge","conf","ev","EV_floor",
            "fail_edge","fail_conf","fail_ev"
        ]].sort_values(["week_label","schedule_date","game_id","side"], ascending=[True, True, True, True])
          .reset_index(drop=True))

        print("[AUDIT] Dump de TODOS los juegos (con métricas y flags):")
        with pd.option_context("display.max_rows", 300, "display.max_columns", 60, "display.width", 220):
            print(aud_disp)

        near = aud.copy()
        near["near_edge"] = near["edge"].astype(float) >= (tau * 0.85)
        near["near_conf"] = near["conf"].astype(float) >= (conf_min * 0.85)
        near["near_ev"]   = near["ev"].astype(float)   >= (np.maximum(near["EV_floor"].astype(float), CFG["LH_MIN_POS_EV"]) * 0.85)
        near_miss = near[(near["near_edge"] | near["near_conf"] | near["near_ev"])
                         & (near["fail_edge"] | near["fail_conf"] | near["fail_ev"])]
        near_miss = near_miss.sort_values(
            ["week_label","schedule_date","game_id","edge","ev"],
            ascending=[True, True, True, False, False]
        )
        if not near_miss.empty:
            print("[AUDIT] Near-misses (candidatos que casi pasan) — top 12:")
            with pd.option_context("display.max_rows", 200, "display.max_columns", 60, "display.width", 220):
                print(near_miss[[
                    "week_label","schedule_date","game_id","side","team","opponent",
                    "decimal_odds","model_prob","market_prob_nv","edge","conf","ev","EV_floor",
                    "fail_edge","fail_conf","fail_ev","near_edge","near_conf","near_ev"
                ]].head(12))
        else:
            print("[AUDIT] Near-misses: (ninguno)")

        fails = aud[["fail_edge","fail_conf","fail_ev"]].copy()
        total = len(fails)
        if total > 0:
            n_edge = int(fails["fail_edge"].sum())
            n_conf = int(fails["fail_conf"].sum())
            n_ev   = int(fails["fail_ev"].sum())
            print(f"[AUDIT] Resumen de caídas (sobre {total} lados evaluados): edge {n_edge}, conf {n_conf}, ev {n_ev}")
        else:
            print("[AUDIT] Resumen de caídas: (no hay filas evaluadas)")
    except Exception as e:
        print("[AUDIT] WARN: auditoría no pudo ejecutarse:", repr(e))
    # === FIN AUDIT =============================================

    # Selección base
    picks = pick_per_game_best(ev_devig, CFG, allow_two_sides=False)
    dprint("Candidatos tras pick_per_game_best (base):", len(picks))

    # Relajación escalonada hasta MIN_BETS_PER_WEEK
    if len(picks) < CFG["MIN_BETS_PER_WEEK"]:
        dprint("Relaxation: semana por debajo de mínimo -> aplicando pasos.")
        base_cfg = CFG.copy()
        for step in RELAX_STEPS:
            CFG.update(step)
            dprint(f"Relax step -> EDGE_TAU={CFG['EDGE_TAU']} | CONF_MIN={CFG['CONF_MIN']} | EV_BASE_MIN={CFG['EV_BASE_MIN']}" +
                   (" | allow_two_sides=ON" if step.get("allow_two_sides") else ""))
            # Refiltrar desde ev_devig (no recalc)
            c = pick_per_game_best(ev_devig, CFG, allow_two_sides=step.get("allow_two_sides", False))
            if len(c) >= CFG["MIN_BETS_PER_WEEK"]:
                picks = c
                break
            else:
                # conservar lo mejor si mejora
                if len(c) > len(picks):
                    picks = c
        else:
            dprint("Relaxation: no se alcanzó el mínimo; devolviendo picks disponibles.")

        # restaurar config base para no contaminar próxima corrida
        for k, v in base_cfg.items(): CFG[k] = v

    # Cap al máximo
    if len(picks) > CFG["MAX_BETS_PER_WEEK"]:
        picks = picks.sort_values(["ev","edge"], ascending=[False,False]).head(CFG["MAX_BETS_PER_WEEK"])

    dprint(f"Candidatos finales para {wk_label}:", len(picks))
    if not picks.empty:
        with pd.option_context("display.max_rows", 200, "display.max_columns", 60, "display.width", 220):
            print(picks[["week_label","schedule_date","game_id","side","team","opponent",
                         "decimal_odds","model_prob","market_prob_nv","edge","ev"]])

    # ------------ Stake plan (usa BK0 pero NO exporta bankroll) ------------
    existing = pd.read_csv(BETS_OUT_PATH, low_memory=False) if os.path.exists(BETS_OUT_PATH) else pd.DataFrame()
    if "schedule_date" in existing.columns:
        existing["schedule_date"] = pd.to_datetime(existing["schedule_date"], errors="coerce", utc=True)
    if "week_label" not in existing.columns and "week" in existing.columns:
        existing["week_label"] = existing["week"].apply(week_label_from_num)

    BK0 = build_bk0_for_week(existing, wk_label, CFG)
    dprint(f"BK0 para {wk_label} (solo para caps): {BK0:.2f}")

    # stake
    stakes = []
    for _, r in picks.iterrows():
        st = kelly_stake(BK0, float(r["model_prob"]), float(r["decimal_odds"]), float(r["edge"]), CFG)
        stakes.append(st)
    picks = picks.copy()
    picks["stake"] = np.round(stakes, 2)
    picks["profit"] = np.nan  # se completará cuando termine el juego

    # ----------------------------
    # APPEND/UPSERT: solo semana vigente; nunca tocar semanas pasadas
    # ----------------------------
    # Ordenar columnas amigable
    col_order = [
        "season","week","week_label","schedule_date",
        "side","team","opponent",
        "decimal_odds","ml","market_prob_nv","model_prob",
        "edge","ev","game_id","stake","profit"
    ]
    for c in col_order:
        if c not in picks.columns: picks[c] = pd.NA
    picks = picks[col_order].copy()

    # Lectura existente y upsert solo por (season, week, game_id, side) de la semana vigente
    if existing is not None and not existing.empty:
        for c in col_order:
            if c not in existing.columns: existing[c] = pd.NA

        existing_keys = existing[existing["week_label"].astype(str).eq(wk_label)][["season","week","game_id","side"]]
        new_keys = picks[["season","week","game_id","side"]]
        mask_dupe = pd.merge(new_keys.assign(_k=1), existing_keys.assign(_k=1),
                             on=["season","week","game_id","side"], how="left")["_k_y"].notna()
        # Eliminar los que chocan en semana vigente
        ex = existing[~existing["week_label"].astype(str).eq(wk_label)].copy()
        dups = picks[mask_dupe.values]
        if len(dups) > 0:
            dprint(f"Upsert: se eliminarán {len(dups)} filas existentes de semanas modificables por clave match (upsert).")
        combined = pd.concat([ex, picks], ignore_index=True)
    else:
        combined = picks.copy()

    # Orden estable final
    combined = combined[col_order]
    combined = combined.sort_values(["week_label","schedule_date","game_id","side"]).reset_index(drop=True)

    combined.to_csv(BETS_OUT_PATH, index=False)
    print(f"[select_bets] upsert wrote {BETS_OUT_PATH} | prev_rows={(0 if existing is None else len(existing))} | total={len(combined)} | weeks_upserted=['{wk_label}']")

if __name__ == "__main__":
    main()

