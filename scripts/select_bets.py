#!/usr/bin/env python3
"""
select_bets.py (pregame staking only)

- Entrena el modelo (HGB + calibración + meta LR) igual al notebook.
- Usa stats pregame históricos (4 temporadas previas) + temporada actual.
- Une con odds actuales (data/live/odds.csv) solo para temporada target.
- Calcula EV/edge, aplica estrategia con filtros estrictos y planifica stake pregame
  usando como bankroll base por semana el `bankroll_week_final` de la semana anterior
  (si no existe, cae a INITIAL_BANKROLL). NUNCA usa 'won' ni resultados.
- Exporta apuestas seleccionadas a data/live/bets.csv en modo UPSERT-SEMANA-VIGENTE
  (no toca semanas finalizadas; reescribe solo semana vigente).
"""

import os, re, warnings
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from math import log, exp

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

# --------- Filtros y caps (MISMA LÓGICA BASE) ---------
CFG = dict(
    INITIAL_BANKROLL = 1000.0,

    # Kelly & caps (solo para stake; no se escribe bankroll a CSV)
    KELLY_FRACTION   = 0.25,
    KELLY_PCT_CAP    = 0.05,
    ABS_STAKE_CAP    = 300.0,
    WEEKLY_CAP_PCT   = 0.50,

    # Rango cuotas
    ODDS_MIN         = 1.20,
    ODDS_MAX         = 3.40,

    # Confianza mínima
    CONF_MIN         = 0.090,

    # Umbrales base para edge/EV (se pueden relajar)
    EDGE_TAU         = 0.060,
    EDGE_TAU_MIN     = 0.040,

    # EV floor base + pendiente por bandas
    EV_BASE_MIN      = 0.010,     # (ligeramente menor por adaptativo)
    EV_SLOPE         = 0.012,

    # Semana 1-2 opcionalmente fuera
    SKIP_W1_W2       = True,

    # Devig
    DEVIG_SINGLE_SIDE = "raw",

    # Escala Kelly por edge
    KELLY_EDGE_WEIGHT = dict(EDGE_REF=0.12, MIN_SCALE=0.60, MAX_SCALE=1.00),

    # Límite picks
    MIN_BETS_PER_WEEK = 5,
    MAX_BETS_PER_WEEK = 8,
    MAX_BIG_DOGS_PER_WEEK = 1,

    # Bandas
    BANDS = dict(
        FAV = dict(odds_lt=1.60,                 tau=0.050, conf=0.075, ev_slope=0.010),
        MID = dict(odds_ge=1.60, odds_lt=2.40,   tau=0.055, conf=0.080, ev_slope=0.012),
        DOG = dict(odds_ge=2.40, odds_le=3.40,   tau=0.065, conf=0.095, ev_slope=0.017,
                   extra_if_ge3_10=dict(tau=0.070, conf=0.100))
    ),
)

# Pasos de relajación (ligeramente más agresivos para alcanzar 5-8)
RELAX_STEPS = [
    dict(EDGE_TAU=0.050, CONF_MIN=0.080, EV_BASE_MIN=0.008, allow_two_sides=False),
    dict(EDGE_TAU=0.0475, CONF_MIN=0.075, EV_BASE_MIN=0.006, allow_two_sides=False),
    dict(EDGE_TAU=0.045, CONF_MIN=0.070, EV_BASE_MIN=0.005, allow_two_sides=False),
    dict(EDGE_TAU=0.0425, CONF_MIN=0.067, EV_BASE_MIN=0.0045, allow_two_sides=False),
    dict(EDGE_TAU=0.040, CONF_MIN=0.065, EV_BASE_MIN=0.004, allow_two_sides=False),
    dict(EDGE_TAU=0.0375, CONF_MIN=0.062, EV_BASE_MIN=0.0038, allow_two_sides=False),
    dict(EDGE_TAU=0.035, CONF_MIN=0.060, EV_BASE_MIN=0.0035, allow_two_sides=True),  # último recurso
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
# Self-learn (desde bets.csv)
# ----------------------------
def _safe_logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return log(p / (1.0 - p))

def _safe_sigmoid(z: float) -> float:
    return 1.0 / (1.0 + exp(-z))

def load_self_learn_from_bets(
    bets_path: str,
    current_season: int,
    current_week_label: str,
    lookback_weeks: int = 6,
    min_bets_team: int = 4,
    min_bets_league: int = 20,
    calib_cap: float = 0.02,
    shrink_alpha: float = 6.0
):
    out = dict(league=dict(league_calib_delta=0.0,
                           league_unit_pnl=None,
                           hold_fallback=0.045),
               team_deltas={})

    if not os.path.exists(bets_path):
        dprint(f"WARN self-learn: {bets_path} no existe; sin ajustes.")
        return out

    try:
        b = pd.read_csv(bets_path, low_memory=False)
    except Exception as e:
        dprint("WARN self-learn: no se pudo leer bets.csv:", repr(e))
        return out

    need = {"season","week_label","team","stake","profit"}
    missing = [c for c in need if c not in b.columns]
    if missing:
        dprint(f"WARN self-learn: bets.csv sin columnas {missing}; sin ajustes.")
        return out

    b["season"] = pd.to_numeric(b["season"], errors="coerce").astype("Int64")
    b = b[b["season"].eq(int(current_season))].copy()
    if "week_order" not in b.columns:
        b["week_order"] = b["week_label"].astype(str).map(ORDER_INDEX).fillna(999).astype(int)

    cur_ord = ORDER_INDEX.get(str(current_week_label), 999)
    hist = (b[(b["week_order"] < cur_ord)]
              .dropna(subset=["stake","profit"]))
    if hist.empty:
        dprint("WARN self-learn: no hay historial (semana pasada) en bets.csv; sin ajustes.")
        return out

    low_ord = max(0, cur_ord - lookback_weeks)
    win = hist[hist["week_order"] >= low_ord].copy()

    sum_stake = float(win["stake"].clip(lower=0).sum())
    sum_profit = float(win["profit"].fillna(0).sum())
    if sum_stake > 0 and win.shape[0] >= min_bets_league:
        league_roi = sum_profit / sum_stake
        delta = float(np.clip(league_roi * 0.15, -calib_cap, calib_cap))
        out["league"]["league_calib_delta"] = delta
        out["league"]["league_unit_pnl"] = league_roi
    else:
        out["league"]["league_calib_delta"] = 0.0
        out["league"]["league_unit_pnl"] = None

    g = (win.groupby("team", dropna=False)
            .agg(n=("stake","count"), stake=("stake","sum"), profit=("profit","sum"))
            .reset_index())
    g = g[(g["stake"] > 0) & (g["n"] >= min_bets_team)].copy()
    if not g.empty:
        g["roi"] = g["profit"] / g["stake"]
        g["w"] = g["n"] / (g["n"] + shrink_alpha)
        g["delta"] = (g["roi"] * 0.30).clip(-calib_cap, calib_cap) * g["w"]
        out["team_deltas"] = {str(r["team"]): float(r["delta"]) for _, r in g.iterrows()}

    dprint("Self-learn (bets) -> league:",
           {"league_calib_delta": out["league"]["league_calib_delta"],
            "league_unit_pnl": out["league"]["league_unit_pnl"],
            "hold_fallback": out["league"]["hold_fallback"]})
    dprint("Self-learn (bets) -> teams n:", len(out["team_deltas"]))
    return out

def apply_self_learn_to_probs(df: pd.DataFrame, sl: dict) -> pd.DataFrame:
    out = df.copy()
    if "model_prob" not in out.columns or "team" not in out.columns:
        return out
    league_k = float(sl.get("league", {}).get("league_calib_delta", 0.0))
    team_d = sl.get("team_deltas", {}) or {}
    def _adj_row(r):
        p = float(r["model_prob"])
        k = league_k + float(team_d.get(str(r["team"]), 0.0))
        z = _safe_logit(p) + k
        return float(np.clip(_safe_sigmoid(z), 1e-6, 1-1e-6))
    out["model_prob"] = out.apply(_adj_row, axis=1)
    return out

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
    for y in range(target_season-4, target_season):
        p = os.path.join(ARCHIVE_DIR, f"season={y}", "stats.csv")
        if os.path.exists(p):
            tmp = pd.read_csv(p, low_memory=False)
            tmp["season"] = pd.to_numeric(tmp["season"], errors="coerce").astype("Int64")
            tmp["week"]   = pd.to_numeric(tmp["week"], errors="coerce").astype("Int64")
            tmp["team"]   = tmp["team"].astype(str).map(norm_team)
            frames.append(tmp)
    if os.path.exists(LIVE_STATS_PATH):
        tmp = pd.read_csv(LIVE_STATS_PATH, low_memory=False)
        tmp["season"] = pd.to_numeric(tmp["season"], errors="coerce").astype("Int64")
        tmp["week"]   = pd.to_numeric(tmp["week"], errors="coerce").astype("Int64")
        tmp["team"]   = tmp["team"].astype(str).map(norm_team)
        frames.append(tmp)
    if not frames:
        raise FileNotFoundError("No se encontraron stats pregame para entrenar/pred.")
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
# Modelo + métricas (idéntico en esencia)
# ----------------------------
def run_model(master_all: pd.DataFrame, target_season: int):
    df = master_all.copy()

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

    y_train = train_df["home_win"].dropna().astype(int).values
    y_val   = val_df["home_win"].dropna().astype(int).values

    X_train = train_df[feat_cols].copy()
    X_val   = val_df[feat_cols].copy()
    X_test  = test_df[feat_cols].copy()

    train_meds = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_meds); X_val = X_val.fillna(train_meds); X_test = X_test.fillna(train_meds)

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

    # Preds para temporada target
    test_df = test_df.copy()
    test_df["week_label"] = test_df["week"].apply(week_label_from_num)
    test_preds = test_df[["season","week","week_label","home_team","away_team"]].copy()
    test_preds["p_home_win_lr_cal"] = p_te_cal
    test_preds["p_home_win_meta"]   = p_te_meta

    # --- métricas visibles ---
    try:
        print("Model Results:")
        tr_acc  = accuracy_score(y_train, (p_tr_cal>=0.5).astype(int))
        tr_auc  = roc_auc_score(y_train, p_tr_cal)
        tr_ll   = log_loss(y_train, p_tr_cal, labels=[0,1])
        va_acc  = accuracy_score(y_val, (p_va_cal>=0.5).astype(int))
        va_auc  = roc_auc_score(y_val, p_va_cal)
        va_ll   = log_loss(y_val, p_va_cal, labels=[0,1])
        print(f"train | ACC {tr_acc:.3f} | ROC_AUC {tr_auc:.3f} | LOGLOSS {tr_ll:.3f}")
        print(f"val   | ACC {va_acc:.3f} | ROC_AUC {va_auc:.3f} | LOGLOSS {va_ll:.3f}")
    except Exception:
        print("Model Results:\ntrain | ACC nan | ROC_AUC nan | LOGLOSS nan\nval   | ACC nan | ROC_AUC nan | LOGLOSS nan")

    return test_preds

# ----------------------------
# EV, selección
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

    ev = add_week_order(ev)
    ev["game_id"] = ev.apply(lambda r: make_game_id_row(r["week_label"], r["team"], r["opponent"]), axis=1)

    # Rango de cuotas
    dprint("Filtro odds range antes:", len(ev)*1)
    ev = ev[ev["decimal_odds"].between(cfg["ODDS_MIN"], cfg["ODDS_MAX"])]
    dprint("Filtro odds range después:", len(ev)*1)

    # Confianza mínima
    dprint("Filtro confianza antes:", len(ev)*1)
    ev = ev[np.abs(ev["model_prob"] - 0.5) >= cfg["CONF_MIN"]]
    dprint("Filtro confianza después:", len(ev)*1)

    if cfg["SKIP_W1_W2"]:
        ev = ev[~ev["week_label"].isin(["Week 1","Week 2"])]

    ev = ev.sort_values(["week_order","schedule_date","game_id"]).reset_index(drop=True)
    return ev

def devig_per_game(ev_df: pd.DataFrame, single_side_mode: str) -> pd.DataFrame:
    ev = ev_df.copy()
    if "market_prob_nv" not in ev.columns:
        ev["market_prob_nv"] = ev["decimal_odds"].apply(implied_from_decimal)

    tmp = ev.copy()
    tmp["p_raw"] = 1.0 / tmp["decimal_odds"].clip(1e-9)
    sides_per_game = tmp.groupby("game_id")["side"].transform("nunique")
    sum_p = tmp.groupby("game_id")["p_raw"].transform("sum")
    dprint("DEVIG pares antes:", int((sides_per_game>=2).sum()/2), "de", len(tmp))

    mask_pairs = sides_per_game >= 2
    ev.loc[mask_pairs, "market_prob_nv"] = (tmp.loc[mask_pairs, "p_raw"] / sum_p[mask_pairs]).clip(1e-6, 1-1e-6)

    ev["market_prob_nv"] = ev["market_prob_nv"].clip(1e-6, 1-1e-6)
    dprint("DEVIG aplicado; NaNs market_prob_nv:", int(ev["market_prob_nv"].isna().sum()))
    return ev

def ev_floor(decimal_odds, ev_base_min, ev_slope):
    d = float(decimal_odds)
    return ev_base_min + max(0.0, d - 2.0) * ev_slope

def ev_slope_for_row(row, cfg):
    d = float(row["decimal_odds"])
    b = cfg["BANDS"]
    if d < b["FAV"]["odds_lt"]: return b["FAV"]["ev_slope"]
    if d < b["MID"]["odds_lt"]: return b["MID"]["ev_slope"]
    return b["DOG"]["ev_slope"]

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

def pick_per_game_best(ev: pd.DataFrame, cfg, allow_two_sides: bool) -> pd.DataFrame:
    tau = max(cfg["EDGE_TAU"], cfg["EDGE_TAU_MIN"])

    def ok_ev(row):
        slope = ev_slope_for_row(row, cfg)
        floor = ev_floor(row["decimal_odds"], cfg["EV_BASE_MIN"], slope)
        return row["ev"] >= floor

    c = ev.copy()
    # edge
    dprint("Filtro edge antes:", len(c))
    c = c[c["edge"] >= tau]
    dprint("Filtro edge después:", len(c))
    # EV floor
    dprint("Filtro EV floor antes:", len(c))
    c = c[c.apply(ok_ev, axis=1)]
    dprint("Filtro EV floor después:", len(c))

    # un solo lado por juego, salvo que allow_two_sides=True
    if not allow_two_sides and not c.empty:
        c = (c.sort_values(["week_order","schedule_date","game_id","edge","ev"], ascending=[True,True,True,False,False])
               .drop_duplicates("game_id"))

    pre = len(c)
    c = apply_band_rules(c, cfg)
    dropped = pre - len(c)
    dprint(f"Band rules drop: {dropped} de {pre}")

    # caps por semana
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
    return c.reset_index(drop=True)

# -------- Kelly fraccional + caps ----------
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

# ----------------------------
# UPSERT SEMANA VIGENTE (no tocar semanas cerradas)
# ----------------------------
APPEND_KEYS = ["season","week","game_id","side"]

def upsert_current_week(existing: pd.DataFrame, new_rows: pd.DataFrame, week_label: str) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new_rows.copy()

    ex = existing.copy()
    if "week_label" not in ex.columns and "week" in ex.columns:
        ex["week_label"] = ex["week"].apply(week_label_from_num)

    # separar filas semana vigente y otras
    ex_other = ex[~ex["week_label"].astype(str).eq(week_label)].copy()
    ex_cur   = ex[ex["week_label"].astype(str).eq(week_label)].copy()

    # claves existentes (solo semana vigente)
    have = set(map(tuple, ex_cur[APPEND_KEYS].astype(object).to_numpy().tolist())) if not ex_cur.empty else set()
    new_key_list = list(map(tuple, new_rows[APPEND_KEYS].astype(object).to_numpy().tolist()))
    mask_new = ~pd.Series(new_key_list).isin(have)
    only_new = new_rows[mask_new.values].copy()

    # sobrescribir (upsert) filas que hagan match de clave
    merged = pd.concat([
        ex_other,
        ex_cur[~ex_cur[APPEND_KEYS].astype(object).apply(tuple, axis=1).isin(new_key_list)],
        new_rows
    ], ignore_index=True)

    return merged

# ----------------------------
# Detección de semana vigente
# ----------------------------
def detect_current_week_label(odds_cur: pd.DataFrame) -> str:
    # lista de semanas presentes
    weeks_present = sorted(pd.to_numeric(odds_cur["week"], errors="coerce").dropna().astype(int).unique().tolist())
    dprint("Semanas presentes en odds (season target):", weeks_present)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=18)

    # próximos/vigentes por fecha
    oc = odds_cur.copy()
    oc["schedule_date"] = pd.to_datetime(oc["schedule_date"], errors="coerce", utc=True)
    oc = oc[oc["schedule_date"] >= cutoff]
    if oc.empty:
        # fallback: última semana con odds en archivo
        wk = max(weeks_present) if weeks_present else None
    else:
        wk = int(oc["week"].min())

    if wk is None:
        raise RuntimeError("No hay semanas con odds para determinar semana vigente.")

    wk_label = week_label_from_num(wk)
    dprint("Semana vigente detectada por fechas (>= now-18h):", wk_label)
    return wk_label

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(os.path.dirname(BETS_OUT_PATH), exist_ok=True)

    target_season = getenv_int("TARGET_SEASON", detect_target_season())
    target_week_override = getenv_int_or_none("TARGET_WEEK")
    dprint("Resolved -> target_season:", target_season, "| TARGET_WEEK override:", target_week_override)

    # Carga
    odds_all = load_current_odds()
    stats_all = load_pregame_stats_for_seasons(target_season)

    # Odds solo de temporada target (+ semana si override)
    odds_cur = odds_all[odds_all["season"].eq(target_season)].copy()
    wk_label = None
    if target_week_override is not None:
        wk_label = week_label_from_num(target_week_override)
    else:
        wk_label = detect_current_week_label(odds_cur)
    dprint("Semana elegida para picks:", wk_label)

    if target_week_override is not None:
        odds_cur = odds_cur[odds_cur["week"].eq(target_week_override)].copy()
    else:
        # solo la semana vigente
        odds_cur = odds_cur[odds_cur["week_label"].astype(str).eq(wk_label)].copy()

    dprint("Master target rows:", len(odds_cur))
    # Ensamble master para target y para histórico
    master_target = build_master(odds_cur, stats_all)

    # auditoría mínima de columnas pregame
    pre_cols = [c for c in master_target.columns if re.search(r'(?:_pre(_ewm)?|_pre_ytd|_pre_l8)$', c)]
    dprint("Audit: columnas pregame presentes:", len(pre_cols), "OK")
    miss_home = master_target[[c for c in master_target.columns if c.startswith("home_")]].isna().all(axis=1).sum()
    miss_away = master_target[[c for c in master_target.columns if c.startswith("away_")]].isna().all(axis=1).sum()
    dprint(f"Audit: filas sin pregame (home)={miss_home} / (away)={miss_away} de {len(master_target)}")

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
    master_hist = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
    master_all = pd.concat([master_hist, master_target], ignore_index=True) if not master_hist.empty else master_target.copy()
    dprint("Master ALL rows (hist+target):", len(master_all))

    # Modelo
    test_preds = run_model(master_all, target_season)

    # Unir probs con odds target SOLO semana vigente
    cols_key = ["season","week","home_team","away_team"]
    need_cols = cols_key + ["week_label","schedule_date",
                            "decimal_home","decimal_away","ml_home","ml_away",
                            "market_prob_home_nv","market_prob_away_nv","home_line","spread_favorite","over_under_line"]
    merge_df = master_target[[c for c in need_cols if c in master_target.columns]].copy()
    if "week_label" not in merge_df.columns:
        merge_df["week_label"] = merge_df["week"].apply(week_label_from_num)

    pred = test_preds[["season","week","home_team","away_team","p_home_win_meta"]].copy()
    dfm  = merge_df.merge(pred, on=cols_key, how="inner")

    # EV home/away (semana vigente)
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

    # === SELF-LEARN APLICADO A PROBAS (después de construir ev_predictions) ===
    SL = load_self_learn_from_bets(
        bets_path=BETS_OUT_PATH,
        current_season=target_season,
        current_week_label=wk_label,
        lookback_weeks=6, min_bets_team=4, min_bets_league=20,
        calib_cap=0.02, shrink_alpha=6.0
    )
    ev_predictions = apply_self_learn_to_probs(ev_predictions, SL)

    # Log opcional de ejemplo del self-learn
    try:
        _dbg_show = (ev_predictions[["team","side","decimal_odds","model_prob"]]
                     .copy().sort_values("team").head(8))
        print("[DBG] Self-learn sample (team, side, decimal_odds, model_prob adj):")
        with pd.option_context("display.max_rows", 20, "display.width", 160):
            print(_dbg_show)
    except Exception as e:
        dprint("WARN: dbg self-learn sample:", repr(e))

    # Calcular EV/edge
    ev_predictions = ev_predictions[pd.to_numeric(ev_predictions["decimal_odds"], errors="coerce").notna()].copy()
    ev_predictions["decimal_odds"] = ev_predictions["decimal_odds"].astype(float)
    ev_predictions["ev"]   = ev_predictions["model_prob"]*(ev_predictions["decimal_odds"]-1) - (1-ev_predictions["model_prob"])
    ev_predictions = ev_predictions.sort_values(["schedule_date","week","team","side"]).reset_index(drop=True)

    # Sanitizado y DEVIG (solo semana vigente)
    ev_base = sanitize_ev(ev_predictions, CFG)
    dprint(f"EV base rows para {wk_label} tras sanitize:", len(ev_base))
    ev_base["game_id"] = ev_base.apply(lambda r: make_game_id_row(r["week_label"], r["team"], r["opponent"]), axis=1)
    ev_devig = devig_per_game(ev_base.copy(), single_side_mode=CFG["DEVIG_SINGLE_SIDE"])

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

        tau         = float(CFG.get("EDGE_TAU", 0.06))
        conf_min    = float(CFG.get("CONF_MIN", 0.09))
        ev_base_min = float(CFG.get("EV_BASE_MIN", 0.010))

        def _ev_floor_row(r):
            return ev_floor(float(r["decimal_odds"]), ev_base_min, ev_slope_for_row(r, CFG))
        aud["EV_floor"] = aud.apply(_ev_floor_row, axis=1)

        aud_disp = (aud[[
            "week_label","schedule_date","game_id","side","team","opponent",
            "decimal_odds","model_prob","market_prob_nv","edge","conf","ev","EV_floor",
            "fail_edge","fail_conf","fail_ev"
        ]].assign(fail_edge=lambda x: x["edge"]<tau,
                  fail_conf=lambda x: x["conf"]<conf_min,
                  fail_ev=lambda x: x["ev"]<x["EV_floor"])
          .sort_values(["week_label","schedule_date","game_id","side"])
          .reset_index(drop=True))

        print("[AUDIT] Dump de TODOS los juegos (con métricas y flags):")
        with pd.option_context("display.max_rows", 300, "display.max_columns", 50, "display.width", 200):
            print(aud_disp)

        near = aud.copy()
        near["conf"] = (near["model_prob"].astype(float) - 0.5).abs()
        near["EV_floor"] = near.apply(_ev_floor_row, axis=1)
        near["fail_edge"] = near["edge"].astype(float) < tau
        near["fail_conf"] = near["conf"].astype(float) < conf_min
        near["fail_ev"]   = near["ev"].astype(float)   < near["EV_floor"].astype(float)
        near["near_edge"] = near["edge"].astype(float) >= (tau * 0.85)
        near["near_conf"] = near["conf"].astype(float) >= (conf_min * 0.85)
        near["near_ev"]   = near["ev"].astype(float)   >= (near["EV_floor"].astype(float) * 0.85)
        near_miss = near[(near["near_edge"] | near["near_conf"] | near["near_ev"])
                         & (near["fail_edge"] | near["fail_conf"] | near["fail_ev"])]
        near_miss = near_miss.sort_values(["week_label","schedule_date","game_id","edge","ev"],
                                          ascending=[True,True,True,False,False])
        if not near_miss.empty:
            print("[AUDIT] Near-misses (candidatos que casi pasan) — top 12:")
            with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 200):
                print(near_miss[[
                    "week_label","schedule_date","game_id","side","team","opponent",
                    "decimal_odds","model_prob","market_prob_nv","edge","conf","ev","EV_floor",
                    "fail_edge","fail_conf","fail_ev","near_edge","near_conf","near_ev"
                ]].head(12))
        else:
            print("[AUDIT] Near-misses: (ninguno)")

        fails = aud[["edge","model_prob","decimal_odds"]].copy()
        total = len(aud)
        fe = int((aud["edge"] < tau).sum())
        fc = int(((aud["model_prob"]-0.5).abs() < conf_min).sum())
        aud_ev_floor = aud.apply(_ev_floor_row, axis=1)
        fv = int((aud["ev"] < aud_ev_floor).sum())
        print(f"[AUDIT] Resumen de caídas (sobre {total} lados evaluados): edge {fe}, conf {fc}, ev {fv}")
    except Exception as e:
        print("[AUDIT] WARN: auditoría no pudo ejecutarse:", repr(e))
    # === FIN AUDIT =============================================

    # Selección base
    picks = pick_per_game_best(ev_devig, CFG, allow_two_sides=False)
    dprint("Candidatos tras pick_per_game_best (base):", len(picks))

    # Relajación escalonada hasta mínimo
    if len(picks) < CFG["MIN_BETS_PER_WEEK"]:
        dprint("Relaxation: semana por debajo de mínimo -> aplicando pasos.")
        cur_cfg = CFG.copy()
        for step in RELAX_STEPS:
            cur_cfg.update(step)
            dprint(f"Relax step -> EDGE_TAU={cur_cfg['EDGE_TAU']:.04f} | CONF_MIN={cur_cfg['CONF_MIN']:.03f} | EV_BASE_MIN={cur_cfg['EV_BASE_MIN']:.03f}"
                   + (" | allow_two_sides=ON" if step.get("allow_two_sides") else ""))
            # re-filtrar con cfg relajado
            picks_relax = pick_per_game_best(ev_devig, cur_cfg, allow_two_sides=step.get("allow_two_sides", False))
            if len(picks_relax) >= CFG["MIN_BETS_PER_WEEK"]:
                picks = picks_relax
                break
        if len(picks) < CFG["MIN_BETS_PER_WEEK"]:
            dprint("Relaxation: no se alcanzó el mínimo; devolviendo picks disponibles.")

    # Cap a 8 máximo (ya lo hace pick_per_game_best, pero por si llegamos por allow_two_sides)
    if len(picks) > CFG["MAX_BETS_PER_WEEK"]:
        picks = picks.sort_values(["ev","edge"], ascending=[False,False]).head(CFG["MAX_BETS_PER_WEEK"])

    dprint(f"Candidatos finales para {wk_label}: {len(picks)}")
    if not picks.empty:
        with pd.option_context("display.max_rows", 60, "display.width", 180):
            print(picks[["week_label","schedule_date","game_id","side","team","opponent",
                         "decimal_odds","model_prob","market_prob_nv","edge","ev"]])

    # Plan stakes (usa BK0, pero no escribimos columna bankroll)
    BK0 = CFG["INITIAL_BANKROLL"]
    dprint(f"BK0 para {wk_label} (solo para caps): {BK0:.2f}")
    stakes = []
    for _, r in picks.iterrows():
        st = kelly_stake(BK0, float(r["model_prob"]), float(r["decimal_odds"]), float(r["edge"]), CFG)
        stakes.append(st)
    picks = picks.copy()
    picks["stake"] = np.round(stakes, 2)
    picks["profit"] = np.nan  # a completar post-juego por tu pipeline de settle

    # Orden de columnas de salida (limpio/legible, SIN bankroll)
    out_cols = [
        "season","week","week_label","schedule_date",
        "side","team","opponent",
        "decimal_odds","ml","market_prob_nv","model_prob",
        "ev","edge",
        "week_order","game_id",
        "stake","profit"
    ]
    for c in out_cols:
        if c not in picks.columns: picks[c] = pd.NA
    picks = picks[out_cols]

    # UPSERT solo semana vigente
    if os.path.exists(BETS_OUT_PATH):
        try:
            existing = pd.read_csv(BETS_OUT_PATH, low_memory=False)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    prev_rows = 0 if existing is None or existing.empty else len(existing)
    if existing is not None and not existing.empty and "week_label" not in existing.columns and "week" in existing.columns:
        existing["week_label"] = existing["week"].apply(week_label_from_num)

    # upsert
    merged = upsert_current_week(existing, picks, wk_label)
    # log removals por clave
    if existing is not None and not existing.empty:
        ex_cur = existing[existing["week_label"].astype(str).eq(wk_label)]
        new_keys = set(map(tuple, picks[APPEND_KEYS].astype(object).to_numpy().tolist()))
        removed = ex_cur[~ex_cur[APPEND_KEYS].astype(object).apply(tuple, axis=1).isin(new_keys)]
        if not removed.empty:
            dprint(f"Upsert: se eliminarán {len(removed)} filas existentes de semanas modificables por clave match (upsert).")

    merged.to_csv(BETS_OUT_PATH, index=False)
    print(f"[select_bets] upsert wrote {BETS_OUT_PATH} | prev_rows={prev_rows} | total={len(merged)} | weeks_upserted=['{wk_label}']")

if __name__ == "__main__":
    main()

