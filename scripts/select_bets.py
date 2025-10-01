#!/usr/bin/env python3
"""
select_bets.py — SOLO semana vigente (upcoming), upsert-safe, sin tocar semanas pasadas.

• Entrena HGB + calibración + meta LR (igual base anterior).
• Usa stats pregame históricas (hasta la semana previa) + temporada actual.
• Une con odds de la semana vigente (no pasadas, no futuras).
• Filtros con bandas + umbrales; relajación hasta alcanzar un mínimo de picks/semana.
• Staking (Kelly fraccional + caps) usando BK0 semanal (de weeks previas), PERO NO se escribe
  ninguna columna de bankroll en el CSV (solo stake y datos de la apuesta).
• Escribe en data/live/bets.csv por “upsert”: solo reemplaza la semana vigente. Semanas pasadas
  quedan congeladas (no se tocan).
• Logs “rayos X”: métricas, Brier + bins, funnel, importancias, distros, near-misses y backtest
  de política en el año de validación.
"""

import os, re, warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ----------------------------
# Config & paths
# ----------------------------
ET = ZoneInfo("America/New_York")
LIVE_ODDS_PATH  = os.path.join("data", "live", "odds.csv")
LIVE_STATS_PATH = os.path.join("data", "live", "stats.csv")
ARCHIVE_DIR     = os.path.join("data", "archive")
BETS_OUT_PATH   = os.path.join("data", "live", "bets.csv")

TEAM_FIX = {
    "STL":"LA","LAR":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV","WSH":"WAS","JAC":"JAX",
    "GNB":"GB","KAN":"KC","NWE":"NE","NOR":"NO","SFO":"SF","TAM":"TB"
}

ORDER_LABELS = [f"Week {i}" for i in range(1,19)] + ["Wild Card","Divisional","Conference","Super Bowl"]
ORDER_INDEX  = {lab:i for i,lab in enumerate(ORDER_LABELS)}
MIN_PROB = 1e-6

CFG = dict(
    INITIAL_BANKROLL = 1000.0,

    # Kelly & caps (interno; NO se escribe bankroll)
    KELLY_FRACTION   = 0.25,
    KELLY_PCT_CAP    = 0.05,
    ABS_STAKE_CAP    = 300.0,
    WEEKLY_CAP_PCT   = 0.50,

    # Filtros base
    ODDS_MIN         = 1.20,
    ODDS_MAX         = 3.40,
    CONF_MIN         = 0.090,
    EDGE_TAU         = 0.060,
    EDGE_TAU_MIN     = 0.040,
    EV_BASE_MIN      = 0.012,
    EV_SLOPE         = 0.012,
    SKIP_W1_W2       = True,

    # De-vig
    DEVIG_SINGLE_SIDE = "raw",

    # Kelly scaling por edge
    KELLY_EDGE_WEIGHT = dict(EDGE_REF=0.12, MIN_SCALE=0.60, MAX_SCALE=1.00),

    # Límites picks
    MAX_BETS_PER_WEEK = 8,      # techo “suave”
    MAX_BIG_DOGS_PER_WEEK = 1,

    # Mínimo por semana (relajación)
    MIN_PICKS_PER_WEEK = 5,

    # Pasos de relajación escalonada
    RELAX_STEPS = [
        dict(EDGE_TAU=0.050, CONF_MIN=0.080, EV_BASE_MIN=0.008),
        dict(EDGE_TAU=0.045, CONF_MIN=0.075, EV_BASE_MIN=0.006),
        dict(EDGE_TAU=0.040, CONF_MIN=0.070, EV_BASE_MIN=0.004),
    ],
)

DEBUG = True
def dprint(*args):
    if DEBUG:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[DBG {ts}]", *args, flush=True)

# ----------------------------
# Semanas (corte Mié 02:00 ET)
# ----------------------------
def _after_cutover_et(dt_utc: datetime) -> bool:
    now_et = dt_utc.astimezone(ET)
    # Mié >= 02:00 ET => AFTER cutoff
    if now_et.weekday() > 2:  # Thu..Sun
        return True
    if now_et.weekday() < 2:  # Mon/Tue
        return False
    return now_et.hour >= 2

def _labor_day_et(year: int) -> datetime:
    d = datetime(year, 9, 1, tzinfo=ET)
    while d.weekday() != 0:
        d += timedelta(days=1)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def _week1_tnf_et(year: int) -> datetime:
    labor = _labor_day_et(year)
    thu   = labor + timedelta(days=3)
    return thu.replace(hour=20, minute=20)

def _tuesday_anchor_et(year: int) -> datetime:
    tnf = _week1_tnf_et(year)
    week_monday = (tnf - timedelta(days=tnf.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return week_monday + timedelta(days=1)  # Tue 00:00 ET

def _calendar_week_nat(season: int, now_utc: datetime) -> int:
    now_et = now_utc.astimezone(ET)
    anchor = _tuesday_anchor_et(season)
    if now_et < anchor:
        wk = 1
    else:
        wk = int(((now_et - anchor).days // 7) + 1)
    return max(1, min(22, wk))

def effective_week_completed(season: int, now_utc: datetime | None = None) -> int:
    """
    Semana “efectiva” COMPLETADA: hasta Mié 01:59 ET la semana efectiva es la anterior.
    A partir de Mié 02:00 ET se considera ya completada la última jornada.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    wk_nat = _calendar_week_nat(season, now_utc)
    return wk_nat if _after_cutover_et(now_utc) else max(1, wk_nat - 1)

def current_upcoming_week(season: int, now_utc: datetime | None = None) -> int:
    """Semana de apuestas vigente (la próxima por jugar)."""
    return min(18, effective_week_completed(season, now_utc) + 1)

def week_label_from_num(n:int) -> str:
    return f"Week {n}" if 1<=n<=18 else {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")

def add_week_order(df):
    out = df.copy()
    out["week_label"] = out["week_label"].astype(str)
    out["week_order"] = out["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    return out

# ----------------------------
# Utilidades varias
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
    frames, seasons_seen = [], []
    for y in range(target_season-4, target_season):
        p = os.path.join(ARCHIVE_DIR, f"season={y}", "stats.csv")
        if os.path.exists(p):
            tmp = pd.read_csv(p, low_memory=False)
            tmp["season"] = pd.to_numeric(tmp["season"], errors="coerce").astype("Int64")
            tmp["week"]   = pd.to_numeric(tmp["week"], errors="coerce").astype("Int64")
            tmp["team"]   = tmp["team"].astype(str).map(norm_team)
            frames.append(tmp); seasons_seen.append(y)
    if os.path.exists(LIVE_STATS_PATH):
        tmp = pd.read_csv(LIVE_STATS_PATH, low_memory=False)
        tmp["season"] = pd.to_numeric(tmp["season"], errors="coerce").astype("Int64")
        tmp["week"]   = pd.to_numeric(tmp["week"], errors="coerce").astype("Int64")
        tmp["team"]   = tmp["team"].astype(str).map(norm_team)
        frames.append(tmp)
        seasons_seen.append(int(pd.to_numeric(tmp["season"], errors="coerce").dropna().max()))
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
# Modelo + métricas
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

    # Probabilidades
    p_tr = hgb.predict_proba(X_train)[:,1]
    p_va = hgb.predict_proba(X_val)[:,1]
    p_te = hgb.predict_proba(X_test)[:,1] if len(X_test) else np.array([])

    # Calibración global + por bins de spread
    iso_global = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_val)

    bin_edges = [-21, -10.5, -6.5, -3.5, -0.5, 0.5, 3.5, 6.5, 10.5, 21]
    cut_bins = lambda x: pd.cut(x, bins=bin_edges, include_lowest=True)
    val_bins  = cut_bins(val_df["home_line"])
    test_bins = cut_bins(test_df["home_line"]) if len(test_df) else pd.Series(dtype="category")

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

    # Meta LR (spread prior)
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

    p_tr_meta = np.clip(meta_lr.predict_proba(meta_tr)[:,1], 1e-6, 1-1e-6)
    p_va_meta = np.clip(meta_lr.predict_proba(meta_val)[:,1], 1e-6, 1-1e-6)
    p_te_meta = (np.clip(meta_lr.predict_proba(meta_test)[:,1], 1e-6, 1-1e-6)
                 if len(meta_test) else np.array([]))

    # -------- Logs de performance (train/val/test*)
    def print_calibration(name, y_true, p):
        if len(y_true)==0: 
            print(f"{name} calib: n/a"); return
        brier = brier_score_loss(y_true, p)
        prob_true, prob_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        acc   = accuracy_score(y_true, (p>=0.5).astype(int))
        auc   = roc_auc_score(y_true, p)
        ll    = log_loss(y_true, np.vstack([1-p, p]).T, labels=[0,1])
        print(f"{name} | ACC {acc:.3f} | ROC_AUC {auc:.3f} | LOGLOSS {ll:.3f} | BRIER {brier:.3f}")
        for qt, qp in zip(prob_true, prob_pred):
            print(f"  bin: pred~{qp:.3f} → emp~{qt:.3f}")

    print("Model Results:")
    print_calibration("train", y_train, p_tr_meta)
    print_calibration("val  ", y_val,   p_va_meta)

    if len(test_df.dropna(subset=['home_win'])):
        mask_fin = test_df['home_win'].notna().values
        print_calibration("test*", test_df['home_win'].dropna().astype(int).values, p_te_meta[mask_fin])
    else:
        print("test* | sin labels finales aún (solo métricas cuando haya resultados).")

    # Feature importances
    try:
        fi = pd.Series(hgb.feature_importances_, index=feat_cols).sort_values(ascending=False)
        print("[Top 20 features]")
        print(fi.head(20).round(4))
    except Exception:
        pass

    # Preds para temporada target
    test_df = test_df.copy()
    test_df["week_label"] = test_df["week"].apply(week_label_from_num)
    test_preds = test_df[["season","week","week_label","home_team","away_team"]].copy()
    test_preds["p_home_win_meta"]   = p_te_meta
    return test_preds

# ----------------------------
# EV, filtros, bandas
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

    # Filtros base
    before = len(ev)
    ev = ev[ev["decimal_odds"].between(cfg["ODDS_MIN"], cfg["ODDS_MAX"])]
    ev = ev[np.abs(ev["model_prob"] - 0.5) >= cfg["CONF_MIN"]]
    if cfg["SKIP_W1_W2"]:
        ev = ev[~ev["week_label"].isin(["Week 1","Week 2"])]
    ev = ev.sort_values(["week_order","schedule_date","game_id"]).reset_index(drop=True)

    print(f"[Funnel] lados in={before} → after={len(ev)} (odds/conf/W1-2)")
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
    # bandas
    if d < 1.60: return 0.010
    if d < 2.40: return 0.012
    return 0.017

def apply_band_rules(df: pd.DataFrame, cfg) -> pd.DataFrame:
    x = df.copy()
    d  = x["decimal_odds"].astype(float)
    conf = np.abs(x["model_prob"] - 0.5)
    edge = x["edge"].astype(float)
    keep = pd.Series(True, index=x.index)

    # FAV
    m = d < 1.60; keep &= (~m) | ((edge >= 0.050) & (conf >= 0.075))
    # MID
    m = (d >= 1.60) & (d < 2.40); keep &= (~m) | ((edge >= 0.055) & (conf >= 0.080))
    # DOG
    m = (d >= 2.40) & (d <= 3.40); keep &= (~m) | ((edge >= 0.065) & (conf >= 0.095))
    m2 = d >= 3.10; keep &= (~m2) | ((edge >= 0.070) & (conf >= 0.100))
    return x[keep].reset_index(drop=True)

def pick_per_game_best(ev: pd.DataFrame, cfg) -> pd.DataFrame:
    tau = max(cfg["EDGE_TAU"], cfg["EDGE_TAU_MIN"])
    def ok_ev(row):
        slope = ev_slope_for_row(row, cfg)
        return row["ev"] >= ev_floor(row["decimal_odds"], cfg["EV_BASE_MIN"], slope)
    c = ev.copy()
    c = c[c["edge"] >= tau]
    c = c[c.apply(ok_ev, axis=1)]
    c = (c.sort_values(["week_order","schedule_date","game_id","edge"], ascending=[True,True,True,False])
           .drop_duplicates("game_id"))
    c = apply_band_rules(c, cfg)

    # Cap por banda
    out = []
    for wk, g in c.groupby("week_label", sort=False):
        # limitar big dogs
        m_big = g["decimal_odds"] >= 3.10
        if m_big.sum() > CFG["MAX_BIG_DOGS_PER_WEEK"]:
            g_big  = g[m_big].sort_values("edge", ascending=False).head(CFG["MAX_BIG_DOGS_PER_WEEK"])
            g_rest = g[~m_big]
            g = pd.concat([g_rest, g_big], ignore_index=True)
        # techo suave total
        g = g.sort_values(["ev","edge"], ascending=[False,False]).head(CFG["MAX_BETS_PER_WEEK"])
        out.append(g)
    c = pd.concat(out, ignore_index=True) if out else c
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
# SELF-LEARN (ajustes liga/equipo)
# ----------------------------
def load_team_league_adjusters(target_season: int):
    """
    Usa bets.csv histórico para medir unit PnL por equipo y liga en la temporada target
    (solo semanas FINALIZADAS) y genera pequeños desplazamientos de calibración.
    """
    try:
        if not os.path.exists(BETS_OUT_PATH):
            return pd.DataFrame(), dict(league_calib_delta=0.0, league_unit_pnl=0.0)
        b = pd.read_csv(BETS_OUT_PATH, low_memory=False)
        # Solo filas con resultado final esta temporada
        b = b[pd.to_numeric(b.get("season"), errors="coerce").eq(target_season)]
        if "status_short" in b.columns:
            b = b[b["status_short"].astype(str).str.upper().eq("FINAL")]
        if b.empty: 
            return pd.DataFrame(), dict(league_calib_delta=0.0, league_unit_pnl=0.0)
        # unit pnl
        b["unit_pnl"] = np.where(
            b["result"].astype(str).str.upper().eq("WIN"),
            (pd.to_numeric(b["decimal_odds"], errors="coerce")-1.0).fillna(0.0),
            np.where(b["result"].astype(str).str.upper().eq("LOSS"), -1.0, 0.0)
        )
        team_adj = (b.groupby("team", as_index=False)
                      .agg(unit_pnl=("unit_pnl","mean"),
                           n=("unit_pnl","size")))
        # desplazamiento liga (pequeño)
        lg = dict(
            league_calib_delta = float(np.clip(team_adj["unit_pnl"].mean() * 0.02, -0.03, 0.03)),
            league_unit_pnl    = float(team_adj["unit_pnl"].mean())
        )
        return team_adj, lg
    except Exception:
        return pd.DataFrame(), dict(league_calib_delta=0.0, league_unit_pnl=0.0)

def apply_self_learn(p: pd.Series, team_series: pd.Series, team_adj_df: pd.DataFrame, league_adj: dict) -> pd.Series:
    """
    Ajuste suave: desplaza p ligeramente según team/unit_pnl y pequeño sesgo de liga.
    """
    out = p.astype(float).copy()
    if team_adj_df is None or team_adj_df.empty: 
        return np.clip(out + league_adj.get("league_calib_delta",0.0), 1e-6, 1-1e-6)
    mp = team_adj_df.set_index("team")["unit_pnl"].to_dict()
    delta_team = team_series.map(mp).fillna(0.0).astype(float) * 0.02  # 2% por unidad PnL
    out = out + delta_team.values + league_adj.get("league_calib_delta",0.0)
    return np.clip(out, 1e-6, 1-1e-6)

# ----------------------------
# APPEND/UPSERT SOLO SEMANA VIGENTE
# ----------------------------
APPEND_KEYS = ["season","week","game_id","side"]

def week_is_final_mask(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(False, index=df.index)
    if "week_is_final" in df.columns:
        s = s | df["week_is_final"].astype(str).str.lower().isin(["true","1","yes"])
    if "status_short" in df.columns:
        s = s | df["status_short"].astype(str).str.upper().eq("FINAL")
    if "status" in df.columns:
        s = s | df["status"].astype(str).str.upper().eq("FINAL")
    return s

def upsert_only_week(existing: pd.DataFrame, new_rows: pd.DataFrame, target_week_label: str) -> pd.DataFrame:
    """
    Congela semanas pasadas. Elimina y reemplaza SOLO filas de target_week_label.
    """
    ex = existing.copy()
    if not ex.empty and "week_label" not in ex.columns and "week" in ex.columns:
        ex["week_label"] = ex["week"].apply(week_label_from_num)

    # Filas de la semana objetivo -> eliminar
    to_keep = ex[ex["week_label"].astype(str) != str(target_week_label)].copy() if not ex.empty else ex

    # Alinear columnas (no agregar bankroll en salida)
    cols_out = list(set(to_keep.columns.tolist()) | set(new_rows.columns.tolist()))
    # Nunca incluir 'bankroll' si estuviera en alguna versión previa
    cols_out = [c for c in cols_out if c != "bankroll"]

    for c in cols_out:
        if c not in to_keep.columns:  to_keep[c]  = pd.NA
        if c not in new_rows.columns: new_rows[c] = pd.NA

    to_keep = to_keep[cols_out]
    new_rows = new_rows[cols_out]
    combined = pd.concat([to_keep, new_rows], ignore_index=True)
    return combined

# ----------------------------
# Backtest política (año validación)
# ----------------------------
def policy_backtest_one_year(master_all: pd.DataFrame, year: int, cfg) -> None:
    df_y = master_all[master_all["season"].eq(year)].copy()
    if df_y.empty or "home_win" not in df_y.columns:
        print(f"[Policy backtest] {year}: sin datos suficientes."); return

    # Construir probs usando heurística simple (NO reentrenamos aquí por brevedad),
    # asumimos ya hay 'p_home_win_meta' si venimos tras run_model para ese año;
    # si no, hacemos proxy con spread prior:
    if "p_home_win_meta" not in df_y.columns:
        prior_lr = LogisticRegression(C=2.0, solver="liblinear", max_iter=200)
        mask_hist = master_all["season"] < year
        if mask_hist.sum() < 200:
            print(f"[Policy backtest] {year}: histórico insuficiente para prior proxy.")
            return
        prior_lr.fit(master_all.loc[mask_hist, ["home_line"]], master_all.loc[mask_hist, "home_win"])
        df_y["p_home_win_meta"] = np.clip(prior_lr.predict_proba(df_y[["home_line"]])[:,1], 1e-6, 1-1e-6)

    # EV table
    homes = pd.DataFrame({
        "season": df_y["season"], "week": df_y["week"], "week_label": df_y["week"].apply(week_label_from_num),
        "schedule_date": df_y.get("schedule_date"),
        "side":"home", "team": df_y["home_team"], "opponent": df_y["away_team"],
        "decimal_odds": df_y.get("decimal_home", np.nan), "ml": df_y.get("ml_home", np.nan),
        "market_prob_nv": df_y.get("market_prob_home_nv", np.nan),
        "model_prob": df_y["p_home_win_meta"], "home_win": df_y["home_win"]
    })
    aways = pd.DataFrame({
        "season": df_y["season"], "week": df_y["week"], "week_label": df_y["week"].apply(week_label_from_num),
        "schedule_date": df_y.get("schedule_date"),
        "side":"away", "team": df_y["away_team"], "opponent": df_y["home_team"],
        "decimal_odds": df_y.get("decimal_away", np.nan), "ml": df_y.get("ml_away", np.nan),
        "market_prob_nv": df_y.get("market_prob_away_nv", np.nan),
        "model_prob": 1.0 - df_y["p_home_win_meta"], "home_win": 1-df_y["home_win"].astype("Int64").fillna(pd.NA)
    })
    ev_all = pd.concat([homes, aways], ignore_index=True)
    ev_all = ev_all[pd.to_numeric(ev_all["decimal_odds"], errors="coerce").notna()].copy()
    ev_all["decimal_odds"] = ev_all["decimal_odds"].astype(float)

    ev_all = sanitize_ev(ev_all, cfg)
    ev_all = devig_per_game(ev_all, cfg["DEVIG_SINGLE_SIDE"])
    ev_all["edge"] = ev_all["model_prob"] - ev_all["market_prob_nv"]
    ev_all["ev"]   = ev_all["model_prob"]*(ev_all["decimal_odds"]-1) - (1-ev_all["model_prob"])

    cand = pick_per_game_best(ev_all, cfg)

    # ROI a 1 unidad fija
    if "home_win" not in cand.columns:
        print(f"[Policy backtest] {year}: sin labels de resultado para ROI.")
        return
    picks = len(cand)
    if picks == 0:
        print(f"[Policy backtest] {year}: 0 picks.")
        return
    pnl = np.where(
        cand["home_win"].astype(int).values == (cand["side"].astype(str).str.lower().eq("home")).astype(int).values,
        cand["decimal_odds"].values - 1.0,
        -1.0
    )
    roi = pnl.mean()
    hit = (pnl > 0).mean()
    print(f"[Policy backtest] {year} | picks={picks} | hit {hit:.3f} | ROI/unit {roi:.3f}")

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(os.path.dirname(BETS_OUT_PATH), exist_ok=True)

    target_season = getenv_int("TARGET_SEASON", detect_target_season())
    target_week_override = getenv_int_or_none("TARGET_WEEK")
    dprint("Resolved -> target_season:", target_season, "| TARGET_WEEK override:", target_week_override)

    # Semana vigente (solo esa se apostará)
    wk_eff_completed = effective_week_completed(target_season)
    wk_target = target_week_override if target_week_override is not None else current_upcoming_week(target_season)
    wk_label  = week_label_from_num(wk_target)

    # Carga odds y stats
    odds_all = load_current_odds()
    team_adj, league_adj = load_team_league_adjusters(target_season)
    dprint("Self-learn adjusters -> teams rows:", len(team_adj), "| league:", league_adj)

    stats_all = load_pregame_stats_for_seasons(target_season)

    # Limitar odds SOLO temporada target y SOLO semana vigente (ni pasadas, ni futuras)
    odds_target = odds_all[(odds_all["season"].eq(target_season)) & (odds_all["week"].eq(wk_target))].copy()
    if odds_target.empty:
        raise RuntimeError(f"No hay odds para la semana vigente {wk_label}.")

    master_target = build_master(odds_target, stats_all)
    dprint("Master target rows:", len(master_target))

    # Auditoría: stats presentes
    pre_cols = [c for c in master_target.columns if re.search(r'(?:_pre(_ewm)?|_pre_ytd|_pre_l8)$', c)]
    print(f"Audit: columnas pregame presentes: {len(pre_cols)} OK")
    miss_h = master_target[pre_cols].filter(like="home_", axis=1).isna().all(axis=1).sum()
    miss_a = master_target[pre_cols].filter(like="away_", axis=1).isna().all(axis=1).sum()
    print(f"Audit: filas sin pregame (home)={miss_h} / (away)={miss_a} de {len(master_target)}")
    print("Audit: 'week' NA rows:", master_target["week"].isna().sum())

    # Cargar odds históricas (para entrenamiento y backtest)
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

    # Modelo
    test_preds = run_model(master_all, target_season)

    # Unir probs con odds target
    cols_key = ["season","week","home_team","away_team"]
    need_cols = cols_key + ["week_label","schedule_date",
                            "decimal_home","decimal_away","ml_home","ml_away",
                            "market_prob_home_nv","market_prob_away_nv","home_line","spread_favorite","over_under_line"]
    merge_df = master_target[[c for c in need_cols if c in master_target.columns]].copy()
    if "week_label" not in merge_df.columns:
        merge_df["week_label"] = merge_df["week"].apply(week_label_from_num)

    pred = test_preds[["season","week","home_team","away_team","p_home_win_meta"]].copy()
    dfm  = merge_df.merge(pred, on=cols_key, how="inner")

    # EV home/away + self-learn tweak
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
    # Ajuste self-learn (suave)
    ev_predictions["model_prob"] = apply_self_learn(ev_predictions["model_prob"], ev_predictions["team"], team_adj, league_adj)

    # Completar ML si falta
    if "ml" in ev_predictions.columns and "decimal_odds" in ev_predictions.columns:
        m = ev_predictions["ml"].isna() & ev_predictions["decimal_odds"].notna()
        ev_predictions.loc[m, "ml"] = ev_predictions.loc[m, "decimal_odds"].apply(decimal_to_american)

    ev_predictions = ev_predictions[pd.to_numeric(ev_predictions["decimal_odds"], errors="coerce").notna()].copy()
    ev_predictions["decimal_odds"] = ev_predictions["decimal_odds"].astype(float)

    # EV / edge
    ev_predictions["ev"]   = ev_predictions["model_prob"]*(ev_predictions["decimal_odds"]-1) - (1-ev_predictions["model_prob"])
    ev_predictions = ev_predictions.sort_values(["schedule_date","week","team","side"]).reset_index(drop=True)

    # Limpieza/sanitizado y devig
    ev_base = sanitize_ev(ev_predictions, CFG)
    ev_base["game_id"] = ev_base.apply(lambda r: make_game_id_row(r["week_label"], r["team"], r["opponent"]), axis=1)
    ev_devig = devig_per_game(ev_base.copy(), single_side_mode=CFG["DEVIG_SINGLE_SIDE"])

    # Distros generales
    def quick_dist(label, s):
        if len(s)==0: return
        q = s.quantile([.1,.25,.5,.75,.9]).round(4).to_dict()
        print(f"{label} dist: mean={s.mean():.4f} p50={q[0.5]:.4f} p75={q[0.75]:.4f} p90={q[0.9]:.4f}")
    quick_dist("edge", ev_devig["model_prob"] - ev_devig["market_prob_nv"])
    quick_dist("ev",   ev_devig["ev"].astype(float))

    # edge y ev recalculados (post devig)
    p = ev_devig["model_prob"].astype(float)
    d = ev_devig["decimal_odds"].astype(float)
    mkt = ev_devig["market_prob_nv"].astype(float)
    ev_devig["edge"] = p - mkt
    ev_devig["ev"]   = p*(d-1) - (1-p)

    # Selección base
    cand = pick_per_game_best(ev_devig, CFG)

    # Near-misses
    def near_misses(ev, cfg):
        tau = cfg["EDGE_TAU"]
        ev2 = ev.copy()
        ev2["edge_gap"] = ev2["edge"] - tau
        ev2["ev_floor"] = ev2["decimal_odds"].apply(lambda dd: ev_floor(dd, cfg["EV_BASE_MIN"], ev_slope_for_row({"decimal_odds":dd}, cfg)))
        ev2["ev_gap"] = ev2["ev"] - ev2["ev_floor"]
        nm = ev2[(ev2["edge_gap"].between(-0.015,0)) | (ev2["ev_gap"].between(-0.004,0))]
        return nm.sort_values(["week_label","edge_gap","ev_gap"])
    nm = near_misses(ev_devig, CFG)
    print("[Near-misses] rows:", len(nm))
    print(nm[["week_label","team","opponent","decimal_odds","model_prob","market_prob_nv","edge","ev","edge_gap","ev_gap"]].head(10))

    # Relajación para alcanzar mínimo SOLO en la semana vigente
    def count_week(df, label): 
        return int(df[df["week_label"].astype(str).eq(str(label))]["game_id"].nunique())
    base_ct = count_week(cand, wk_label)
    weeks_below_min = [wk_label] if base_ct < CFG["MIN_PICKS_PER_WEEK"] else []
    if weeks_below_min:
        dprint("Relaxation: week below min ->", weeks_below_min)
        for step in CFG["RELAX_STEPS"]:
            CFG["EDGE_TAU"]   = step["EDGE_TAU"]
            CFG["CONF_MIN"]   = step["CONF_MIN"]
            CFG["EV_BASE_MIN"]= step["EV_BASE_MIN"]
            print(f"Relax step -> EDGE_TAU={CFG['EDGE_TAU']:.3f} | CONF_MIN={CFG['CONF_MIN']:.3f} | EV_BASE_MIN={CFG['EV_BASE_MIN']:.3f}")
            cand = pick_per_game_best(ev_devig, CFG)
            if count_week(cand, wk_label) >= CFG["MIN_PICKS_PER_WEEK"]:
                break

    # Solo mantener la semana vigente
    cand = cand[cand["week_label"].astype(str).eq(str(wk_label))].reset_index(drop=True)

    # --------- BANKROLL POR SEMANA (interno) ----------
    # (Usado SOLO para stake; NO se escribe columna bankroll)
    if os.path.exists(BETS_OUT_PATH):
        try:
            existing = pd.read_csv(BETS_OUT_PATH, low_memory=False)
            if "schedule_date" in existing.columns:
                existing["schedule_date"] = pd.to_datetime(existing["schedule_date"], errors="coerce", utc=True)
            if "week_label" not in existing.columns and "week" in existing.columns:
                existing["week_label"] = existing["week"].apply(week_label_from_num)
        except Exception as e:
            dprint("WARN: no se pudo leer bets.csv existente; se asumirá INITIAL_BANKROLL. err:", e)
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    # BK0 = último bankroll_week_final de semana previa, si existe; si no, INITIAL
    def resolve_bk0_for_week(wl: str) -> float:
        if existing.empty: return CFG["INITIAL_BANKROLL"]
        ex = existing.copy()
        if "week_label" not in ex.columns and "week" in ex.columns:
            ex["week_label"] = ex["week"].apply(week_label_from_num)
        if "bankroll_week_final" not in ex.columns:
            return CFG["INITIAL_BANKROLL"]
        cur_ord = ORDER_INDEX.get(wl, 999)
        ex["wo"] = ex["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
        prev = ex[(ex["wo"] < cur_ord) & ex["bankroll_week_final"].notna()]
        if prev.empty: return CFG["INITIAL_BANKROLL"]
        return float(prev.sort_values("wo").groupby("week_label")["bankroll_week_final"].last().iloc[-1])

    BK0 = resolve_bk0_for_week(wk_label)
    print(f"BK0 (solo para stake, NO se escribe): {BK0:.2f}")

    # Asignar stake por apuesta (sin escribir bankroll)
    def plan_stakes(df, BK0):
        if df.empty:
            return df.assign(stake=0.0)
        # Cap semanal
        cap_w = BK0 * CFG["WEEKLY_CAP_PCT"]
        # Propuesta inicial
        st = []
        for _, r in df.iterrows():
            st.append(kelly_stake(BK0, float(r["model_prob"]), float(r["decimal_odds"]), float(r["edge"]), CFG))
        df2 = df.copy()
        df2["stake"] = np.round(st, 2)
        # Respetar cap semanal (escalar si excede)
        total = float(df2["stake"].sum())
        if total > cap_w and total > 0:
            scale = cap_w / total
            df2["stake"] = (df2["stake"] * scale).apply(lambda x: float(np.floor(x*100)/100.0))
            print(f"[Stake] scaled by {scale:.3f} to respect weekly cap {cap_w:.2f}")
        return df2

    planned_bets = plan_stakes(cand, BK0)

    # Exposición semana
    if not planned_bets.empty:
        expo = planned_bets.groupby("week_label").agg(picks=("team","size"), stake=("stake","sum")).reset_index()
        print("[Exposición por semana]"); print(expo)

    # ----------------------------
    # APPEND/UPSERT SOLO SEMANA VIGENTE (sin columna bankroll)
    # ----------------------------
    combined = upsert_only_week(existing, planned_bets, wk_label)
    # Redondeos ordenados
    if "edge" in combined.columns: combined["edge"] = pd.to_numeric(combined["edge"], errors="coerce").round(6)
    if "ev" in combined.columns:   combined["ev"]   = pd.to_numeric(combined["ev"],   errors="coerce").round(6)
    if "stake" in combined.columns:combined["stake"]= pd.to_numeric(combined["stake"],errors="coerce").round(2)

    combined.to_csv(BETS_OUT_PATH, index=False)
    print(f"[select_bets] upsert wrote {BETS_OUT_PATH} | prev_rows={len(existing)} | total={len(combined)} | week_upserted={wk_label}")

    # ----------------------------
    # Backtest de política (año de validación)
    # ----------------------------
    # Determinar año de validación usado en run_model
    hist_years = sorted(master_all.loc[master_all["season"] < target_season, "season"].dropna().unique().tolist())
    if len(hist_years) >= 2:
        val_year = hist_years[-1]
        policy_backtest_one_year(master_all, val_year, CFG)

if __name__ == "__main__":
    main()


