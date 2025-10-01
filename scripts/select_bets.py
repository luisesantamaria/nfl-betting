#!/usr/bin/env python3
"""
select_bets.py — picks pregame (min 5, max 8) con rayos X y self-learning.

- Modelo: HGB + calibración (isotónica global + por bin de spread) + meta-LR con baseline de spread.
- Entrenamiento: usa 4 temporadas previas (si existen) + temporada actual para test.
- Self-learn: ajusta probs con señales de pnl.csv por equipo y por liga.
- Semana vigente: detectada por fechas de odds (>= now-18h). Solo genera picks de esa semana.
- Filtros: odds range, confianza, devig por juego, edge, EV floor dependiente de cuota, reglas por bandas.
- Relajación: escalonada (más profunda) para llegar a un mínimo de 5 picks (tope 8). Último recurso: permitir 2 lados del mismo juego si ambos EV>0.
- Escritura: upsert solo semana vigente; no toca semanas finalizadas. No escribe 'bankroll' en bets.csv.

"""

import os, re, warnings
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
# Config
# ----------------------------
DEBUG = True
ET = ZoneInfo("America/New_York")

LIVE_ODDS_PATH  = os.path.join("data", "live", "odds.csv")
LIVE_STATS_PATH = os.path.join("data", "live", "stats.csv")
LIVE_PNL_PATH   = os.path.join("data", "live", "pnl.csv")
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

    # Kelly & caps
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

    DEVIG_SINGLE_SIDE = "raw",

    KELLY_EDGE_WEIGHT = dict(EDGE_REF=0.12, MIN_SCALE=0.60, MAX_SCALE=1.00),

    MAX_BETS_PER_WEEK = 8,  # tope “duro”
    MIN_BETS_PER_WEEK = 5,  # mínimo deseado

    # Bandas
    BANDS = dict(
        FAV = dict(odds_lt=1.60,                 tau=0.050, conf=0.075, ev_slope=0.010),
        MID = dict(odds_ge=1.60, odds_lt=2.40,   tau=0.055, conf=0.080, ev_slope=0.012),
        DOG = dict(odds_ge=2.40, odds_le=3.40,   tau=0.065, conf=0.095, ev_slope=0.017,
                   extra_if_ge3_10=dict(tau=0.070, conf=0.100))
    ),
)

RELAX_STEPS = [
    dict(EDGE_TAU=0.050, CONF_MIN=0.080, EV_BASE_MIN=0.008),
    dict(EDGE_TAU=0.045, CONF_MIN=0.075, EV_BASE_MIN=0.006),
    dict(EDGE_TAU=0.040, CONF_MIN=0.070, EV_BASE_MIN=0.004),
    # Nuevos pasos de “última milla”
    dict(EDGE_TAU=0.0375, CONF_MIN=0.065, EV_BASE_MIN=0.0035),
    dict(EDGE_TAU=0.0350, CONF_MIN=0.060, EV_BASE_MIN=0.0030, allow_two_sides=True),  # último recurso
]

# ----------------------------
# Utils y logging
# ----------------------------
def dprint(*args):
    if DEBUG:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[DBG {ts}]", *args, flush=True)

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
    # temporadas previas
    for y in range(target_season-4, target_season):
        p = os.path.join(ARCHIVE_DIR, f"season={y}", "stats.csv")
        if os.path.exists(p):
            tmp = pd.read_csv(p, low_memory=False)
            tmp["season"] = pd.to_numeric(tmp["season"], errors="coerce").astype("Int64")
            tmp["week"]   = pd.to_numeric(tmp["week"], errors="coerce").astype("Int64")
            tmp["team"]   = tmp["team"].astype(str).map(norm_team)
            frames.append(tmp)
    # temporada actual live
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

def load_self_learn_adjusters():
    league = dict(league_calib_delta=0.0, league_unit_pnl=0.0)
    per_team = pd.DataFrame(columns=["team","unit_pnl","n","calib_delta"])
    if not os.path.exists(LIVE_PNL_PATH):
        dprint("Self-learn adjusters -> pnl.csv no existe; sin ajustes.")
        return per_team, league

    try:
        pnl = pd.read_csv(LIVE_PNL_PATH)
        # esperamos columnas: week_label, team, profit (stake-normalized opcional), result/ status_short
        if "profit" not in pnl.columns:
            return per_team, league
        # agregados por equipo
        g = pnl.groupby("team", dropna=True)["profit"].agg(["sum","count"]).reset_index()
        g["unit_pnl"] = g["sum"] / g["count"].replace(0, np.nan)
        # map a delta calibración suave (sigmoide pequeña)
        # +0.5% por unidad de pnl positivo hasta máx +3%; -0.5% por unidad negativa hasta mín -3%
        g["calib_delta"] = g["unit_pnl"].clip(-6, 6) * 0.005
        g["calib_delta"] = g["calib_delta"].clip(-0.03, 0.03)
        per_team = g[["team","unit_pnl","count","calib_delta"]].rename(columns={"count":"n"})

        league_unit = pnl["profit"].mean()
        league_cal  = max(-0.03, min(0.03, league_unit * 0.004))  # ±3% máx
        league = dict(league_calib_delta=league_cal, league_unit_pnl=league_unit)
        dprint("Self-learn adjusters -> teams rows:", len(per_team), "| league:", league)
    except Exception as e:
        dprint("WARN self-learn: error leyendo pnl.csv:", e)
    return per_team, league

# ----------------------------
# Semana vigente desde odds
# ----------------------------
def detect_current_week_label(odds_df: pd.DataFrame):
    # semanas presentes
    present_weeks = sorted(odds_df["week"].dropna().unique().tolist())
    dprint("Semanas presentes en odds (season target):", present_weeks)

    now = datetime.now(timezone.utc)
    horizon = now - timedelta(hours=18)
    if "schedule_date" not in odds_df.columns:
        # fallback a la última semana numérica
        wk = max(present_weeks) if present_weeks else None
        return week_label_from_num(int(wk)) if wk is not None else None

    cand = odds_df[(odds_df["schedule_date"] >= horizon)]
    if cand.empty:
        # si no hay partidos futuros/próximos, usamos la semana máxima que exista
        wk = max(present_weeks) if present_weeks else None
        lab = week_label_from_num(int(wk)) if wk is not None else None
        dprint("Semana vigente fallback (no futuros >= now-18h):", lab)
        return lab

    wk = int(cand["week"].dropna().max())
    lab = week_label_from_num(wk)
    dprint("Semana vigente detectada por fechas (>= now-18h):", lab)
    return lab

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
# Modelo + métricas (con rayos X)
# ----------------------------
def safe_metrics(y_true, p):
    res = {"ACC": np.nan, "ROC_AUC": np.nan, "LOGLOSS": np.nan}
    try:
        if len(y_true) != len(p):
            dprint("METRICS SHAPE MISMATCH:", len(y_true), "vs", len(p))
            return res
        if len(y_true) == 0:
            return res
        # rayos X de probabilidades
        p_arr = np.asarray(p, dtype=float)
        nan_ct = np.isnan(p_arr).sum()
        dprint("Metrics p-nans:", int(nan_ct), "| p[min,med,max]:", np.nanmin(p_arr), np.nanmedian(p_arr), np.nanmax(p_arr))

        # ACC
        pred_cls = (p_arr >= 0.5).astype(int)
        res["ACC"] = float(accuracy_score(y_true, pred_cls))
        # ROC_AUC
        try:
            res["ROC_AUC"] = float(roc_auc_score(y_true, p_arr))
        except Exception as e:
            dprint("WARN ROC_AUC:", e)
            res["ROC_AUC"] = np.nan
        # LOGLOSS
        try:
            res["LOGLOSS"] = float(log_loss(y_true, np.clip(p_arr, 1e-6, 1-1e-6)))
        except Exception as e:
            dprint("WARN LOGLOSS:", e)
            res["LOGLOSS"] = np.nan
        return res
    except Exception as e:
        dprint("WARN safe_metrics:", e)
        return res

def run_model(master_all: pd.DataFrame, target_season: int, adj_team: pd.DataFrame, adj_league: dict):
    df = master_all.copy()

    # fallback home_line
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

    # splits
    hist_mask = df["season"] < target_season
    base = df.dropna(subset=["home_line"]).copy()

    train_years = []
    val_year = None
    hist_years = sorted(base.loc[hist_mask, "season"].dropna().unique().tolist())
    if len(hist_years) >= 4:
        train_years = hist_years[-4:-1]; val_year = hist_years[-1]
    elif len(hist_years) >= 2:
        train_years = hist_years[:-1]; val_year = hist_years[-1]
    else:
        raise RuntimeError("Se requieren al menos 2 temporadas históricas para entrenar/validar.")

    train_df = base[base["season"].isin(train_years)].copy()
    val_df   = base[base["season"].eq(val_year)].copy()
    test_df  = base[base["season"].eq(target_season)].copy()

    dprint(f"Features usadas: {len(feat_cols)} | train_rows={len(train_df)} | val_rows={len(val_df)} | test_rows={len(test_df)}")

    y_train = train_df["home_win"].dropna().astype(int).values
    y_val   = val_df["home_win"].dropna().astype(int).values

    X_train = train_df[feat_cols].copy()
    X_val   = val_df[feat_cols].copy()
    X_test  = test_df[feat_cols].copy()

    train_meds = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_meds); X_val = X_val.fillna(train_meds); X_test = X_test.fillna(train_meds)

    dprint("Train y unique:", np.unique(y_train, return_counts=True))
    dprint("Val   y unique:", np.unique(y_val, return_counts=True))

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

    # Meta-LR con baseline de spread
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

    # Métricas con rayos X
    tr_metrics = safe_metrics(y_train, np.clip(meta_lr.predict_proba(meta_tr)[:,1], 1e-6, 1-1e-6))
    va_metrics = safe_metrics(y_val,   np.clip(meta_lr.predict_proba(meta_val)[:,1], 1e-6, 1-1e-6))
    print("Model Results:")
    print(f"train | ACC {tr_metrics['ACC'] if not np.isnan(tr_metrics['ACC']) else '---'} | "
          f"ROC_AUC {tr_metrics['ROC_AUC'] if not np.isnan(tr_metrics['ROC_AUC']) else '---'} | "
          f"LOGLOSS {tr_metrics['LOGLOSS'] if not np.isnan(tr_metrics['LOGLOSS']) else '---'}")
    print(f"val   | ACC {va_metrics['ACC'] if not np.isnan(va_metrics['ACC']) else '---'} | "
          f"ROC_AUC {va_metrics['ROC_AUC'] if not np.isnan(va_metrics['ROC_AUC']) else '---'} | "
          f"LOGLOSS {va_metrics['LOGLOSS'] if not np.isnan(va_metrics['LOGLOSS']) else '---'}")

    # Preds target
    test_df = test_df.copy()
    test_df["week_label"] = test_df["week"].apply(week_label_from_num)
    test_preds = test_df[["season","week","week_label","home_team","away_team","schedule_date"]].copy()
    test_preds["p_home_win_lr_cal"] = p_te_cal
    test_preds["p_home_win_meta"]   = p_te_meta

    # Self-learn: aplica ajustes (liga + por equipo) a las probabilidades modeladas
    # ajuste multiplicativo suave sobre el logit => aquí aproximamos con suma en prob con clip
    if not test_preds.empty:
        league_delta = float(adj_league.get("league_calib_delta", 0.0))
        test_preds["p_home_win_meta"] = np.clip(test_preds["p_home_win_meta"] + league_delta, 1e-6, 1-1e-6)
        if adj_team is not None and not adj_team.empty:
            ad = adj_team.set_index("team")["calib_delta"].to_dict()
            # home adj
            test_preds["p_home_win_meta"] = np.clip(
                test_preds["p_home_win_meta"] + test_preds["home_team"].map(ad).fillna(0.0),
                1e-6, 1-1e-6
            )
            # away adj (resta al home)
            test_preds["p_home_win_meta"] = np.clip(
                test_preds["p_home_win_meta"] - test_preds["away_team"].map(ad).fillna(0.0),
                1e-6, 1-1e-6
            )

    return test_preds

# ----------------------------
# EV y selección (con logs de filtros)
# ----------------------------
def sanitize_ev(ev_df: pd.DataFrame, cfg) -> pd.DataFrame:
    need = ["season","week","week_label","schedule_date","side","team","opponent","decimal_odds","model_prob"]
    missing = [c for c in need if c not in ev_df.columns]
    if missing:
        raise ValueError(f"EV table missing columns: {missing}")

    ev = ev_df.copy()
    ev["schedule_date"] = pd.to_datetime(ev["schedule_date"], errors="coerce", utc=True)
    ev["model_prob"] = pd.to_numeric(ev["model_prob"], errors="coerce").astype(float).clip(MIN_PROB, 1-MIN_PROB)
    ev["decimal_odds"] = pd.to_numeric(ev["decimal_odds"], errors="coerce").astype(float)

    ev = add_week_order(ev)
    ev["game_id"] = ev.apply(lambda r: make_game_id_row(r["week_label"], r["team"], r["opponent"]), axis=1)

    dprint("Filtro odds range antes:", len(ev))
    ev = ev[ev["decimal_odds"].between(cfg["ODDS_MIN"], cfg["ODDS_MAX"])]
    dprint("Filtro odds range después:", len(ev))

    dprint("Filtro confianza antes:", len(ev))
    ev = ev[np.abs(ev["model_prob"] - 0.5) >= cfg["CONF_MIN"]]
    dprint("Filtro confianza después:", len(ev))

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

    mask_pairs = sides_per_game >= 2
    dprint("DEVIG pares antes:", mask_pairs.sum(), "de", len(ev))
    ev.loc[mask_pairs, "market_prob_nv"] = (tmp.loc[mask_pairs, "p_raw"] / sum_p[mask_pairs]).clip(1e-6, 1-1e-6)
    dprint("DEVIG aplicado; NaNs market_prob_nv:", int(ev["market_prob_nv"].isna().sum()))
    ev["market_prob_nv"] = ev["market_prob_nv"].clip(1e-6, 1-1e-6)
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
    dropped = (~keep).sum()
    dprint("Band rules drop:", int(dropped), "de", len(x))
    return x[keep].reset_index(drop=True)

def pick_per_game_best(ev: pd.DataFrame, cfg, allow_two_sides=False) -> pd.DataFrame:
    tau = max(cfg["EDGE_TAU"], cfg["EDGE_TAU_MIN"])

    def ok_ev(row):
        slope = ev_slope_for_row(row, cfg)
        return row["ev"] >= ev_floor(row["decimal_odds"], cfg["EV_BASE_MIN"], slope)

    c = ev.copy()
    dprint("Filtro edge antes:", len(c))
    c = c[c["edge"] >= tau]
    dprint("Filtro edge después:", len(c))

    dprint("Filtro EV floor antes:", len(c))
    c = c[c.apply(ok_ev, axis=1)]
    dprint("Filtro EV floor después:", len(c))

    # En principio 1 lado por juego (mejor edge)
    c = (c.sort_values(["week_order","schedule_date","game_id","edge"], ascending=[True,True,True,False]))

    if allow_two_sides:
        # Permite 2 lados si ambos EV>0, pero nunca más de 2 por juego.
        dprint("allow_two_sides=ON (último recurso)")
        out = []
        for gid, g in c.groupby("game_id", sort=False):
            g = g.sort_values("edge", ascending=False)
            keep = g.head(2)  # como máximo 2 lados
            out.append(keep)
        c = pd.concat(out, ignore_index=True)
    else:
        c = c.drop_duplicates("game_id", keep="first")

    c = apply_band_rules(c, cfg)

    # Cap máximo por semana (sólo por ordenación, más adelante se recorta a tope 8)
    out = []
    for wk, g in c.groupby("week_label", sort=False):
        g = g.sort_values(["ev","edge"], ascending=[False,False]).head(cfg["MAX_BETS_PER_WEEK"])
        out.append(g)
    c = pd.concat(out, ignore_index=True) if out else c
    return c.reset_index(drop=True)

# -------- Kelly fraccional (sin exponer bankroll en CSV) ----------
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
# APPEND/UPSERT solo semana vigente
# ----------------------------
APPEND_KEYS = ["season","week","game_id","side"]

def detect_weeks_state(existing):
    """ Devuelve listas de semanas finalizadas (frozen) y no finalizadas. """
    if existing is None or existing.empty:
        return [], []
    ex = existing.copy()
    if "week_label" not in ex.columns and "week" in ex.columns:
        ex["week_label"] = ex["week"].apply(week_label_from_num)
    frozen = []
    if "week_is_final" in ex.columns:
        g = ex.groupby("week_label")["week_is_final"].max()
        frozen = sorted([wl for wl, v in g.items() if bool(v)])
    not_frozen = [wl for wl in ex["week_label"].unique() if wl not in frozen]
    return frozen, not_frozen

def upsert_only_week(existing: pd.DataFrame, new_rows: pd.DataFrame, wk_label: str) -> pd.DataFrame:
    """
    Upsert: solo reemplaza/insert filas de 'wk_label'.
    Las demás semanas en existing quedan intactas.
    Sin escribir 'bankroll'.
    """
    if existing is None or existing.empty:
        return new_rows.copy()

    ex = existing.copy()
    # Claves existentes de esa semana
    have = set(map(tuple, ex.loc[ex["week_label"].astype(str).eq(str(wk_label)), APPEND_KEYS].astype(object).to_numpy().tolist()))
    new_key_list = list(map(tuple, new_rows[APPEND_KEYS].astype(object).to_numpy().tolist()))
    mask_new = ~pd.Series(new_key_list).isin(have)

    # Quita filas previas de esa semana y agrega nuevas
    ex_rest = ex[~ex["week_label"].astype(str).eq(str(wk_label))].copy()
    new_keep = pd.concat([new_rows[mask_new.values],  # nuevas
                          new_rows[~mask_new.values]], ignore_index=True)
    # Quitamos 'bankroll' si hubiera llegado
    if "bankroll" in new_keep.columns:
        new_keep = new_keep.drop(columns=["bankroll"])

    # Unión de columnas:
    cols_existing = list(ex_rest.columns)
    cols_new = list(new_keep.columns)
    all_cols = cols_existing + [c for c in cols_new if c not in cols_existing]
    for c in all_cols:
        if c not in ex_rest.columns: ex_rest[c] = pd.NA
        if c not in new_keep.columns: new_keep[c] = pd.NA
    ex_rest = ex_rest[all_cols]
    new_keep = new_keep[all_cols]

    combined = pd.concat([ex_rest, new_keep], ignore_index=True)
    return combined

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(os.path.dirname(BETS_OUT_PATH), exist_ok=True)

    target_season = getenv_int("TARGET_SEASON", detect_target_season())
    target_week_override = getenv_int_or_none("TARGET_WEEK")
    dprint("Resolved -> target_season:", target_season, "| TARGET_WEEK override:", target_week_override)

    # Carga odds
    odds_cur_all = load_current_odds()
    odds_cur_all = odds_cur_all[odds_cur_all["season"].eq(target_season)].copy()

    # Semana vigente
    wk_label = (week_label_from_num(target_week_override)
                if target_week_override is not None
                else detect_current_week_label(odds_cur_all))
    if not wk_label:
        raise RuntimeError("No fue posible detectar la semana vigente desde odds.")
    dprint("Semana elegida para picks:", wk_label)

    # Limita odds a semana vigente
    odds_cur = odds_cur_all[odds_cur_all["week_label"].astype(str).eq(str(wk_label))].copy()

    # Pregame (historias + live)
    stats_all = load_pregame_stats_for_seasons(target_season)

    # Master
    master_target = build_master(odds_cur, stats_all)
    dprint("Master target rows:", len(master_target))

    # Auditoría de pregame
    home_cols = [c for c in master_target.columns if c.startswith("home_") and c.endswith(("_pre_ytd","_pre_ewm","_pre_l8"))]
    away_cols = [c for c in master_target.columns if c.startswith("away_") and c.endswith(("_pre_ytd","_pre_ewm","_pre_l8"))]
    dprint("Audit: columnas pregame presentes:", len(home_cols)+len(away_cols), "OK")
    dprint("Audit: filas sin pregame (home)=", int(master_target[home_cols].isna().all(axis=1).sum()),
           "/ (away)=", int(master_target[away_cols].isna().all(axis=1).sum()),
           "de", len(master_target))

    # Odds históricas
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

    # Self-learn adjusters
    adj_team, adj_league = load_self_learn_adjusters()

    # Modelo
    test_preds = run_model(master_all, target_season, adj_team, adj_league)

    # Unir probs con odds target (SOLO semana vigente)
    cols_key = ["season","week","home_team","away_team"]
    need_cols = cols_key + ["week_label","schedule_date",
                            "decimal_home","decimal_away","ml_home","ml_away",
                            "market_prob_home_nv","market_prob_away_nv","home_line","spread_favorite","over_under_line"]
    merge_df = master_target[[c for c in need_cols if c in master_target.columns]].copy()
    if "week_label" not in merge_df.columns:
        merge_df["week_label"] = merge_df["week"].apply(week_label_from_num)

    pred = test_preds[["season","week","home_team","away_team","p_home_win_meta"]].copy()
    dfm  = merge_df.merge(pred, on=cols_key, how="inner")
    dprint("Post-merge odds+pred rows (semana vigente):", len(dfm))

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

    ev_predictions = ev_predictions[pd.to_numeric(ev_predictions["decimal_odds"], errors="coerce").notna()].copy()
    ev_predictions["decimal_odds"] = ev_predictions["decimal_odds"].astype(float)

    # EV / edge
    ev_predictions["ev"]   = ev_predictions["model_prob"]*(ev_predictions["decimal_odds"]-1) - (1-ev_predictions["model_prob"])
    ev_predictions = ev_predictions.sort_values(["schedule_date","week","team","side"]).reset_index(drop=True)

    # Sanitizado y DEVIG
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

    # Selección con relajación escalonada hasta mínimo
    picks = pick_per_game_best(ev_devig, CFG, allow_two_sides=False)
    dprint("Candidatos tras pick_per_game_best (base):", len(picks))

    if len(picks) < CFG["MIN_BETS_PER_WEEK"]:
        dprint("Relaxation: semana por debajo de mínimo -> aplicando pasos.")
        for step in RELAX_STEPS:
            CFG_mod = CFG.copy()
            CFG_mod["EDGE_TAU"]   = step["EDGE_TAU"]
            CFG_mod["CONF_MIN"]   = step["CONF_MIN"]
            CFG_mod["EV_BASE_MIN"]= step["EV_BASE_MIN"]
            allow_two = step.get("allow_two_sides", False)
            dprint(f"Relax step -> EDGE_TAU={CFG_mod['EDGE_TAU']} | CONF_MIN={CFG_mod['CONF_MIN']} | EV_BASE_MIN={CFG_mod['EV_BASE_MIN']}{' | allow_two_sides=ON' if allow_two else ''}")
            picks_relaxed = pick_per_game_best(ev_devig, CFG_mod, allow_two_sides=allow_two)
            if len(picks_relaxed) >= CFG["MIN_BETS_PER_WEEK"]:
                picks = picks_relaxed
                break
        if len(picks) < CFG["MIN_BETS_PER_WEEK"]:
            dprint("Relaxation: no se alcanzó el mínimo; devolviendo picks disponibles.")
    # En cualquier caso, respeta el tope máximo
    picks = picks.sort_values(["ev","edge"], ascending=[False,False]).head(CFG["MAX_BETS_PER_WEEK"])
    dprint(f"Candidatos finales para {wk_label}:", len(picks))
    if not picks.empty:
        print(picks[["week_label","schedule_date","game_id","side","team","opponent","decimal_odds","model_prob","market_prob_nv","edge","ev"]].to_string(index=False))

    # Staking (usa BK0 para cap semanal interno, pero NO se escribe bankroll en CSV)
    # BK0 se deriva del último bankroll_week_final previo, si existiera; aquí solo para stake cap
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

    # BK0 por semana: usa bankroll_week_final previo (si existe), sino inicial.
    # Solo para calcular stake, no se escribe.
    bk0 = CFG["INITIAL_BANKROLL"]
    if existing is not None and not existing.empty and "bankroll_week_final" in existing.columns:
        ex = existing.dropna(subset=["bankroll_week_final"]).copy()
        if "week_label" not in ex.columns and "week" in ex.columns:
            ex["week_label"] = ex["week"].apply(week_label_from_num)
        # último finalizado
        ex["week_order"] = ex["week_label"].map(ORDER_INDEX).fillna(999)
        last = ex.sort_values("week_order").groupby("week_label")["bankroll_week_final"].last()
        # toma la previa a la vigente
        this_ord = ORDER_INDEX.get(wk_label, 999)
        prev = [(ORDER_INDEX.get(wl,999), v) for wl,v in last.items() if ORDER_INDEX.get(wl,999) < this_ord]
        if prev:
            prev.sort(reverse=True)
            bk0 = float(prev[0][1])
    dprint(f"BK0 para {wk_label} (solo para caps): {bk0:.2f}")

    # Calcula stake por fila
    stakes = []
    for _, r in picks.iterrows():
        st = kelly_stake(bk0, float(r["model_prob"]), float(r["decimal_odds"]), float(r["edge"]), CFG)
        stakes.append(st)
    picks = picks.copy()
    picks["stake"] = np.round(stakes, 2)
    picks["profit"] = np.nan  # se llena después del partido
    # No incluimos 'bankroll' en CSV

    # Orden de columnas legible
    desired_order = [
        "season","week","week_label","schedule_date",
        "side","team","opponent",
        "decimal_odds","ml","market_prob_nv","model_prob",
        "ev","edge",
        "week_order","game_id",
        "stake","profit",
        # opcionales que quizá existan en odds merge
        "status","status_short","result","team_score","opponent_score","week_is_final"
    ]
    for c in desired_order:
        if c not in picks.columns:
            picks[c] = pd.NA
    picks = picks[desired_order]

    # UPSERT SOLO semana vigente
    combined = upsert_only_week(existing, picks, wk_label)
    combined.to_csv(BETS_OUT_PATH, index=False)
    print(f"[select_bets] upsert wrote {BETS_OUT_PATH} | prev_rows={len(existing)} | total={len(combined)} | weeks_upserted=['{wk_label}']")

if __name__ == "__main__":
    main()
