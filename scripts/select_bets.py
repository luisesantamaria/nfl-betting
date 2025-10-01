#!/usr/bin/env python3
"""
select_bets.py (pregame staking only)

- Entrena el modelo (HGB + calibración + meta LR).
- Usa stats pregame históricos (4 temporadas previas) + temporada actual.
- Une con odds actuales (data/live/odds.csv) solo para temporada target.
- Calcula EV/edge, aplica filtros y garantiza un mínimo de picks por semana
  (mediante relajación progresiva de umbrales si hace falta).
- Escribe/actualiza data/live/bets.csv con upsert SOLO sobre la semana vigente
  y sin tocar semanas pasadas finalizadas.

Cambios destacados:
- Semana vigente derivada de 'odds.csv' (no calendario).
- Prohíbe semanas pasadas o futuras: solo la semana detectada.
- No incluye columna 'bankroll' en el CSV.
- Orden de columnas estable y legible.
"""

import os, re, warnings
from datetime import datetime, timezone, timedelta
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

# --------- Filtros base + control de mínimos ----------
CFG = dict(
    INITIAL_BANKROLL = 1000.0,  # solo interno si vuelves a activar caps; NO se escribe

    # Kelly & caps (si decides reactivar caps globales)
    KELLY_FRACTION   = 0.25,
    KELLY_PCT_CAP    = 0.05,
    ABS_STAKE_CAP    = 300.0,

    # Filtros globales
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

    # Targets por semana
    MIN_PICKS_PER_WEEK = 5,    # mínimo deseado
    MAX_PICKS_PER_WEEK = 8,    # techo razonable

    # Bandas
    BANDS = dict(
        FAV = dict(odds_lt=1.60,                 tau=0.050, conf=0.075, ev_slope=0.010),
        MID = dict(odds_ge=1.60, odds_lt=2.40,   tau=0.055, conf=0.080, ev_slope=0.012),
        DOG = dict(odds_ge=2.40, odds_le=3.40,   tau=0.065, conf=0.095, ev_slope=0.017,
                   extra_if_ge3_10=dict(tau=0.070, conf=0.100))
    ),

    # Relajación (si no se logra MIN picks)
    RELAX_STEPS = [
        dict(EDGE_TAU=0.050, CONF_MIN=0.080, EV_BASE_MIN=0.008),
        dict(EDGE_TAU=0.045, CONF_MIN=0.075, EV_BASE_MIN=0.006),
        dict(EDGE_TAU=0.040, CONF_MIN=0.070, EV_BASE_MIN=0.004),
    ],
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
# Derivar SEMANA VIGENTE desde ODDS
# ----------------------------
def detect_current_week_from_odds(odds_season_df: pd.DataFrame) -> int:
    """
    Si hay múltiples semanas en odds:
      - prioriza la semana máxima cuyo schedule_date >= (now_utc - 18h)  (evita semanas ya muy pasadas)
      - si nada cumple, usa la semana máxima pura.
    """
    now_utc = datetime.now(timezone.utc)
    weeks = sorted(odds_season_df["week"].dropna().unique().astype(int).tolist())
    dprint("Semanas presentes en odds (season target):", weeks)
    if not weeks:
        raise RuntimeError("No hay semanas en odds para la temporada target.")
    # Candidatas con fecha futura o muy reciente
    mask = odds_season_df["schedule_date"] >= (now_utc - timedelta(hours=18))
    cand = odds_season_df.loc[mask, "week"].dropna().astype(int).unique().tolist()
    if cand:
        wk = max(cand)
        dprint(f"Semana vigente detectada por fechas (>= now-18h): Week {wk}")
        return int(wk)
    wk = max(weeks)
    dprint(f"Semana vigente por fallback a semana máxima en odds: Week {wk}")
    return int(wk)

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
# Modelo + métricas + self-learn ligera
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

    # -------- Hist vs target --------
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

    # Datos de etiqueta
    for part, part_df in [("train",train_df),("val",val_df),("test",test_df)]:
        if "home_win" not in part_df.columns:
            raise RuntimeError(f"'{part}' no tiene 'home_win' para evaluar/entrenar.")

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
    iso_global = IsotonicRegression(out_of_bounds="clip")
    iso_global.fit(p_va, y_val)

    # Métricas rápidas
    def safe_metrics(y_true, p):
        try:
            pred = (p>=0.5).astype(int)
            return dict(
                ACC = accuracy_score(y_true, pred),
                ROC_AUC = roc_auc_score(y_true, p),
                LOGLOSS = log_loss(y_true, p, eps=1e-9)
            )
        except Exception:
            return dict(ACC=np.nan, ROC_AUC=np.nan, LOGLOSS=np.nan)

    m_tr = safe_metrics(y_train, iso_global.transform(p_tr))
    m_va = safe_metrics(y_val,   iso_global.transform(p_va))

    # 'test' solo se usa para score out-of-sample de la temporada target (no entrena)
    if len(p_te):
        # NO conocemos la verdad de 'test' en vivo; imprimimos score solo si hay etiquetas
        if "home_win" in test_df.columns and test_df["home_win"].notna().any():
            y_te = test_df["home_win"].dropna().astype(int).values
            m_te = safe_metrics(y_te, iso_global.transform(p_te[:len(y_te)]))
            print(f"Model Results:\ntrain | ACC {m_tr['ACC']:.3f} | ROC_AUC {m_tr['ROC_AUC']:.3f} | LOGLOSS {m_tr['LOGLOSS']:.3f}\n"
                  f"val   | ACC {m_va['ACC']:.3f} | ROC_AUC {m_va['ROC_AUC']:.3f} | LOGLOSS {m_va['LOGLOSS']:.3f}\n"
                  f"test* | ACC {m_te['ACC']:.3f} | ROC_AUC {m_te['ROC_AUC']:.3f} | LOGLOSS {m_te['LOGLOSS']:.3f}")
        else:
            print(f"Model Results:\ntrain | ACC {m_tr['ACC']:.3f} | ROC_AUC {m_tr['ROC_AUC']:.3f} | LOGLOSS {m_tr['LOGLOSS']:.3f}\n"
                  f"val   | ACC {m_va['ACC']:.3f} | ROC_AUC {m_va['ROC_AUC']:.3f} | LOGLOSS {m_va['LOGLOSS']:.3f}\n"
                  f"test* | ACC ---  | ROC_AUC ---  | LOGLOSS ---")
    else:
        print(f"Model Results:\ntrain | ACC {m_tr['ACC']:.3f} | ROC_AUC {m_tr['ROC_AUC']:.3f} | LOGLOSS {m_tr['LOGLOSS']:.3f}\n"
              f"val   | ACC {m_va['ACC']:.3f} | ROC_AUC {m_va['ROC_AUC']:.3f} | LOGLOSS {m_va['LOGLOSS']:.3f}")

    # Probabilidades calibradas para temporada target
    p_te_cal = iso_global.transform(p_te) if len(p_te) else np.array([])

    # Meta modelo con señal de spread sencilla
    prior_lr = LogisticRegression(C=2.0, solver="liblinear", max_iter=200)
    prior_lr.fit(train_df[["home_line"]], y_train)
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

    meta_test = meta_matrix(test_df, p_te_cal, p_te_sp) if len(test_df) else pd.DataFrame()
    meta_lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=300)
    if len(meta_test):
        # Entrena metamodelo con validación (no con test target)
        p_va_sp = prior_lr.predict_proba(val_df[["home_line"]])[:,1]
        meta_val = meta_matrix(val_df, iso_global.transform(p_va), p_va_sp)
        meta_lr.fit(meta_val, y_val)

    p_te_meta = (np.clip(meta_lr.predict_proba(meta_test)[:,1], 1e-6, 1-1e-6)
                 if len(meta_test) else np.array([]))

    # Preds para temporada target
    test_df = test_df.copy()
    test_df["week_label"] = test_df["week"].apply(week_label_from_num)
    test_preds = test_df[["season","week","week_label","home_team","away_team"]].copy()
    test_preds["p_home_win_meta"] = p_te_meta if len(p_te_meta) else np.array([np.nan]*len(test_df))
    return test_preds

# ----------------------------
# EV, selección y relajación
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

    ev = ev[ev["decimal_odds"].between(cfg["ODDS_MIN"], cfg["ODDS_MAX"])]
    ev = ev[np.abs(ev["model_prob"] - 0.5) >= cfg["CONF_MIN"]]
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
    ev.loc[mask_pairs, "market_prob_nv"] = (tmp.loc[mask_pairs, "p_raw"] / sum_p[mask_pairs]).clip(1e-6, 1-1e-6)

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
    return x[keep].reset_index(drop=True)

def pick_per_game_best(ev: pd.DataFrame, cfg) -> pd.DataFrame:
    tau = max(cfg["EDGE_TAU"], cfg["EDGE_TAU_MIN"])
    def ok_ev(row):
        slope = ev_slope_for_row(row, cfg)
        return row["ev"] >= ev_floor(row["decimal_odds"], cfg["EV_BASE_MIN"], slope)
    c = ev[ev["edge"] >= tau].copy()
    if c.empty: return c
    c = c[c.apply(ok_ev, axis=1)]
    c = (c.sort_values(["week_order","schedule_date","game_id","edge"], ascending=[True,True,True,False])
           .drop_duplicates("game_id"))
    c = apply_band_rules(c, cfg)
    return c.reset_index(drop=True)

def ensure_min_per_week(cand_week: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Si no se llega a MIN_PICKS_PER_WEEK, relaja umbrales en escalones.
    Limita también a MAX_PICKS_PER_WEEK si se excede.
    """
    target_min = cfg.get("MIN_PICKS_PER_WEEK", 0)
    target_max = cfg.get("MAX_PICKS_PER_WEEK", 999)

    if len(cand_week) >= target_min:
        return cand_week.sort_values(["ev","edge"], ascending=[False,False]).head(target_max)

    dprint("Relaxation: semana por debajo de mínimo -> aplicando pasos.")
    base_cfg = dict(EDGE_TAU=cfg["EDGE_TAU"], CONF_MIN=cfg["CONF_MIN"], EV_BASE_MIN=cfg["EV_BASE_MIN"])
    ev_base = cand_week.copy()  # ya pasó filtros básicos; recalcularemos con ajustes

    # Nota: para reintentar, necesitamos volver a filtrar desde los candidatos previos a pick_per_game_best.
    # Para simplificar guardamos las columnas esenciales:
    needed = ["season","week","week_label","schedule_date","side","team","opponent",
              "decimal_odds","ml","market_prob_nv","model_prob","ev","edge","game_id","week_order"]
    ev_base = ev_base[needed].copy()

    for step in cfg.get("RELAX_STEPS", []):
        cfg["EDGE_TAU"]   = step["EDGE_TAU"]
        cfg["CONF_MIN"]   = step["CONF_MIN"]
        cfg["EV_BASE_MIN"]= step["EV_BASE_MIN"]
        dprint(f"Relax step -> EDGE_TAU={cfg['EDGE_TAU']:.3f} | CONF_MIN={cfg['CONF_MIN']:.3f} | EV_BASE_MIN={cfg['EV_BASE_MIN']:.3f}")

        # Re-run pick rules sobre TODO el conjunto semanal original (no solo los ya filtrados)
        tmp = sanitize_ev(ev_base.copy(), cfg)  # vuelve a aplicar CONF/odds, etc.
        tmp = devig_per_game(tmp, cfg["DEVIG_SINGLE_SIDE"])
        p = tmp["model_prob"].astype(float); dds = tmp["decimal_odds"].astype(float); m = tmp["market_prob_nv"].astype(float)
        tmp["edge"] = p - m
        tmp["ev"]   = p*(dds-1) - (1-p)
        out = pick_per_game_best(tmp, cfg)
        if len(out) >= target_min:
            return out.sort_values(["ev","edge"], ascending=[False,False]).head(target_max)

    # No alcanzamos el mínimo ni con relajación: devolvemos lo que haya (cero o pocos)
    dprint("Relaxation: no se alcanzó el mínimo; devolviendo picks disponibles.")
    return cand_week.sort_values(["ev","edge"], ascending=[False,False]).head(target_max)

# ----------------------------
# APPEND-ONLY / UPSERT (solo semana vigente)
# ----------------------------
APPEND_KEYS = ["season","week","game_id","side"]

def upsert_only_week(existing: pd.DataFrame, new_rows: pd.DataFrame, target_week_label: str) -> pd.DataFrame:
    """
    Reemplaza por clave (season,week,game_id,side) SOLO dentro de la semana vigente.
    No toca semanas pasadas ni futuras.
    """
    if existing is None or existing.empty:
        return new_rows.copy()

    ex = existing.copy()
    if "week_label" not in ex.columns and "week" in ex.columns:
        ex["week_label"] = ex["week"].apply(week_label_from_num)

    # Filas que se van a eliminar (claves que colisionan en la semana vigente)
    have = set(map(tuple, ex[APPEND_KEYS].astype(object).to_numpy().tolist())) if not ex.empty else set()
    new_keys = list(map(tuple, new_rows[APPEND_KEYS].astype(object).to_numpy().tolist()))
    mask_match = pd.Series(new_keys).isin(have)

    if mask_match.any():
        dprint(f"Upsert: se eliminarán {mask_match.sum()} filas existentes de semanas modificables por clave match (upsert).")

    # Excluye de ex SOLO la semana vigente con las claves a reemplazar
    if not ex.empty:
        m_same_week = ex["week_label"].astype(str).eq(str(target_week_label))
        if m_same_week.any():
            # elimina las filas de esa semana que tengan la misma clave
            ex_keys = set(map(tuple, ex.loc[m_same_week, APPEND_KEYS].astype(object).to_numpy().tolist()))
            new_keys_set = set(new_keys)
            clash = ex_keys.intersection(new_keys_set)
            if clash:
                clash_idx = ex.loc[m_same_week].apply(lambda r: tuple(r[k] for k in APPEND_KEYS), axis=1).isin(clash)
                ex = pd.concat([ex.loc[~m_same_week], ex.loc[m_same_week & ~clash_idx]], ignore_index=True)

    combined = pd.concat([ex, new_rows], ignore_index=True)

    return combined

# ----------------------------
# Orden de columnas de salida
# ----------------------------
OUT_FIRST = [
    "season","week","week_label","schedule_date",
    "side","team","opponent",
    "decimal_odds","ml","market_prob_nv","model_prob",
    "ev","edge","week_order","game_id",
    "stake","profit",
    "status","result"
]
def order_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    ordered = [c for c in OUT_FIRST if c in cols] + [c for c in cols if c not in OUT_FIRST]
    return df[ordered]

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

    # Odds solo de temporada target
    odds_cur_season = odds_all[odds_all["season"].eq(target_season)].copy()
    if odds_cur_season.empty:
        raise RuntimeError(f"No hay odds para la temporada {target_season}.")

    # Detecta semana vigente desde las ODDS (o respeta override)
    if target_week_override is not None:
        target_week = int(target_week_override)
        dprint(f"Semana vigente por override explícito: Week {target_week}")
    else:
        target_week = detect_current_week_from_odds(odds_cur_season)

    wk_label = week_label_from_num(target_week)
    dprint(f"Semana elegida para picks: {wk_label}")

    # Filtra a SOLO esa semana (prohíbe semanas pasadas/futuras)
    odds_cur = odds_cur_season[odds_cur_season["week"].eq(target_week)].copy()
    if odds_cur.empty:
        # Mensaje explícito con pistas
        semanas = odds_cur_season["week"].dropna().unique().astype(int).tolist()
        raise RuntimeError(f"No hay odds para la semana {wk_label}. Semanas presentes en odds: {semanas}")

    # Master target (solo semana vigente)
    master_target = build_master(odds_cur, stats_all)
    dprint("Master target rows:", len(master_target))

    # Auditoría: verificar que pregame exista en master_target
    pre_cols = [c for c in master_target.columns if re.search(r'(?:_pre(_ewm)?|_pre_ytd|_pre_l8)$', c)]
    missing_home = master_target[[c for c in pre_cols if c.startswith("home_")]].isna().all(axis=1).sum()
    missing_away = master_target[[c for c in pre_cols if c.startswith("away_")]].isna().all(axis=1).sum()
    dprint(f"Audit: columnas pregame presentes: {len(pre_cols)} OK")
    dprint(f"Audit: filas sin pregame (home)={missing_home} / (away)={missing_away} de {len(master_target)}")

    # Cargar odds históricas (4 temporadas previas) para entrenamiento
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

    # Modelo + métricas
    test_preds = run_model(master_all, target_season)

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
    ev_all = pd.concat([homes, aways], ignore_index=True)

    # Completar ML si falta
    if "ml" in ev_all.columns and "decimal_odds" in ev_all.columns:
        m = ev_all["ml"].isna() & ev_all["decimal_odds"].notna()
        ev_all.loc[m, "ml"] = ev_all.loc[m, "decimal_odds"].apply(decimal_to_american)

    ev_all = ev_all[pd.to_numeric(ev_all["decimal_odds"], errors="coerce").notna()].copy()
    ev_all["decimal_odds"] = ev_all["decimal_odds"].astype(float)

    # EV / edge
    ev_all["ev"]   = ev_all["model_prob"]*(ev_all["decimal_odds"]-1) - (1-ev_all["model_prob"])

    # Limpieza y devig
    ev_base = sanitize_ev(ev_all, CFG)
    ev_base = ev_base[ev_base["week_label"].astype(str).eq(week_label)]  # **solo semana vigente**
    ev_base["game_id"] = ev_base.apply(lambda r: make_game_id_row(r["week_label"], r["team"], r["opponent"]), axis=1)
    ev_devig = devig_per_game(ev_base.copy(), single_side_mode=CFG["DEVIG_SINGLE_SIDE"])

    p = ev_devig["model_prob"].astype(float)
    dds = ev_devig["decimal_odds"].astype(float)
    mkt = ev_devig["market_prob_nv"].astype(float)
    ev_devig["edge"] = p - mkt
    ev_devig["ev"]   = p*(dds-1) - (1-p)

    # pick base
    cand_base = pick_per_game_best(ev_devig, CFG)
    # asegurar mínimo por semana
    cand = ensure_min_per_week(cand_base, CFG)

    dprint(f"Candidatos finales para {week_label}: {len(cand)}")
    if len(cand):
        print(cand[["week_label","schedule_date","game_id","side","team","opponent","decimal_odds","model_prob","market_prob_nv","edge","ev"]]
              .sort_values(["schedule_date","game_id"]).to_string(index=False))

    # Staking (simple Kelly fraccional; NO guardamos 'bankroll' en CSV)
    def kelly_fraction_scaled(edge, cfg):
        wd = cfg.get("KELLY_EDGE_WEIGHT")
        if not wd: return cfg["KELLY_FRACTION"]
        scale = np.clip(float(edge) / wd["EDGE_REF"], wd["MIN_SCALE"], wd["MAX_SCALE"])
        return cfg["KELLY_FRACTION"] * float(scale)

    def kelly_stake_simple(p, d, edge, cfg, base_bk=CFG["INITIAL_BANKROLL"]):
        b = max(float(d) - 1.0, 1e-9)
        f_star = (b*float(p) - (1-float(p))) / b
        f_base = max(0.0, f_star)
        f_adj  = kelly_fraction_scaled(edge, cfg)
        stake = base_bk * f_base * f_adj
        stake = min(stake, base_bk*cfg["KELLY_PCT_CAP"], cfg["ABS_STAKE_CAP"])
        return float(np.floor(stake*100)/100.0)

    if cand.empty:
        planned = cand.copy()
        planned["stake"] = 0.0
        planned["profit"] = np.nan
    else:
        planned = cand.copy()
        planned["stake"] = [
            kelly_stake_simple(row["model_prob"], row["decimal_odds"], row["edge"], CFG)
            for _, row in planned.iterrows()
        ]
        planned["profit"] = np.nan

    # Lee bets existentes
    if os.path.exists(BETS_OUT_PATH):
        try:
            existing = pd.read_csv(BETS_OUT_PATH, low_memory=False)
            if "schedule_date" in existing.columns:
                existing["schedule_date"] = pd.to_datetime(existing["schedule_date"], errors="coerce", utc=True)
            if "week_label" not in existing.columns and "week" in existing.columns:
                existing["week_label"] = existing["week"].apply(week_label_from_num)
        except Exception as e:
            dprint("WARN: no se pudo leer bets.csv existente; se asumirá vacío. err:", e)
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    # Unificar columnas necesarias antes de upsert
    for c in APPEND_KEYS:
        if c not in planned.columns and c in ["season","week"]:
            planned[c] = pd.to_numeric(planned[c], errors="coerce").astype("Int64")
    needed_min = set(APPEND_KEYS + ["week_label","schedule_date","side","team","opponent",
                                    "decimal_odds","ml","market_prob_nv","model_prob","ev","edge","week_order","game_id",
                                    "stake","profit"])
    for c in needed_min:
        if c not in planned.columns:
            planned[c] = pd.NA

    # Orden amigable (y NO incluimos 'bankroll')
    planned_out = order_output_columns(planned)

    # UPSERT SOLO SEMANA VIGENTE
    combined = upsert_only_week(existing, planned_out, wk_label)

    # Asegurar orden amigable global
    combined_out = order_output_columns(combined)

    combined_out.to_csv(BETS_OUT_PATH, index=False)
    print(f"[select_bets] upsert wrote {BETS_OUT_PATH} | prev_rows={len(existing)} | total={len(combined_out)} | weeks_upserted=['{wk_label}']")

if __name__ == "__main__":
    main()

