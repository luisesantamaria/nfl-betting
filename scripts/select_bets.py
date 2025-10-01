#!/usr/bin/env python3
"""
select_bets.py (pregame staking only)

- Entrena el modelo (HGB + calibración + meta LR) igual al notebook.
- Usa stats pregame históricos (4 temporadas previas) + temporada actual.
- Une con odds actuales (data/live/odds.csv) solo para temporada target.
- Calcula EV/edge, aplica estrategia con filtros y planifica stake pregame
  usando como bankroll base por semana el `bankroll_week_final` de la semana anterior
  (si no existe, cae a INITIAL_BANKROLL). NUNCA usa 'won' ni resultados.
- Escribe apuestas seleccionadas a data/live/bets.csv en modo UPSERT por semana
  (frozen las semanas ya finalizadas; la semana actual se puede reescribir).
- La columna 'bankroll' ya NO se guarda en el CSV (se usa solo para calcular stake).
"""

import os, re, warnings
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

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

# --------- Filtros (ligeramente relajados) y mínimos ---------
CFG = dict(
    INITIAL_BANKROLL = 1000.0,

    # Kelly & caps
    KELLY_FRACTION   = 0.25,
    KELLY_PCT_CAP    = 0.05,
    ABS_STAKE_CAP    = 300.0,
    WEEKLY_CAP_PCT   = 0.50,     # cap semanal = 50% del bankroll base de esa semana

    # Filtros globales
    ODDS_MIN         = 1.20,
    ODDS_MAX         = 3.60,     # un poco más de rango arriba
    CONF_MIN         = 0.085,    # leve relajación
    EDGE_TAU         = 0.055,    # leve relajación
    EDGE_TAU_MIN     = 0.040,
    EV_BASE_MIN      = 0.010,    # leve relajación
    EV_SLOPE         = 0.011,
    SKIP_W1_W2       = True,

    DEVIG_SINGLE_SIDE = "raw",

    KELLY_EDGE_WEIGHT = dict(EDGE_REF=0.12, MIN_SCALE=0.60, MAX_SCALE=1.00),

    # Límite superior flexible + mínimo semanal
    MAX_BETS_PER_WEEK = 8,
    MIN_BETS_PER_WEEK = 5,   # mínimo deseado por semana si hay ≥5 con EV ≥ 0

    # Bandas
    BANDS = dict(
        FAV = dict(odds_lt=1.60,                 tau=0.048, conf=0.072, ev_slope=0.010),
        MID = dict(odds_ge=1.60, odds_lt=2.50,   tau=0.053, conf=0.078, ev_slope=0.011),
        DOG = dict(odds_ge=2.50, odds_le=3.60,   tau=0.062, conf=0.092, ev_slope=0.016,
                   extra_if_ge3_10=dict(tau=0.067, conf=0.098))
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
    """
    Selecciona por juego (una sola cara) y aplica bandas. Si en una semana
    hay menos de MIN_BETS_PER_WEEK picks, se aplica una relajación gradual,
    pero NUNCA se fuerzan picks con EV negativo.
    """
    tau = max(cfg["EDGE_TAU"], cfg["EDGE_TAU_MIN"])
    def ok_ev(row):
        slope = ev_slope_for_row(row, cfg)
        return row["ev"] >= ev_floor(row["decimal_odds"], cfg["EV_BASE_MIN"], slope)

    # 1) Filtro base: por juego (mejor edge) respetando floor de EV
    c = ev[ev["edge"] >= tau].copy()
    if c.empty:
        return c
    c = c[c.apply(ok_ev, axis=1)]

    # Una por juego (máx 1 side)
    c = (c.sort_values(["week_order","schedule_date","game_id","edge"], ascending=[True,True,True,False])
           .drop_duplicates("game_id"))

    # 2) Bandas
    c = apply_band_rules(c, cfg)

    # 3) Top-N por semana (techo flexible)
    if cfg.get("MAX_BETS_PER_WEEK"):
        out = []
        for wk, g in c.groupby("week_label", sort=False):
            g = g.sort_values(["ev","edge"], ascending=[False,False]).head(cfg["MAX_BETS_PER_WEEK"])
            out.append(g)
        c = pd.concat(out, ignore_index=True)

    # 4) Mínimo por semana (relajación gradual sin EV negativo)
    min_per_week = cfg.get("MIN_BETS_PER_WEEK", 0)
    if min_per_week and min_per_week > 0:
        final_chunks = []
        for wk, g in c.groupby("week_label", sort=False):
            if len(g) >= min_per_week:
                final_chunks.append(g)
                continue

            # Pool de candidatos adicionales (misma semana) con EV≥0
            wk_pool = ev[(ev["week_label"] == wk) & (ev["ev"] >= 0)].copy()

            # Eliminar juegos ya tomados
            taken_gids = set(g["game_id"].tolist())
            wk_pool = wk_pool[~wk_pool["game_id"].isin(taken_gids)]

            # Relajación 1: bajar tau al EDGE_TAU_MIN y sin bandas
            tau_relax = cfg.get("EDGE_TAU_MIN", 0.04)
            add1 = wk_pool[wk_pool["edge"] >= tau_relax].copy()

            # Relajación 2: si aún falta, ignora tau y quédate con top por EV≥0
            need = max(0, min_per_week - len(g))
            add_pool = (add1 if len(add1) >= need else wk_pool).copy()

            # Escoge máximo una cara por juego
            add_pool = (add_pool.sort_values(["game_id","ev","edge"], ascending=[True,False,False])
                                  .drop_duplicates("game_id"))

            # Ordena por EV, luego edge
            add_pool = add_pool.sort_values(["ev","edge"], ascending=[False,False])

            # Agrega hasta completar el mínimo (sin exceder MAX_BETS_PER_WEEK si existe)
            cap = cfg.get("MAX_BETS_PER_WEEK")
            max_to_add = need if cap is None else max(0, min(need, cap - len(g)))
            add_pick = add_pool.head(max_to_add)

            g2 = pd.concat([g, add_pick], ignore_index=True).sort_values(["ev","edge"], ascending=[False,False])
            final_chunks.append(g2)

        c = pd.concat(final_chunks, ignore_index=True)

    return c.reset_index(drop=True)

# -------- Kelly fraccional + caps (PREGAME, usando BK0 por semana) ----------
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

def plan_pregame_stakes(bets_in: pd.DataFrame, cfg, bk_by_week: dict):
    """
    Asigna stake PRE-GAME por semana usando BK0 = bankroll base de esa semana,
    tomado de bk_by_week (derivado de bankroll_week_final previo). 'profit' NaN.
    No agrega columna 'bankroll' en la salida (solo se usa internamente).
    """
    if bets_in.empty:
        return bets_in.assign(stake=0.0, profit=np.nan), pd.DataFrame(), {
            "bets_planned": 0, "total_planned_stake": 0.0, "bankroll_assumed": None
        }

    data = add_week_order(bets_in).sort_values(["week_order","schedule_date","game_id"]).reset_index(drop=True)

    stakes = []
    for wk, g in data.groupby("week_label", sort=False):
        BK0 = float(bk_by_week.get(wk, cfg["INITIAL_BANKROLL"]))
        for _, r in g.iterrows():
            st = kelly_stake(BK0, float(r["model_prob"]), float(r["decimal_odds"]), float(r["edge"]), cfg)
            stakes.append(st)

    out = data.copy()
    out["stake"] = np.round(stakes, 2)
    out["profit"] = np.nan

    wk_summary = (out.groupby("week_label", sort=False)
                    .agg(planned_stake=("stake","sum"))
                    .reset_index())

    summary = {
        "bets_planned": int(len(out)),
        "total_planned_stake": float(np.round(out["stake"].sum(), 2)),
        "bankroll_assumed": {wk: float(bk_by_week.get(wk, cfg["INITIAL_BANKROLL"])) for wk in out["week_label"].unique()}
    }
    return out, wk_summary, summary

# ----------------------------
# UPSERT + bankroll_week_final (para BK0)
# ----------------------------
APPEND_KEYS = ["season","week","game_id","side"]

def build_bk_by_week(existing: pd.DataFrame, planned_weeks: list, cfg) -> dict:
    """
    Para cada week_label en 'planned_weeks', usa el último bankroll_week_final
    de una semana previa (week_order menor). Si no hay, usa INITIAL_BANKROLL.
    """
    bk_by_week = {}
    if existing is None or existing.empty or "bankroll_week_final" not in existing.columns:
        for wk in planned_weeks: bk_by_week[wk] = cfg["INITIAL_BANKROLL"]
        return bk_by_week

    ex = existing.copy()
    if "week_label" not in ex.columns and "week" in ex.columns:
        ex["week_label"] = ex["week"].apply(week_label_from_num)
    if "week_order" not in ex.columns:
        ex["week_order"] = ex["week_label"].map(ORDER_INDEX).fillna(999).astype(int)

    wk2final = (ex.dropna(subset=["bankroll_week_final"])
                  .sort_values(["week_order"])
                  .groupby("week_label")["bankroll_week_final"]
                  .last()
                  .to_dict())

    for wk in planned_weeks:
        wk_ord = ORDER_INDEX.get(str(wk), 999)
        prev_vals = [(ORDER_INDEX.get(str(lbl), 999), val)
                     for lbl, val in wk2final.items()
                     if ORDER_INDEX.get(str(lbl), 999) < wk_ord]
        if prev_vals:
            prev_vals.sort(key=lambda x: x[0], reverse=True)
            bk_by_week[wk] = float(prev_vals[0][1])
        else:
            bk_by_week[wk] = cfg["INITIAL_BANKROLL"]
    return bk_by_week

def upsert_by_week(existing: pd.DataFrame, new_rows: pd.DataFrame, modifiable_weeks: list) -> pd.DataFrame:
    """
    Elimina de 'existing' las filas de semanas 'modificables' con misma clave (season,week,game_id,side)
    y luego inserta 'new_rows' de esas semanas. Las semanas 'frozen' quedan intactas.
    """
    if existing is None or existing.empty:
        return new_rows.copy()

    ex = existing.copy()
    if "week_label" not in ex.columns and "week" in ex.columns:
        ex["week_label"] = ex["week"].apply(week_label_from_num)

    # Partir en frozen vs modificables
    frozen_mask = ~ex["week_label"].isin(modifiable_weeks)
    frozen_part = ex[frozen_mask].copy()
    mod_part    = ex[~frozen_mask].copy()

    # Filtrar new_rows solo a semanas modificables
    nr = new_rows[new_rows["week_label"].isin(modifiable_weeks)].copy()

    # Clave en mod_part
    if not mod_part.empty:
        have = set(map(tuple, mod_part[APPEND_KEYS].astype(object).to_numpy().tolist()))
    else:
        have = set()

    # De la parte modificable actual, removemos las claves que vamos a insertar
    if not nr.empty and not mod_part.empty:
        rem_keys = set(map(tuple, nr[APPEND_KEYS].astype(object).to_numpy().tolist()))
        keep_mask = ~pd.Series(list(map(tuple, mod_part[APPEND_KEYS].astype(object).to_numpy().tolist()))).isin(rem_keys)
        mod_part = mod_part[keep_mask.values].copy()

    combined = pd.concat([frozen_part, mod_part, nr], ignore_index=True)

    # Unión de columnas consistente
    all_cols = list({c for c in combined.columns} | {c for c in new_rows.columns})
    for c in all_cols:
        if c not in combined.columns: combined[c] = pd.NA
        if c not in new_rows.columns:  new_rows[c]  = pd.NA
    combined = combined[all_cols]

    return combined

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(os.path.dirname(BETS_OUT_PATH), exist_ok=True)

    target_season = getenv_int("TARGET_SEASON", detect_target_season())
    target_week_override = getenv_int_or_none("TARGET_WEEK")
    dprint("Resolved -> target_season:", target_season, "| TARGET_WEEK override:", target_week_override)

    # Carga
    odds_cur = load_current_odds()
    stats_all = load_pregame_stats_for_seasons(target_season)

    # Odds solo de temporada target (+ semana si override)
    odds_cur = odds_cur[odds_cur["season"].eq(target_season)].copy()
    if target_week_override is not None:
        odds_cur = odds_cur[odds_cur["week"].eq(target_week_override)].copy()

    # Master target
    master_target = build_master(odds_cur, stats_all)
    dprint("Master target rows:", len(master_target))

    # Auditorías: confirmar que cada partido tiene pregame
    pre_cols_home = [c for c in master_target.columns if c.startswith("home_") and c.endswith(("_pre_ytd","_pre_ewm","_pre_l8"))]
    pre_cols_away = [c for c in master_target.columns if c.startswith("away_") and c.endswith(("_pre_ytd","_pre_ewm","_pre_l8"))]
    dprint("Audit: columnas pregame presentes:", len(set(pre_cols_home+pre_cols_away)), "OK")

    miss_home = master_target[pre_cols_home].isna().all(axis=1).sum()
    miss_away = master_target[pre_cols_away].isna().all(axis=1).sum()
    dprint(f"Audit: filas sin pregame (home)={miss_home} / (away)={miss_away} de {len(master_target)}")

    if "week" in master_target.columns and master_target["week"].isna().any():
        dprint("Audit: 'week' NA rows:", master_target["week"].isna().sum())
    else:
        dprint("Audit: 'week' NA rows:", 0)

    if len(master_target):
        sample_cols = ["season","week","week_label","home_team","away_team","schedule_date",
                       "home_off_epa_per_play_pre_ytd","away_off_epa_per_play_pre_ytd",
                       "home_def_epa_allowed_pre_ytd","away_def_epa_allowed_pre_ytd"]
        dprint("Audit: sample de 5 juegos con pregame:\n", master_target[sample_cols].head(5).to_string(index=False))

    # Cargar odds históricas (4 temporadas previas)
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
    dprint("Post-merge odds+pred rows:", len(dfm), "| juegos target:", dfm[["season","week","home_team","away_team"]].drop_duplicates().shape[0])

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

    # Limpieza/sanitizado y devig
    ev_base = sanitize_ev(ev_predictions, CFG)
    ev_base["game_id"] = ev_base.apply(lambda r: make_game_id_row(r["week_label"], r["team"], r["opponent"]), axis=1)
    ev_devig = devig_per_game(ev_base.copy(), single_side_mode=CFG["DEVIG_SINGLE_SIDE"])

    p = ev_devig["model_prob"].astype(float)
    d = ev_devig["decimal_odds"].astype(float)
    mkt = ev_devig["market_prob_nv"].astype(float)
    ev_devig["edge"] = p - mkt
    ev_devig["ev"]   = p*(d-1) - (1-p)

    # Candidatos
    cand = pick_per_game_best(ev_devig, CFG)

    # --------- BANKROLL POR SEMANA (desde bets.csv existente) ----------
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

    planned_weeks = sorted(cand["week_label"].astype(str).unique(), key=lambda wl: ORDER_INDEX.get(wl, 999))
    # Semanas finalizadas (con week_is_final True o status_short == 'Final') quedan frozen
    frozen_weeks = []
    if not existing.empty and "week_is_final" in existing.columns:
        frozen_weeks = sorted(existing.loc[existing["week_is_final"]==True, "week_label"].dropna().astype(str).unique(),
                              key=lambda wl: ORDER_INDEX.get(wl, 999))
    elif not existing.empty and "status_short" in existing.columns:
        frozen_weeks = sorted(existing.loc[existing["status_short"].astype(str).str.lower().eq("final"),
                                           "week_label"].dropna().astype(str).unique(),
                              key=lambda wl: ORDER_INDEX.get(wl, 999))

    dprint("Weeks planificadas:", planned_weeks)
    dprint("Weeks finalizadas (frozen):", frozen_weeks)

    modifiable_weeks = [wk for wk in planned_weeks if wk not in frozen_weeks]
    dprint("Weeks modificables (upsert):", modifiable_weeks if modifiable_weeks else "[]")

    bk_by_week = build_bk_by_week(existing, planned_weeks, CFG)
    dprint("BK0 por semana (prev o inicial):", bk_by_week)

    # --------- PLAN PREGAME (usando BK0 por semana) ----------
    planned_bets_raw = cand.copy()
    planned_bets, weekly_plan, S = plan_pregame_stakes(planned_bets_raw, CFG, bk_by_week)

    # UPSERT por semanas modificables
    combined = upsert_by_week(existing, planned_bets, modifiable_weeks)

    # Quitar columna 'bankroll' si existiera por versiones previas
    combined = combined.drop(columns=["bankroll"], errors="ignore")

    combined.to_csv(BETS_OUT_PATH, index=False)
    print(f"[select_bets] upsert wrote {BETS_OUT_PATH} | prev_rows={len(existing)} | total={len(combined)} | weeks_upserted={modifiable_weeks}")

if __name__ == "__main__":
    main()
