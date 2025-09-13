import pandas as pd
import streamlit as st
from .paths import ODDS_DIRS
from .utils import norm_abbr, week_label_to_num

def _candidate_odds_files(year: int):
    names = [
        f"odds_season_{year}.csv",
        f"target_odds_{year}.csv",
        f"targets_odds_{year}.csv",
        "target_odds.csv",
        "targets_odds.csv",
        "historical_odds.csv",
    ]
    for d in ODDS_DIRS:
        for n in names:
            p = d / n
            if p.exists(): yield p
        for p in d.glob(f"*odds*{year}*.csv"):
            yield p

@st.cache_data
def load_scores_table(year: int) -> pd.DataFrame:
    frames = []
    seen = set()
    for p in _candidate_odds_files(year):
        key = p.resolve().as_posix()
        if key in seen: continue
        seen.add(key)
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        need = {"home_team","away_team","week","season"}
        if not need.issubset(set(df.columns)): continue

        df = df.copy()
        for c in ("home_team","away_team"):
            df[c] = df[c].astype(str).map(norm_abbr)
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        df["week"]   = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
        if "schedule_date" in df.columns:
            df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")

        df = df[df["season"].astype("Int64").eq(year)]
        if df.empty: continue

        df["pair"] = [f"{a}_{b}" if a < b else f"{b}_{a}" for a,b in zip(df["home_team"], df["away_team"])]

        keep = ["season","week","home_team","away_team","pair",
                "score_home","score_away","schedule_date"]
        keep = [c for c in keep if c in df.columns]
        frames.append(df[keep])

    if not frames:
        return pd.DataFrame(columns=["season","week","home_team","away_team","pair","score_home","score_away","schedule_date"])

    out = pd.concat(frames, ignore_index=True)
    out = (out.sort_values(["season","week","schedule_date"])
              .drop_duplicates(subset=["season","week","pair"], keep="last"))
    return out

def ensure_week_num_column(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "wk_num" not in x.columns:
        if "week" in x.columns and pd.api.types.is_numeric_dtype(x["week"]):
            x["wk_num"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
        elif "week_label" in x.columns:
            x["wk_num"] = x["week_label"].apply(week_label_to_num).astype("Int64")
        else:
            x["wk_num"] = pd.Series([None]*len(x), dtype="Int64")
    return x

@st.cache_data
def enrich_bets_with_scores(bets_df: pd.DataFrame, year: int) -> pd.DataFrame:
    if bets_df.empty: return bets_df
    b = bets_df.copy()
    for c in ("team","opponent"):
        if c in b.columns:
            b[c] = b[c].astype(str).map(norm_abbr)
    b = ensure_week_num_column(b)
    b["pair"] = [f"{a}_{o}" if a < o else f"{o}_{a}" for a,o in zip(b.get("team",""), b.get("opponent",""))]

    scores = load_scores_table(year)
    if scores.empty: return b

    merged = b.merge(
        scores,
        left_on=["season","wk_num","pair"],
        right_on=["season","week","pair"],
        how="left",
        suffixes=("","_sc")
    )
    if "schedule_date" not in b.columns and "schedule_date_sc" in merged.columns:
        merged["schedule_date"] = merged["schedule_date_sc"]
    for c in ("home_team","away_team"):
        if c not in merged.columns and f"{c}_sc" in merged.columns:
            merged[c] = merged[f"{c}_sc"]

    drop_cols = [c for c in merged.columns if c.endswith("_sc")] + ["week_y"]
    merged = merged.rename(columns={"week_x":"week"}).drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore")
    return merged
