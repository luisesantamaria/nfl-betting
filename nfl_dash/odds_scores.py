import pandas as pd
import streamlit as st
from .paths import ARCHIVE_DIR
from .utils import norm_abbr, week_label_to_num

def _coerce_score_cols(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # acepta variantes: home_score/away_score → score_home/score_away
    if "score_home" not in x.columns and "home_score" in x.columns:
        x = x.rename(columns={"home_score": "score_home"})
    if "score_away" not in x.columns and "away_score" in x.columns:
        x = x.rename(columns={"away_score": "score_away"})
    return x

def _ensure_wk_num(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "wk_num" in x.columns:
        return x
    if "week" in x.columns and pd.api.types.is_numeric_dtype(x["week"]):
        x["wk_num"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
    elif "week" in x.columns:
        x["wk_num"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
        if x["wk_num"].isna().all() and "week_label" in x.columns:
            x["wk_num"] = x["week_label"].apply(week_label_to_num).astype("Int64")
    elif "week_label" in x.columns:
        x["wk_num"] = x["week_label"].apply(week_label_to_num).astype("Int64")
    else:
        x["wk_num"] = pd.Series([None] * len(x), dtype="Int64")
    return x

@st.cache_data
def load_scores_table(year: int) -> pd.DataFrame:
    p = ARCHIVE_DIR / f"season={year}" / "odds.csv"
    if not p.exists():
        return pd.DataFrame(
            columns=[
                "season","wk_num","pair","home_team","away_team",
                "score_home","score_away","schedule_date"
            ]
        )

    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame(
            columns=[
                "season","wk_num","pair","home_team","away_team",
                "score_home","score_away","schedule_date"
            ]
        )

    df = _coerce_score_cols(df).copy()

    # normaliza equipos
    for c in ("home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)

    # season fija al año
    df["season"] = year

    # week/label → wk_num
    df = _ensure_wk_num(df)

    # fecha
    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")

    # clave
    df["pair"] = [
        f"{a}_{b}" if a < b else f"{b}_{a}"
        for a, b in zip(df.get("home_team", ""), df.get("away_team", ""))
    ]

    keep = [
        "season","wk_num","pair","home_team","away_team",
        "score_home","score_away","schedule_date"
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].drop_duplicates(subset=["season","wk_num","pair"], keep="last").reset_index(drop=True)
    return out

def _pair_from_row(row: pd.Series) -> str:
    a = norm_abbr(row.get("team", "")) or norm_abbr(row.get("home_team", ""))
    b = norm_abbr(row.get("opponent", "")) or norm_abbr(row.get("away_team", ""))
    if not a and not b:
        return ""
    if a and b:
        return f"{a}_{b}" if a < b else f"{b}_{a}"
    return ""

@st.cache_data
def enrich_bets_with_scores(bets_df: pd.DataFrame, year: int) -> pd.DataFrame:
    if bets_df.empty:
        return bets_df

    b = bets_df.copy()

    # season
    if "season" not in b.columns:
        b["season"] = year
    else:
        b["season"] = pd.to_numeric(b["season"], errors="coerce").fillna(year).astype(int)

    # normaliza equipos
    for c in ("team","opponent","home_team","away_team"):
        if c in b.columns:
            b[c] = b[c].astype(str).map(norm_abbr)

    # week → wk_num
    b = _ensure_wk_num(b)

    # pair
    b["pair"] = b.apply(_pair_from_row, axis=1)

    scores = load_scores_table(year)
    if scores.empty:
        return b

    merged = b.merge(
        scores,
        on=["season","wk_num","pair"],
        how="left",
        suffixes=("","_sc"),
        validate="m:1"
    )

    if "schedule_date" not in merged.columns and "schedule_date_sc" in merged.columns:
        merged["schedule_date"] = merged["schedule_date_sc"]

    for c in ("home_team","away_team"):
        if c not in merged.columns and f"{c}_sc" in merged.columns:
            merged[c] = merged[f"{c}_sc"]

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_sc")], errors="ignore")
    return merged
