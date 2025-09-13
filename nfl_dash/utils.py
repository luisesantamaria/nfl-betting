from datetime import datetime, timedelta, timezone
import pandas as pd
from .config import SEASON_RULES

ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX  = {lab: i for i, lab in enumerate(ORDER_LABELS)}
PLAYOFF_LABEL_TO_NUM = {"Wild Card":19,"Divisional":20,"Conference":21,"Super Bowl":22}

TEAM_FIX = {
    "STL":"LA","LAR":"LA","LA":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV","WSH":"WAS","JAC":"JAX",
    "GNB":"GB","KAN":"KC","NWE":"NE","NOR":"NO","SFO":"SF","TAM":"TB",
}

def norm_abbr(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().upper()
    return TEAM_FIX.get(s, s)

def add_week_order(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["week_label"] = x["week_label"].astype(str)
    x["__order"] = x["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    x = x.sort_values("__order").drop(columns="__order")
    x["week_label"] = pd.Categorical(x["week_label"], categories=ORDER_LABELS, ordered=True)
    return x

def week_label_to_num(val) -> int:
    if pd.isna(val): return 999
    s = str(val)
    if s.startswith("Week "):
        try: return int(s.split(" ")[1])
        except: return 999
    return PLAYOFF_LABEL_TO_NUM.get(s, 999)

def kpis_from_pnl(df: pd.DataFrame):
    profits = pd.to_numeric(df.get("profit"), errors="coerce").fillna(0.0)
    stakes  = pd.to_numeric(df.get("stake"),  errors="coerce").fillna(0.0)
    banks   = pd.to_numeric(df.get("bankroll"), errors="coerce")
    first_bankroll   = float(banks.iloc[0]); first_profit = float(profits.iloc[0])
    initial_bankroll = float(first_bankroll - first_profit)
    final_bankroll   = float(banks.iloc[-1])
    total_profit     = float(profits.sum())
    total_stake      = float(stakes.sum())
    yield_pct        = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0
    return initial_bankroll, final_bankroll, total_profit, total_stake, yield_pct, profits, stakes

def american_to_decimal(v):
    try: v = float(v)
    except: return float("nan")
    return 1 + (100/abs(v) if v < 0 else v/100)

def decimal_to_american(d):
    try: d = float(d)
    except: return float("nan")
    return round((d - 1) * 100, 0) if d >= 2.0 else round(-100 / (d - 1), 0)

def season_stage(year: int, pnl_df: pd.DataFrame) -> str:
    rule = SEASON_RULES.get(year, {})
    now = datetime.now(timezone.utc)
    if not pnl_df.empty:
        labels = set(map(str, pnl_df["week_label"].astype(str).unique()))
        if "Super Bowl" in labels or "Conference" in labels:
            return "ended"
    start = datetime.fromisoformat(rule.get("season_start", f"{year}-09-05")).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(rule.get("season_end",   f"{year+1}-02-12")).replace(tzinfo=timezone.utc)
    activate_from = start - timedelta(days=int(rule.get("activate_days_before", 0)))
    if now < activate_from: return "locked"
    if now < start: return "preseason"
    if now <= end: return "in_season"
    return "ended"
