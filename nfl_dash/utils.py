from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
from typing import Dict, Any

from .config import SEASON_RULES  # puede estar vacío; calculamos defaults si falta


# ---------------- Week labels / ordering ----------------
ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + [
    "Wild Card", "Divisional", "Conference", "Super Bowl"
]
ORDER_INDEX: Dict[str, int] = {lab: i for i, lab in enumerate(ORDER_LABELS)}


def week_label_from_num(n) -> str:
    try:
        n = int(n)
    except Exception:
        return "Week 999"
    if 1 <= n <= 18:
        return f"Week {n}"
    return {19: "Wild Card", 20: "Divisional", 21: "Conference", 22: "Super Bowl"}.get(n, f"Week {n}")


# ---------------- Odds helpers ----------------
def american_to_decimal(ml) -> float | None:
    try:
        x = float(ml)
    except Exception:
        return None
    if x == 0 or math.isnan(x):
        return None
    if x > 0:
        return 1.0 + (x / 100.0)
    return 1.0 + (100.0 / abs(x))


# ---------------- Team name normalization ----------------
_ABBR_MAP = {
    # AFC
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "denver broncos": "DEN",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "miami dolphins": "MIA",
    "new england patriots": "NE",
    "new york jets": "NYJ",
    "pittsburgh steelers": "PIT",
    "tennessee titans": "TEN",
    # NFC
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "dallas cowboys": "DAL",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "los angeles rams": "LAR",
    "minnesota vikings": "MIN",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "philadelphia eagles": "PHI",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "washington commanders": "WAS",
    # históricos / variantes
    "washington football team": "WAS",
    "oakland raiders": "LV",
    "san diego chargers": "LAC",
    "st. louis rams": "LAR",
    "st louis rams": "LAR",
}
def norm_abbr(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).strip().lower()
    return _ABBR_MAP.get(key, name if len(name) <= 4 else name)  # si ya es abreviación, la deja


# ---------------- Season stage ----------------
def _first_monday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    while d.weekday() != 0:  # 0=Monday
        d += pd.Timedelta(days=1)
    return d


def _first_thursday_on_or_after(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts
    while d.weekday() != 3:  # 3=Thursday
        d += pd.Timedelta(days=1)
    return d


def _second_sunday_feb(year: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=2, day=1, tz="UTC")
    # first Sunday
    while d.weekday() != 6:
        d += pd.Timedelta(days=1)
    # second Sunday
    d += pd.Timedelta(days=7)
    # Super Bowl evening; usamos fin de día para no cortar antes
    return d.replace(hour=23, minute=59, second=59)


def _default_bounds(year: int) -> Dict[str, pd.Timestamp]:
    labor_day = _first_monday(year, 9)  # primer lunes de septiembre
    kickoff_thu = _first_thursday_on_or_after(labor_day)
    regular_start = kickoff_thu.normalize()
    post_end = _second_sunday_feb(year + 1)
    return {"regular_start": regular_start, "post_end": post_end}


def season_stage(year: int, now_utc: pd.Timestamp | None = None) -> str:
    if now_utc is None:
        now_utc = pd.Timestamp.utcnow().tz_localize("UTC")

    rules: Dict[str, Any] = SEASON_RULES.get(year, {})
    if "regular_start" in rules and pd.notna(rules["regular_start"]):
        regular_start = pd.to_datetime(rules["regular_start"], utc=True)
    else:
        regular_start = _default_bounds(year)["regular_start"]

    if "post_end" in rules and pd.notna(rules["post_end"]):
        post_end = pd.to_datetime(rules["post_end"], utc=True)
    else:
        post_end = _default_bounds(year)["post_end"]

    if now_utc < regular_start:
        return "Preseason"
    if now_utc <= post_end:
        return "In Season"
    return "Season Ended"
