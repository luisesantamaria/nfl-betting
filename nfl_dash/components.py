# nfl_dash/components.py
from __future__ import annotations
import pandas as pd
import streamlit as st

# --- helpers ---
def decimal_to_american(dec: float | int | None) -> float | None:
    try:
        d = float(dec)
    except Exception:
        return None
    if d <= 1.0:
        return None
    if d >= 2.0:
        return round((d - 1.0) * 100.0)
    return round(-100.0 / (d - 1.0))

def norm_abbr(x: str) -> str:
    if x is None:
        return ""
    return str(x).strip()

# Si tienes un módulo logos, úsalo; si no, usa fallback (círculo con siglas)
try:
    from .logos import get_logo_url as _get_logo_url  # opcional
except Exception:
    _get_logo_url = None

def get_logo_url(team_abbr: str) -> str | None:
    if _get_logo_url:
        try:
            return _get_logo_url(team_abbr)
        except Exception:
            return None
    return None  # fallback: sin logos (se verá el círculo con siglas)

# --- UI card ---
def bet_card(row: pd.Series):
    htm = norm_abbr(row.get("home_team", "")) or norm_abbr(row.get("team", ""))
    atm = norm_abbr(row.get("away_team", "")) or norm_abbr(row.get("opponent", ""))
    side = str(row.get("side", "")).title() if pd.notna(row.get("side")) else ""

    wl_profit = pd.to_numeric(row.get("profit"), errors="coerce")
    stake     = pd.to_numeric(row.get("stake"), errors="coerce")
    dec       = pd.to_numeric(row.get("decimal_odds"), errors="coerce")
    ml        = pd.to_numeric(row.get("ml"), errors="coerce")
    if pd.isna(ml) and pd.notna(dec):
        ml = decimal_to_american(dec)

    if pd.isna(wl_profit): status_color, status_label = "#888888", "OPEN"
    elif wl_profit > 0:    status_color, status_label = "#1FA37C", "WIN"
    elif wl_profit < 0:    status_color, status_label = "#D64545", "LOSS"
    else:                  status_color, status_label = "#8884D8", "PUSH"

    wk = str(row.get("week_label", row.get("week", "")))
    date_txt = str(row.get("schedule_date", ""))[:10] if pd.notna(row.get("schedule_date")) else ""

    sh = pd.to_numeric(row.get("score_home"), errors="coerce")
    sa = pd.to_numeric(row.get("score_away"), errors="coerce")
    has_score = pd.notna(sh) and pd.notna(sa)

    def logo_tag(team_abbr: str, fallback_bg: str = "#222"):
        url = get_logo_url(team_abbr) if team_abbr else None
        if url:
            return f"<img src='{url}' width='44' height='44' style='object-fit:contain;'/>"
        t = (team_abbr or 'NA')[:3]
        return (
            f"<div style='width:44px;height:44px;border-radius:50%;"
            f"background:{fallback_bg};color:#fff;display:flex;align-items:center;"
            f"justify-content:center;font-weight:800;'>{t}</div>"
        )

    left_logo  = logo_tag(htm, "#1f2937")
    right_logo = logo_tag(atm, "#374151")

    score_html = (
        f"<div style='font-weight:900;font-size:26px;letter-spacing:.5px;'>{int(sh)} — {int(sa)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>TBD</div>"
    )

    subline_left  = f"{htm}{(' • '+side) if side else ''}".strip()
    subline_right = f"{atm}".strip()

    ml_txt    = f"{ml:+.0f}" if pd.notna(ml) else "—"
    stake_txt = f\"${stake:,.2f}\" if pd.notna(stake) else "—"
    prof_txt  = f\"${wl_profit:,.2f}\" if pd.notna(wl_profit) else "—"

    st.markdown(f"""
    <div style="
        border:1px solid #e9e9e9;border-radius:12px;padding:12px; margin-bottom:10px;
        background:linear-gradient(180deg, rgba(0,0,0,0.02), rgba(0,0,0,0.00));
        font-size:12.5px;
    ">
      <div style="display:flex; align-items:center; justify-content:space-between;">
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:8px;height:8px;border-radius:50%;background:{status_color};"></div>
          <div style="font-weight:600;">{wk}</div>
          <div style="opacity:.7;">{date_txt}</div>
        </div>
        <div style="font-weight:800;color:{status_color}">{status_label}</div>
      </div>

      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:10px;">
        <div style="width:44px; display:flex; align-items:center; justify-content:center;">
          {left_logo}
        </div>
        <div style="flex:1; text-align:center;">
          {score_html}
        </div>
        <div style="width:44px; display:flex; align-items:center; justify-content:center;">
          {right_logo}
        </div>
      </div>

      <div style="display:flex; align-items:center; justify-content:space-between; margin-top:6px;">
        <div style="font-size:12px; opacity:.7; font-weight:600;">{subline_left}</div>
        <div style="font-size:12px; opacity:.5; font-weight:700;">vs</div>
        <div style="font-size:12px; opacity:.7; font-weight:600; text-align:right;">{subline_right}</div>
      </div>

      <div style="display:flex; gap:14px; margin-top:10px; flex-wrap:wrap;">
        <div><span style="opacity:.6;">Moneyline:</span> <strong>{ml_txt}</strong></div>
        <div><span style="opacity:.6;">Stake:</span> <strong>{stake_txt}</strong></div>
        <div><span style="opacity:.6;">Profit:</span> <strong style="color:{status_color};">{prof_txt}</strong></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
