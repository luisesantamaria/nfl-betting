import streamlit as st
import pandas as pd
from .logos import get_logo_url
from .utils import norm_abbr, decimal_to_american

YELLOW_LIVE = "#FBBF24"   # punto amarillo para LIVE
GREEN_WIN   = "#1FA37C"
RED_LOSS    = "#D64545"
GREY_NEUT   = "#888888"
PURPLE_PUSH = "#8884D8"


def bet_card(row: pd.Series):
    # Abreviaturas/hard data
    htm = norm_abbr(row.get("home_team", "")) or norm_abbr(row.get("team", ""))
    atm = norm_abbr(row.get("away_team", "")) or norm_abbr(row.get("opponent", ""))
    side = str(row.get("side", "")).lower().strip() if pd.notna(row.get("side")) else ""

    # Deducción del pick (a quién le apostaste) como abbr
    pick_abbr = ""
    if side == "home":
        pick_abbr = htm
    elif side == "away":
        pick_abbr = atm
    elif side in {"team", "moneyline", "ml"}:
        # fallback si tu pipeline usa "team" como lado genérico
        pick_abbr = norm_abbr(row.get("team", ""))

    # Numeric
    wl_profit = pd.to_numeric(row.get("profit"), errors="coerce")
    stake     = pd.to_numeric(row.get("stake"), errors="coerce")
    dec       = pd.to_numeric(row.get("decimal_odds"), errors="coerce")
    ml        = pd.to_numeric(row.get("ml"), errors="coerce")
    if pd.isna(ml) and pd.notna(dec):
        ml = decimal_to_american(dec)

    # Estado LIVE/FINAL/OPEN
    state = str(row.get("state", "")).lower().strip() if pd.notna(row.get("state")) else ""
    short = str(row.get("status_short", "")).strip() if pd.notna(row.get("status_short")) else ""

    if state == "in":                        # juego en vivo
        status_color, status_label = YELLOW_LIVE, (short or "LIVE")
    elif pd.isna(wl_profit):                 # sin profit calculado -> abierto
        status_color, status_label = GREY_NEUT, "OPEN"
    elif wl_profit > 0:
        status_color, status_label = GREEN_WIN, "WIN"
    elif wl_profit < 0:
        status_color, status_label = RED_LOSS, "LOSS"
    else:
        status_color, status_label = PURPLE_PUSH, "PUSH"

    # Encabezado
    wk = str(row.get("week_label", row.get("week", "")))
    date_txt = str(row.get("schedule_date", ""))[:10] if pd.notna(row.get("schedule_date")) else ""

    # Score (si existe)
    sh = row.get("home_score", None)
    sa = row.get("away_score", None)
    has_score = pd.notna(sh) and pd.notna(sa)

    def logo_tag(team_abbr: str, fallback_bg: str = "#222", size: int = 44):
        url = get_logo_url(team_abbr) if team_abbr else None
        if url:
            return f"<img src='{url}' width='{size}' height='{size}' style='object-fit:contain;'/>"
        t = (team_abbr or 'NA')[:3]
        return (
            f"<div style='width:{size}px;height:{size}px;border-radius:50%;"
            f"background:{fallback_bg};color:#fff;display:flex;align-items:center;"
            f"justify-content:center;font-weight:800;'>{t}</div>"
        )

    left_logo  = logo_tag(htm, "#1f2937", 44)
    right_logo = logo_tag(atm, "#374151", 44)

    score_html = (
        f"<div style='font-weight:900;font-size:26px;letter-spacing:.5px;'>{int(sh)} — {int(sa)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>TBD</div>"
    )

    subline_left  = f"{htm}{(' • ' + ('Away' if side=='away' else 'Home')) if side in {'home','away'} else ''}".strip()
    subline_right = f"{atm}".strip()

    ml_txt    = f"{ml:+.0f}" if pd.notna(ml) else "—"
    stake_txt = f"${stake:,.2f}" if pd.notna(stake) else "—"
    prof_txt  = f"${wl_profit:,.2f}" if pd.notna(wl_profit) else "—"

    # Pill "Pick"
    pick_html = (
        f"<span style='padding:.15rem .5rem;border-radius:999px;border:1px solid rgba(255,255,255,.1);"
        f"background:rgba(255,255,255,.06);font-weight:700;'>Pick: {pick_abbr}</span>"
        if pick_abbr else ""
    )

    # Render
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
          {"<div style='font-size:11px;opacity:.75;margin-top:2px;font-weight:700;'>"+short+"</div>" if (state=='in' and short) else ""}
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

      <div style="display:flex; align-items:center; gap:12px; margin-top:10px; white-space:nowrap; flex-wrap:nowrap;">
        {pick_html}
        <div><span style="opacity:.6;">ML:</span> <strong>{ml_txt}</strong></div>
        <div><span style="opacity:.6;">Stake:</span> <strong>{stake_txt}</strong></div>
        <div><span style="opacity:.6;">Profit:</span> <strong style="color:{status_color};">{prof_txt}</strong></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
