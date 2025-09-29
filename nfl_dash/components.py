import streamlit as st
import pandas as pd
from .logos import get_logo_url
from .utils import norm_abbr, decimal_to_american

def _safe_abbr(x) -> str:
    try:
        return norm_abbr(x) if pd.notna(x) else ""
    except Exception:
        return ""

def _fmt_money(x) -> str:
    try:
        v = float(x)
        return f"${v:,.2f}"
    except Exception:
        return "—"

def _status_color_and_label(profit):
    if pd.isna(profit):
        return ("#facc15", "LIVE/OPEN")
    profit = float(profit)
    if profit > 0:
        return ("#16a34a", "WIN")
    if profit < 0:
        return ("#ef4444", "LOSS")
    return ("#8b5cf6", "PUSH")

def _logo_tag(team_abbr: str, fallback_bg: str = "#222", size: int = 44) -> str:
    team_abbr = (team_abbr or "").strip()
    url = get_logo_url(team_abbr) if team_abbr else None
    if url:
        return f"<img src='{url}' alt='{team_abbr}' width='{size}' height='{size}' style='object-fit:contain;'/>"
    t = (team_abbr or 'NA')[:3]
    return (
        f"<div style='width:{size}px;height:{size}px;border-radius:50%;"
        f"background:{fallback_bg};color:#fff;display:flex;align-items:center;"
        f"justify-content:center;font-weight:800;'>{t}</div>"
    )

def bet_card(row: pd.Series):
    home = _safe_abbr(row.get("home_team", "")) or _safe_abbr(row.get("team", ""))
    away = _safe_abbr(row.get("away_team", "")) or _safe_abbr(row.get("opponent", ""))
    side = str(row.get("side", "")).lower() if pd.notna(row.get("side")) else ""

    pick_abbr = _safe_abbr(row.get("team", ""))
    if not pick_abbr:
        pick_abbr = home if side == "home" else away if side == "away" else ""

    sh = pd.to_numeric(row.get("score_home"), errors="coerce")
    sa = pd.to_numeric(row.get("score_away"), errors="coerce")
    short = str(row.get("short", ""))
    state = str(row.get("status", "")).upper()

    stake = pd.to_numeric(row.get("stake"), errors="coerce")
    dec = pd.to_numeric(row.get("decimal_odds"), errors="coerce")
    ml  = pd.to_numeric(row.get("ml"), errors="coerce")
    if pd.isna(ml) and pd.notna(dec):
        ml = decimal_to_american(dec)

    profit = pd.to_numeric(row.get("profit"), errors="coerce")
    status_color, status_label = _status_color_and_label(profit)

    wk = str(row.get("week_label", row.get("week", "")))
    date_txt = str(row.get("schedule_date", ""))[:10] if pd.notna(row.get("schedule_date")) else ""

    live = (state == "IN")
    dot_color = "#facc15" if live or pd.isna(profit) else ("#16a34a" if float(profit or 0) > 0 else "#ef4444")

    left_logo  = _logo_tag(home, "#1f2937", 44)
    right_logo = _logo_tag(away, "#374151", 44)

    has_score = pd.notna(sh) and pd.notna(sa)
    score_html = (
        f"<div style='font-weight:900;font-size:30px;letter-spacing:.5px;'>{int(sh)} — {int(sa)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:800;font-size:16px;'>TBD</div>"
    )
    short_html = f"<div style='font-size:12px;opacity:.65;font-weight:700;margin-top:4px;'>{short}</div>" if short else ""

    ml_txt    = f"{ml:+.0f}" if pd.notna(ml) else "—"
    stake_txt = _fmt_money(stake)
    prof_txt  = _fmt_money(profit)

    html = f"""
    <div style="border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:14px 14px 12px;margin-bottom:12px;background:rgba(255,255,255,.02);">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="display:flex;align-items:center;gap:10px;">
          <div style="width:10px;height:10px;border-radius:50%;background:{dot_color};"></div>
          <div style="font-size:18px;font-weight:800;">{wk}</div>
          <div style="opacity:.7">{date_txt}</div>
        </div>
        <div style="font-weight:900;letter-spacing:.4px;color:{status_color};">{status_label}</div>
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;gap:14px;margin-top:10px;">
        <div style="width:46px;display:flex;align-items:center;justify-content:center;">{left_logo}</div>
        <div style="flex:1;text-align:center;">{score_html}{short_html}</div>
        <div style="width:46px;display:flex;align-items:center;justify-content:center;">{right_logo}</div>
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;margin-top:6px;">
        <div style="font-size:12px;opacity:.8;font-weight:700;">{home} • Home</div>
        <div style="font-size:12px;opacity:.6;font-weight:800;">vs</div>
        <div style="font-size:12px;opacity:.8;font-weight:700;text-align:right;">{away} • Away</div>
      </div>

      <div style="display:flex;align-items:center;gap:10px;flex-wrap:nowrap;margin-top:10px;">
        <div style="display:flex;align-items:center;gap:6px;font-size:11px;">
          <span style="opacity:.7;">Pick:</span>
          <span style="display:inline-block;padding:.15rem .5rem;border-radius:999px;border:1px solid rgba(255,255,255,.18);font-weight:800;letter-spacing:.3px;font-size:11px;opacity:.95;">{pick_abbr or '—'}</span>
        </div>
        <div style="font-size:11.5px;"><span style="opacity:.7;">ML:</span> <strong>{ml_txt}</strong></div>
        <div style="font-size:11.5px;"><span style="opacity:.7;">Stake:</span> <strong>{stake_txt}</strong></div>
        <div style="font-size:11.5px;"><span style="opacity:.7;">Profit:</span> <strong style="color:{status_color};">{prof_txt}</strong></div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    return None

def game_card(row: pd.Series):
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    hs   = row.get("home_score", None)
    as_  = row.get("away_score", None)
    state = str(row.get("state","")).lower()
    short = str(row.get("short",""))

    live  = (state == "in")
    final = (state == "post")
    if live:
        color = "#facc15"; label = "LIVE"
    elif final:
        color = "#10B981"; label = "FINAL"
    else:
        color = "#6B7280"; label = short if short else "SCHEDULED"

    left_logo  = _logo_tag(home, "#1f2937", 44)
    right_logo = _logo_tag(away, "#374151", 44)

    has_score = pd.notna(hs) and pd.notna(as_)
    score_html = (
        f"<div style='font-weight:900;font-size:28px;letter-spacing:.5px;'>{int(hs)} — {int(as_)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>—</div>"
    )

    html = f"""
    <div style="border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:14px;margin-bottom:12px;background:rgba(255,255,255,.02);">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:10px;height:10px;border-radius:50%;background:{color};"></div>
          <div style="font-weight:800;">{home} vs {away}</div>
        </div>
        <div style="font-weight:900;color:{color}">{label}</div>
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;gap:14px;margin-top:12px;">
        <div style="width:46px;display:flex;align-items:center;justify-content:center;">{left_logo}</div>
        <div style="flex:1;text-align:center;">{score_html}</div>
        <div style="width:46px;display:flex;align-items:center;justify-content:center;">{right_logo}</div>
      </div>

      <div style="display:flex;align-items:center;justify-content:center;margin-top:8px;">
        <div style="font-size:12px;opacity:.75;font-weight:700;">{short}</div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    return None
