from __future__ import annotations
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_bets_for_season
from nfl_dash.utils import logo_url, ORDER_INDEX

def _color_for_profit(x: float) -> str:
    return "#0f9d58" if x > 0 else ("#d93025" if x < 0 else "#999999")

def render(season: int) -> None:
    st.subheader("Bets", anchor=False)

    df = load_bets_for_season(season)
    if df.empty:
        st.info("No hay bets para esta temporada.")
        return

    if "week_label" in df.columns:
        df["__order"] = df["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
        df = df.sort_values(["__order", "schedule_date", "team", "opponent"], na_position="last")

    n_cols = 3
    cols = st.columns(n_cols, gap="small")

    for i, (_, r) in enumerate(df.iterrows()):
        c = cols[i % n_cols]
        with c:
            home = str(r.get("home_team") or "")
            away = str(r.get("away_team") or "")
            team = str(r.get("team") or "")
            opp  = str(r.get("opponent") or "")

            # pick logos: si r['side']=="home", team es home, etc.
            left_team  = team
            right_team = opp

            l_logo = logo_url(left_team) or logo_url(home) or logo_url(team)
            r_logo = logo_url(right_team) or logo_url(away) or logo_url(opp)

            week = r.get("week_label") or ""
            sched = r.get("schedule_date")
            sched_str = pd.to_datetime(sched).strftime("%b %d, %Y") if pd.notna(sched) else ""

            profit = float(r.get("profit", 0) or 0)
            stake  = float(r.get("stake", 0) or 0)
            odds   = r.get("ml") if pd.notna(r.get("ml")) else r.get("decimal_odds")
            odds_str = f"ML {int(odds):+d}" if pd.notna(odds) and not isinstance(odds, float) else f"Dec {odds:.2f}" if pd.notna(odds) else "—"

            score_h = r.get("score_home")
            score_a = r.get("score_away")
            score_str = ""
            if pd.notna(score_h) and pd.notna(score_a):
                try:
                    score_str = f"{int(score_h)}–{int(score_a)}"
                except Exception:
                    score_str = f"{score_h}–{score_a}"

            bg = "#111418"
            border = "#2a2f36"
            winbar = _color_for_profit(profit)

            st.markdown(
                f"""
                <div style="
                  border:1px solid {border};
                  border-radius:14px;
                  padding:12px 12px 10px;
                  background:{bg};
                ">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <div style="font-size:12px;color:#9aa0a6;">{week or ''}</div>
                    <div style="font-size:12px;color:#9aa0a6;">{sched_str}</div>
                  </div>

                  <div style="display:flex;gap:10px;align-items:center;justify-content:space-between;">
                    <div style="display:flex;align-items:center;gap:8px;min-width:0;">
                      <img src="{l_logo or ''}" style="width:38px;height:38px;object-fit:contain;" />
                      <div style="font-weight:600;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{left_team}</div>
                    </div>

                    <div style="text-align:center;min-width:80px;">
                      <div style="font-weight:700;font-size:20px;letter-spacing:0.2px;">{score_str or ''}</div>
                      <div style="font-size:11px;color:#9aa0a6;">{odds_str}</div>
                    </div>

                    <div style="display:flex;align-items:center;gap:8px;min-width:0;justify-content:flex-end;">
                      <div style="font-weight:600;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:right;">{right_team}</div>
                      <img src="{r_logo or ''}" style="width:38px;height:38px;object-fit:contain;" />
                    </div>
                  </div>

                  <div style="display:flex;justify-content:space-between;margin-top:10px;">
                    <div style="font-size:12px;color:#9aa0a6;">Stake</div>
                    <div style="font-size:12px;color:#e8eaed;">${stake:,.2f}</div>
                  </div>
                  <div style="display:flex;justify-content:space-between;margin-top:6px;">
                    <div style="font-size:12px;color:#9aa0a6;">Profit</div>
                    <div style="font-size:12px;color:{winbar};font-weight:600;">${profit:,.2f}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
