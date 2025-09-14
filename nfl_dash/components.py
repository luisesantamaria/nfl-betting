import streamlit as st
import pandas as pd

def _pill(text: str, color_bg: str, color_fg: str = "#fff"):
    st.markdown(
        f"""
        <span style="
            display:inline-block;
            padding:2px 8px;
            border-radius:10px;
            background:{color_bg};
            color:{color_fg};
            font-size:12px;
            font-weight:600;">
            {text}
        </span>
        """,
        unsafe_allow_html=True,
    )

def bet_card(row: pd.Series):
    # row: espera campos típicos: week_label, team, opponent, scoreline, ml/decimal_odds, stake, profit, won, schedule_date
    with st.container(border=True):
        top = st.columns([0.5, 0.5, 0.5])
        with top[0]:
            st.caption(str(row.get("week_label") or ""))
        with top[1]:
            status = row.get("won")
            if pd.isna(status):
                _pill("PENDING", "#6c757d")
            elif int(status) == 1:
                _pill("WIN", "#1a7f37")
            else:
                _pill("LOSS", "#c92a2a")
        with top[2]:
            dt = row.get("schedule_date")
            if pd.notna(dt):
                st.caption(str(dt))

        # marcador grande en una sola línea
        team = str(row.get("team") or "")
        opp  = str(row.get("opponent") or "")
        score = row.get("scoreline")
        st.markdown(
            f"""
            <div style="display:flex;justify-content:center;align-items:center;gap:8px;">
                <div style="font-weight:700;font-size:18px;">{team}</div>
                <div style="font-weight:700;font-size:20px;">{score if score else "—"}</div>
                <div style="font-weight:700;font-size:18px;">{opp}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # línea secundaria (moneyline y edge/ev si existe)
        ml = row.get("ml")
        dec = row.get("decimal_odds")
        odds_txt = f"Moneyline {int(ml)}" if pd.notna(ml) else (f"@ {dec:.2f}" if pd.notna(dec) else "")
        sub = []
        if odds_txt: sub.append(odds_txt)
        if pd.notna(row.get("edge")): sub.append(f"Edge {row.get('edge'):.3f}")
        if pd.notna(row.get("ev")):   sub.append(f"EV {row.get('ev'):.3f}")
        if sub:
            st.caption(" • ".join(sub))

        # pie: stake / profit
        foot = st.columns(2)
        with foot[0]:
            if pd.notna(row.get("stake")):
                st.metric("Stake", f"${float(row.get('stake')):,.2f}")
        with foot[1]:
            if pd.notna(row.get("profit")):
                val = float(row.get("profit"))
                delta = f"{val:,.2f}"
                st.metric("Profit", f"${val:,.2f}", delta=None)
