import pandas as pd
import streamlit as st


def apply_dashboard_css():
    st.markdown(
        """
        <style>
        .main { background-color: #0e1117; }

        .stMetric {
            background-color: #1a1c24;
            padding: 16px 18px;
            border-radius: 12px;
            border: 1px solid #2d3748;
        }
        div[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #9aa4b2; }

        .status-pill {
            display:inline-block; padding:4px 14px; border-radius:999px;
            font-size:0.78rem; font-weight:600; letter-spacing:.02em;
        }
        .pill-long   { background:rgba(34,197,94,0.15); color:#4ade80; border:1px solid rgba(74,222,128,0.35); }
        .pill-flat   { background:rgba(148,163,184,0.12); color:#94a3b8; border:1px solid rgba(148,163,184,0.3); }
        .pill-paused { background:rgba(250,204,21,0.12); color:#fbbf24; border:1px solid rgba(250,204,21,0.35); }

        section[data-testid="stSidebar"] { border-right: 1px solid #2d3748; }

        div[data-testid="stExpander"] {
            background-color:#151822; border:1px solid #2d3748; border-radius:10px;
        }

        hr { border-color:#2d3748; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_row(equity_now, initial_capital, sharpe, max_dd, win_rate, n_trades):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("มูลค่าพอร์ตปัจจุบัน", f"{equity_now:,.2f}", f"{equity_now - initial_capital:+,.2f}")
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c3.metric("Max Drawdown", f"{max_dd:.2f}%")
    c4.metric("Win Rate", f"{win_rate:.1f}%")
    c5.metric("จำนวนออเดอร์ที่ปิดแล้ว", f"{n_trades}")


def render_equity_and_drawdown(equity_series, index=None):
    equity_series = pd.Series(equity_series)
    if index is not None:
        equity_series.index = index
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**เส้นมูลค่าพอร์ต (Equity Curve)**")
        st.line_chart(equity_series, height=240)
    with col_b:
        _, drawdown = __import__("strategy").compute_max_drawdown(equity_series)
        st.markdown("**Drawdown (%)**")
        if not drawdown.empty:
            st.area_chart(drawdown, height=240, color="#f87171")
        else:
            st.info("ยังไม่มีข้อมูลพอสำหรับคำนวณ Drawdown")


def download_logs_button(logs_df, filename):
    if logs_df is not None and not logs_df.empty:
        csv = logs_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ ส่งออกผลเทรดเป็น CSV", data=csv, file_name=filename, mime="text/csv")
