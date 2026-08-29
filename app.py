import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from strategy import run_strategy, compute_sharpe, compute_max_drawdown, compute_win_rate
from ui import apply_dashboard_css, render_metrics_row, render_equity_and_drawdown, download_logs_button
from tradingview_widget import render_tradingview_chart

# ============================================================
# 1. UI setup
# ============================================================
st.set_page_config(page_title="AI Trading Bot", layout="wide", page_icon="🤖")
apply_dashboard_css()

st.title("🤖 AI Trading Dashboard")
st.caption("ระบบทดสอบและจำลองเทรดอัตโนมัติ (ปรับปรุง UI ให้ใช้งานง่ายขึ้น)")

# ============================================================
# 2. Sidebar settings
# ============================================================
with st.sidebar:
    st.header("⚙️ ตั้งค่าระบบ")
    initial_capital = st.number_input("ทุนเริ่มต้น (USD)", value=10000.0, step=1000.0)
    asset = st.selectbox("สินทรัพย์", ["BTC/USDT", "ETH/USDT", "GOLD", "AAPL"])
    tv_symbol = (
        f"BINANCE:{asset.replace('/', '')}"
        if "USDT" in asset
        else ("OANDA:XAUUSD" if asset == "GOLD" else f"NASDAQ:{asset}")
    )

    with st.expander("🛡️ จัดการความเสี่ยง", expanded=True):
        stop_loss_pct = st.slider("Stop Loss (%)", 1.0, 10.0, 3.0, 0.5)
        take_profit_pct = st.slider("Take Profit (%)", 1.0, 20.0, 6.0, 0.5)

    with st.expander("📐 พารามิเตอร์กลยุทธ์", expanded=False):
        ema_fast = st.slider("EMA เร็ว", 2, 20, 5)
        ema_slow = st.slider("EMA ช้า", 10, 60, 20)
        rsi_buy_max = st.slider("RSI สูงสุดที่ยอมให้เปิดซื้อ", 50, 90, 70)
        rsi_sell_min = st.slider("RSI ที่เริ่มขายทำกำไรตามสัญญาณ", 60, 95, 75)

# ============================================================
# 3. Trading logic helpers
# ============================================================
def simulate_backtest(prices, start_capital, stop_loss_pct, take_profit_pct, ema_fast, ema_slow, rsi_buy_max, rsi_sell_min):
    df = pd.DataFrame({"Close": prices})
    df = run_strategy(df, ema_fast=ema_fast, ema_slow=ema_slow, rsi_buy_max=rsi_buy_max, rsi_sell_min=rsi_sell_min)

    capital = start_capital
    position = 0
    units = 0.0
    entry_price = 0.0
    logs = []
    equity_curve = [start_capital]

    for i in range(1, len(df)):
        price = float(df["Close"].iloc[i])
        sig = int(df["Signal"].iloc[i])
        date_str = df.index[i].strftime("%Y-%m-%d") if hasattr(df.index[i], "strftime") else str(df.index[i])

        if sig == 1 and position == 0:
            position = 1
            entry_price = price
            units = capital / max(entry_price, 1e-8)
            logs.append({"วันที่": date_str, "การกระทำ": "BUY (ซื้อ)", "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital, 2), "PnL": None})
        elif position == 1:
            pnl_pct = ((price - entry_price) / max(entry_price, 1e-8)) * 100
            if pnl_pct <= -stop_loss_pct:
                pnl = units * (price - entry_price)
                capital += pnl
                logs.append({"วันที่": date_str, "การกระทำ": "🛑 STOP LOSS", "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital, 2), "PnL": round(pnl, 2)})
                position = 0
                units = 0.0
            elif pnl_pct >= take_profit_pct:
                pnl = units * (price - entry_price)
                capital += pnl
                logs.append({"วันที่": date_str, "การกระทำ": "🎯 TAKE PROFIT", "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital, 2), "PnL": round(pnl, 2)})
                position = 0
                units = 0.0
            elif sig == -1:
                pnl = units * (price - entry_price)
                capital += pnl
                logs.append({"วันที่": date_str, "การกระทำ": "SELL (ตามสัญญาณ)", "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital, 2), "PnL": round(pnl, 2)})
                position = 0
                units = 0.0

        unrealized = units * (price - entry_price) if position == 1 else 0.0
        equity_curve.append(capital + unrealized)

    return pd.DataFrame(logs), equity_curve


# ============================================================
# 4. Tabs
# ============================================================
tab_backtest, tab_live = st.tabs(["📊 Backtest (ทดสอบย้อนหลัง)", "⚡ Live Simulation (จำลองเรียลไทม์)"])

with tab_backtest:
    render_tradingview_chart(tv_symbol, key="backtest")
    if st.button("▶️ เริ่มรัน Backtest", type="primary"):
        np.random.seed(42)
        mock_prices = 100 * (1 + np.random.normal(0, 0.02, 100).cumsum())
        logs_df, eq_curve = simulate_backtest(
            mock_prices,
            initial_capital,
            stop_loss_pct,
            take_profit_pct,
            ema_fast,
            ema_slow,
            rsi_buy_max,
            rsi_sell_min,
        )

        sharpe = compute_sharpe(pd.Series(eq_curve).pct_change().dropna(), annualize_factor=252)
        max_dd, _ = compute_max_drawdown(eq_curve)
        win_rate, n_trades = compute_win_rate(logs_df)

        st.markdown("### สรุปผลการทดสอบ")
        render_metrics_row(eq_curve[-1], initial_capital, sharpe, max_dd, win_rate, n_trades)
        render_equity_and_drawdown(eq_curve)

        st.markdown("### 📋 บันทึกผลการเทรด")
        if not logs_df.empty:
            st.dataframe(logs_df, use_container_width=True)
            download_logs_button(logs_df, f"backtest_{asset.replace('/', '')}.csv")
            st.toast(f"Backtest เสร็จสิ้น — กำไร/ขาดทุนสุทธิ {eq_curve[-1] - initial_capital:+,.2f} USD", icon="✅")
        else:
            st.info("ไม่มีสัญญาณซื้อขายเกิดขึ้นในรอบทดสอบนี้")

with tab_live:
    FRAGMENT_SUPPORTED = hasattr(st, "fragment")

    if ("live_symbol" not in st.session_state) or (st.session_state.live_symbol != asset):
        np.random.seed()
        st.session_state.live_symbol = asset
        st.session_state.live_prices = [100.0]
        st.session_state.live_dates = [datetime.now()]
        st.session_state.live_capital = initial_capital
        st.session_state.live_position = 0
        st.session_state.live_units = 0.0
        st.session_state.live_entry_price = 0.0
        st.session_state.live_logs = []
        st.session_state.live_equity = [initial_capital]
        st.session_state.live_paused = False
        st.session_state.live_last_action = None

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.subheader(f"🔴 จำลองการทำงานเรียลไทม์ของ {asset}")
    with top_r:
        st.session_state.live_paused = st.toggle("หยุดการอัปเดตชั่วคราว", value=st.session_state.live_paused)

    if not FRAGMENT_SUPPORTED:
        st.warning("เวอร์ชัน Streamlit นี้ไม่รองรับ st.fragment แบบเต็ม จึงใช้การอัปเดตแบบกดปุ่ม")

    def _execute_tick():
        last_price = st.session_state.live_prices[-1]
        new_price = max(0.01, last_price * (1 + np.random.normal(0, 0.002)))
        st.session_state.live_prices.append(float(new_price))
        st.session_state.live_dates.append(datetime.now())

        price_df = pd.DataFrame({"Close": st.session_state.live_prices})
        price_df = run_strategy(price_df, ema_fast=ema_fast, ema_slow=ema_slow, rsi_buy_max=rsi_buy_max, rsi_sell_min=rsi_sell_min)

        signal = int(price_df["Signal"].iloc[-1])
        price = float(price_df["Close"].iloc[-1])
        ts = st.session_state.live_dates[-1].strftime("%H:%M:%S")

        position = st.session_state.live_position
        units = st.session_state.live_units
        entry_price = st.session_state.live_entry_price
        capital_now = st.session_state.live_capital
        action_msg, icon = None, None

        if signal == 1 and position == 0:
            position = 1
            entry_price = price
            units = capital_now / max(entry_price, 1e-8)
            action_msg = f"เปิดสถานะ BUY ที่ {price:,.2f}"
            icon = "🟢"
            st.session_state.live_logs.append({"เวลา": ts, "การกระทำ": "BUY (ซื้อ)", "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital_now, 2), "PnL": None})
        elif position == 1:
            pnl_pct = ((price - entry_price) / max(entry_price, 1e-8)) * 100
            if pnl_pct <= -stop_loss_pct:
                pnl = units * (price - entry_price)
                capital_now += pnl
                action_msg = f"STOP LOSS ที่ {price:,.2f} ({pnl:+,.2f})"
                icon = "🛑"
                st.session_state.live_logs.append({"เวลา": ts, "การกระทำ": "🛑 STOP LOSS", "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital_now, 2), "PnL": round(pnl, 2)})
                position = 0
                units = 0.0
            elif pnl_pct >= take_profit_pct:
                pnl = units * (price - entry_price)
                capital_now += pnl
                action_msg = f"TAKE PROFIT ที่ {price:,.2f} ({pnl:+,.2f})"
                icon = "🎯"
                st.session_state.live_logs.append({"เวลา": ts, "การกระทำ": "🎯 TAKE PROFIT", "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital_now, 2), "PnL": round(pnl, 2)})
                position = 0
                units = 0.0
            elif signal == -1:
                pnl = units * (price - entry_price)
                capital_now += pnl
                action_msg = f"SELL ตามสัญญาณที่ {price:,.2f} ({pnl:+,.2f})"
                icon = "⚪"
                st.session_state.live_logs.append({"เวลา": ts, "การกระทำ": "SELL (ตามสัญญาณ)", "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital_now, 2), "PnL": round(pnl, 2)})
                position = 0
                units = 0.0

        st.session_state.live_position = position
        st.session_state.live_units = units
        st.session_state.live_entry_price = entry_price
        st.session_state.live_capital = capital_now

        unrealized = units * (price - entry_price) if position == 1 else 0.0
        st.session_state.live_equity.append(capital_now + unrealized)

        if action_msg:
            st.session_state.live_last_action = action_msg
            st.toast(action_msg, icon=icon)

    def _render_live_panel():
        if not st.session_state.live_paused:
            _execute_tick()

        price_now = st.session_state.live_prices[-1]
        equity_now = st.session_state.live_equity[-1]
        logs_df = pd.DataFrame(st.session_state.live_logs)
        sharpe = compute_sharpe(pd.Series(st.session_state.live_equity).pct_change().dropna())
        max_dd, _ = compute_max_drawdown(st.session_state.live_equity)
        win_rate, n_trades = compute_win_rate(logs_df)

        pill_class = "pill-paused" if st.session_state.live_paused else ("pill-long" if st.session_state.live_position == 1 else "pill-flat")
        pill_text = "หยุดชั่วคราว" if st.session_state.live_paused else ("LONG (ถือสถานะซื้อ)" if st.session_state.live_position == 1 else "FLAT (ไม่มีสถานะ)")

        st.markdown(
            f"สถานะ: <span class='status-pill {pill_class}'>{pill_text}</span>"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;ราคาล่าสุด: **{price_now:,.2f}**"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;อัปเดตล่าสุด: {st.session_state.live_dates[-1].strftime('%H:%M:%S')}"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;<span style='color:#9aa4b2'>อัปเดตอัตโนมัติทุก 1 นาที</span>",
            unsafe_allow_html=True,
        )

        if st.session_state.live_last_action:
            st.info(f"เหตุการณ์ล่าสุด: {st.session_state.live_last_action}")

        render_metrics_row(equity_now, initial_capital, sharpe, max_dd, win_rate, n_trades)
        render_equity_and_drawdown(st.session_state.live_equity)

        st.markdown("### 📋 บันทึกผลการเทรด (Live)")
        if not logs_df.empty:
            st.dataframe(logs_df.iloc[::-1], use_container_width=True)
            download_logs_button(logs_df, f"live_{asset.replace('/', '')}.csv")
        else:
            st.info("ยังไม่มีสัญญาณซื้อขายเกิดขึ้น — ระบบจะตรวจสอบทุก 1 นาที")

    render_tradingview_chart(tv_symbol, key="live")
    if FRAGMENT_SUPPORTED:
        st.fragment(run_every=60)(_render_live_panel)
    else:
        force_tick = st.button("🔄 ซิงค์ตอนนี้", type="primary")
        if force_tick:
            _render_live_panel()
        else:
            st.session_state.live_paused = True
            _render_live_panel()
