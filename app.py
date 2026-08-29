import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from strategy import run_strategy, compute_sharpe, compute_max_drawdown, compute_win_rate
from gbt_signal import GBTSignal
from ui import apply_dashboard_css, render_metrics_row, render_equity_and_drawdown, download_logs_button
from tradingview_widget import render_tradingview_chart

st.set_page_config(page_title="AI Trading Bot", layout="wide", page_icon="🤖")
apply_dashboard_css()
st.title("🤖 AI Trading Dashboard")
st.caption("Trading Bot V2 + EMA/RSI + GBT forecasting layer — Backtest / Live Simulation")

with st.sidebar:
    st.header("⚙️ ตั้งค่าระบบ")
    initial_capital = st.number_input("ทุนเริ่มต้น (USD)", value=10000.0, step=1000.0)
    asset = st.selectbox("สินทรัพย์", ["BTC/USDT", "ETH/USDT", "GOLD", "AAPL"])
    tv_symbol = f"BINANCE:{asset.replace('/', '')}" if "USDT" in asset else ("OANDA:XAUUSD" if asset == "GOLD" else f"NASDAQ:{asset}")
    with st.expander("🛡️ จัดการความเสี่ยง", expanded=True):
        stop_loss_pct = st.slider("Stop Loss (%)", 1.0, 10.0, 3.0, 0.5)
        take_profit_pct = st.slider("Take Profit (%)", 1.0, 20.0, 6.0, 0.5)
    with st.expander("📐 พารามิเตอร์กลยุทธ์", expanded=False):
        ema_fast = st.slider("EMA เร็ว", 2, 20, 5)
        ema_slow = st.slider("EMA ช้า", 10, 60, 20)
        rsi_buy_max = st.slider("RSI สูงสุดที่ยอมให้เปิดซื้อ", 50, 90, 70)
        rsi_sell_min = st.slider("RSI ที่เริ่มขาย", 60, 95, 75)
    with st.expander("🧠 GBT Model", expanded=True):
        gbt_enabled = st.toggle("เปิดใช้ GBT", value=False)
        gbt_checkpoint = st.text_input("Checkpoint (.pt/.pth)", value="models/gbt.pt")
        gbt_seq_len = st.number_input("Sequence length", min_value=8, max_value=512, value=32, step=8)
        gbt_threshold = st.slider("GBT minimum move (%)", 0.05, 2.0, 0.10, 0.05) / 100.0
        if gbt_enabled:
            if Path(gbt_checkpoint).exists(): st.success("พบ checkpoint — พร้อมโหลด")
            else: st.warning("ยังไม่พบ checkpoint; GBT จะไม่ส่งสัญญาณ")

def make_gbt():
    if not gbt_enabled: return None
    return GBTSignal(checkpoint=gbt_checkpoint, seq_len=int(gbt_seq_len), threshold=gbt_threshold)

def simulate_backtest(prices, start_capital, stop_loss_pct, take_profit_pct, ema_fast, ema_slow, rsi_buy_max, rsi_sell_min, gbt_signal=None):
    df = pd.DataFrame({"Close": prices})
    df = run_strategy(df, ema_fast=ema_fast, ema_slow=ema_slow, rsi_buy_max=rsi_buy_max, rsi_sell_min=rsi_sell_min, gbt_signal=gbt_signal)
    capital, position, units, entry_price = start_capital, 0, 0.0, 0.0
    logs, equity_curve = [], [start_capital]
    for i in range(1, len(df)):
        price, sig = float(df["Close"].iloc[i]), int(df["Signal"].iloc[i])
        date_str = df.index[i].strftime("%Y-%m-%d") if hasattr(df.index[i], "strftime") else str(df.index[i])
        if sig == 1 and position == 0:
            position, entry_price, units = 1, price, capital / max(price, 1e-8)
            logs.append({"วันที่": date_str, "การกระทำ": "BUY (ซื้อ)", "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital, 2), "PnL": None})
        elif position == 1:
            pnl_pct = ((price - entry_price) / max(entry_price, 1e-8)) * 100
            if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct or sig == -1:
                pnl = units * (price - entry_price); capital += pnl
                action = "🛑 STOP LOSS" if pnl_pct <= -stop_loss_pct else ("🎯 TAKE PROFIT" if pnl_pct >= take_profit_pct else "SELL (ตามสัญญาณ)")
                logs.append({"วันที่": date_str, "การกระทำ": action, "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital, 2), "PnL": round(pnl, 2)})
                position, units = 0, 0.0
        equity_curve.append(capital + (units * (price - entry_price) if position else 0.0))
    return pd.DataFrame(logs), equity_curve, df

tab_backtest, tab_live = st.tabs(["📊 Backtest", "⚡ Live Simulation"])
with tab_backtest:
    render_tradingview_chart(tv_symbol, key="backtest")
    if st.button("▶️ เริ่มรัน Backtest", type="primary"):
        np.random.seed(42)
        mock_prices = 100 * (1 + np.random.normal(0, 0.02, 100).cumsum())
        gbt = make_gbt()
        logs_df, eq_curve, signal_df = simulate_backtest(mock_prices, initial_capital, stop_loss_pct, take_profit_pct, ema_fast, ema_slow, rsi_buy_max, rsi_sell_min, gbt)
        sharpe = compute_sharpe(pd.Series(eq_curve).pct_change().dropna(), annualize_factor=252)
        max_dd, _ = compute_max_drawdown(eq_curve); win_rate, n_trades = compute_win_rate(logs_df)
        st.markdown("### สรุปผลการทดสอบ"); render_metrics_row(eq_curve[-1], initial_capital, sharpe, max_dd, win_rate, n_trades); render_equity_and_drawdown(eq_curve)
        if gbt_enabled:
            if gbt and gbt.available: st.success("GBT loaded and used in the backtest")
            else: st.warning(f"GBT ไม่ได้ส่งสัญญาณ: {gbt.error if gbt else 'disabled'}")
        if not logs_df.empty:
            st.dataframe(logs_df, use_container_width=True); download_logs_button(logs_df, f"backtest_{asset.replace('/', '')}.csv")
        else: st.info("ไม่มีสัญญาณซื้อขายเกิดขึ้น")

with tab_live:
    FRAGMENT_SUPPORTED = hasattr(st, "fragment")
    if ("live_symbol" not in st.session_state) or (st.session_state.live_symbol != asset):
        st.session_state.live_symbol = asset; st.session_state.live_prices = [100.0]; st.session_state.live_dates = [datetime.now()]
        st.session_state.live_capital = initial_capital; st.session_state.live_position = 0; st.session_state.live_units = 0.0; st.session_state.live_entry_price = 0.0
        st.session_state.live_logs = []; st.session_state.live_equity = [initial_capital]; st.session_state.live_paused = False; st.session_state.live_last_action = None
    st.session_state.live_paused = st.toggle("หยุดการอัปเดตชั่วคราว", value=st.session_state.live_paused)
    if not FRAGMENT_SUPPORTED: st.warning("Streamlit รุ่นนี้ไม่มี fragment; ใช้ปุ่มซิงค์แทน")
    def _execute_tick():
        last_price = st.session_state.live_prices[-1]; new_price = max(0.01, last_price * (1 + np.random.normal(0, 0.002)))
        st.session_state.live_prices.append(float(new_price)); st.session_state.live_dates.append(datetime.now())
        price_df = pd.DataFrame({"Close": st.session_state.live_prices}); gbt = make_gbt()
        price_df = run_strategy(price_df, ema_fast=ema_fast, ema_slow=ema_slow, rsi_buy_max=rsi_buy_max, rsi_sell_min=rsi_sell_min, gbt_signal=gbt)
        signal, price = int(price_df["Signal"].iloc[-1]), float(price_df["Close"].iloc[-1]); ts = st.session_state.live_dates[-1].strftime("%H:%M:%S")
        position, units, entry_price, capital_now = st.session_state.live_position, st.session_state.live_units, st.session_state.live_entry_price, st.session_state.live_capital
        action_msg = None
        if signal == 1 and position == 0:
            position, entry_price, units = 1, price, capital_now / max(price, 1e-8); action_msg = f"เปิด BUY ที่ {price:,.2f}"
            st.session_state.live_logs.append({"เวลา": ts, "การกระทำ": "BUY (ซื้อ)", "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital_now, 2), "PnL": None})
        elif position == 1:
            pnl_pct = ((price - entry_price) / max(entry_price, 1e-8)) * 100
            if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct or signal == -1:
                pnl = units * (price - entry_price); capital_now += pnl
                action = "🛑 STOP LOSS" if pnl_pct <= -stop_loss_pct else ("🎯 TAKE PROFIT" if pnl_pct >= take_profit_pct else "SELL (ตามสัญญาณ)")
                action_msg = f"{action} ที่ {price:,.2f} ({pnl:+,.2f})"; st.session_state.live_logs.append({"เวลา": ts, "การกระทำ": action, "ราคา": round(price, 2), "พอร์ตคงเหลือ": round(capital_now, 2), "PnL": round(pnl, 2)}); position, units = 0, 0.0
        st.session_state.live_position, st.session_state.live_units, st.session_state.live_entry_price, st.session_state.live_capital = position, units, entry_price, capital_now
        st.session_state.live_equity.append(capital_now + (units * (price - entry_price) if position else 0.0))
        if action_msg: st.session_state.live_last_action = action_msg; st.toast(action_msg)
    def _render_live_panel():
        if not st.session_state.live_paused: _execute_tick()
        price_now, equity_now = st.session_state.live_prices[-1], st.session_state.live_equity[-1]; logs_df = pd.DataFrame(st.session_state.live_logs)
        sharpe = compute_sharpe(pd.Series(st.session_state.live_equity).pct_change().dropna()); max_dd, _ = compute_max_drawdown(st.session_state.live_equity); win_rate, n_trades = compute_win_rate(logs_df)
        status = "หยุดชั่วคราว" if st.session_state.live_paused else ("LONG" if st.session_state.live_position else "FLAT")
        st.markdown(f"สถานะ: **{status}** · ราคาล่าสุด: **{price_now:,.2f}** · {st.session_state.live_dates[-1].strftime('%H:%M:%S')}")
        if gbt_enabled: st.caption("🧠 GBT: เปิดใช้งาน (ต้องมี trained checkpoint ที่ตรงกับ architecture)")
        if st.session_state.live_last_action: st.info(f"เหตุการณ์ล่าสุด: {st.session_state.live_last_action}")
        render_metrics_row(equity_now, initial_capital, sharpe, max_dd, win_rate, n_trades); render_equity_and_drawdown(st.session_state.live_equity)
        if not logs_df.empty: st.dataframe(logs_df.iloc[::-1], use_container_width=True); download_logs_button(logs_df, f"live_{asset.replace('/', '')}.csv")
        else: st.info("ยังไม่มีสัญญาณซื้อขาย")
    render_tradingview_chart(tv_symbol, key="live")
    if FRAGMENT_SUPPORTED: st.fragment(run_every=60)(_render_live_panel)
    else:
        if st.button("🔄 ซิงค์ตอนนี้", type="primary"): _render_live_panel()
        else: st.session_state.live_paused = True; _render_live_panel()
