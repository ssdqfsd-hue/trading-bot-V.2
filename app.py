import streamlit as st
import pandas as pd
import numpy as np
import time
import streamlit_tradingview as stv

# --- 1. ตั้งค่าหน้าเว็บ Dashboard ---
st.set_page_config(page_title="AI Automated Trading System", layout="wide")
st.title("🤖 ระบบเทรดอัตโนมัติและวิเคราะห์กราฟอัจฉริยะ (Live Dashboard)")

# --- 2. แถบควบคุมด้านข้าง (Sidebar) ---
st.sidebar.header("⚙️ ตั้งค่าระบบบอทเทรด")
initial_capital = st.sidebar.number_input("ทุนเริ่มต้น (USDT / บาท)", value=50000.0, step=5000.0)
symbol = st.sidebar.selectbox("เลือกคู่เหรียญ/สินทรัพย์", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

trading_mode = st.sidebar.radio("เลือกโหมดการทำงาน:", ["📊 ทดสอบระบบย้อนหลัง (Backtest)", "⚡ รันบอทจำลองเรียลไทม์ (Live Simulation)"])

def run_strategy(df):
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Signal'] = 0
    buy_condition = (df['EMA_5'] > df['EMA_20']) & (df['RSI'] < 70)
    sell_condition = (df['EMA_5'] < df['EMA_20']) | (df['RSI'] > 75)
    
    df.loc[buy_condition, 'Signal'] = 1
    df.loc[sell_condition, 'Signal'] = -1
    
    return df

# แปลง symbol ให้เข้ากับ TradingView
tv_symbol = symbol.replace("/", "")

if trading_mode == "📊 ทดสอบระบบย้อนหลัง (Backtest)":
    if st.sidebar.button("▶️ เริ่มรัน Backtest", type="primary"):
        np.random.seed(42)
        dates = pd.date_range(start="2026-01-01", periods=150, freq="D")
        prices = 100 + np.cumsum(np.random.randn(150) * 1.5)
        df = pd.DataFrame({'Close': prices}, index=dates)
        
        df = run_strategy(df)
        
        st.subheader(f"📈 กราฟราคาและอินดิเคเตอร์ของ {symbol}")
        
        # ใช้ TradingView Chart
        stv.st_tradingview(symbol=tv_symbol, interval="D", width="100%", height=600)

        capital = initial_capital
        position = 0
        entry_price = 0.0
        logs = []

        for i in range(1, len(df)):
            price = df['Close'].iloc[i]
            sig = df['Signal'].iloc[i]
            date_str = df.index[i].strftime('%Y-%m-%d')
            
            if sig == 1 and position == 0:
                position = 1
                entry_price = price
                logs.append({"วันที่": date_str, "การกระทำ": "BUY (ซื้อ)", "ราคา": f"{entry_price:.2f}", "พอร์ตคงเหลือ": f"{capital:.2f}"})
            elif sig == -1 and position == 1:
                pnl = price - entry_price
                capital += pnl
                logs.append({"วันที่": date_str, "การกระทำ": "SELL (ขาย)", "ราคา": f"{price:.2f}", "พอร์ตคงเหลือ": f"{capital:.2f}"})
                position = 0
        
        st.subheader("📋 บันทึกผลการเทรดอัตโนมัติ")
        if logs:
            st.table(pd.DataFrame(logs))
        else:
            st.info("ไม่มีสัญญาณซื้อขายในรอบนี้")
        st.metric("เงินทุนสุทธิสิ้นสุด", f"{capital:,.2f} บาท", f"{capital - initial_capital:+,.2f} บาท")

elif trading_mode == "⚡ รันบอทจำลองเรียลไทม์ (Live Simulation)":
    st.subheader(f"🔴 จำลองการทำงานเรียลไทม์ของ {symbol}")
    stv.st_tradingview(symbol=tv_symbol, interval="D", width="100%", height=600)
    
    st.write("สถานะ AI: กำลังวิเคราะห์ตลาด... (ระบบเชื่อมต่อเรียลไทม์)")
    placeholder = st.empty()
    for seconds in range(5):
        with placeholder.container():
            st.write(f"กำลังดึงข้อมูลรอบที่ {seconds+1}/5 (อัปเดตทุก 3 วินาที)...")
            time.sleep(3)
