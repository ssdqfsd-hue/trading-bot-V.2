import numpy as np
import pandas as pd


def run_strategy(df, ema_fast=5, ema_slow=20, rsi_buy_max=70, rsi_sell_min=75, gbt_signal=None):
    df = df.copy()
    df["EMA_fast"] = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["EMA_slow"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(50)

    df["Signal"] = 0
    buy_condition = (df["EMA_fast"] > df["EMA_slow"]) & (df["RSI"] < rsi_buy_max)
    sell_condition = (df["EMA_fast"] < df["EMA_slow"]) | (df["RSI"] > rsi_sell_min)
    df.loc[buy_condition, "Signal"] = 1
    df.loc[sell_condition, "Signal"] = -1

    if gbt_signal is not None and hasattr(gbt_signal, "signal"):
        gbt = []
        for i in range(len(df)):
            try:
                gbt.append(int(gbt_signal.signal(df["Close"].iloc[: i + 1])))
            except Exception:
                gbt.append(0)
        gbt = pd.Series(gbt, index=df.index)
        df["GBT_Signal"] = gbt
        df.loc[gbt == 1, "Signal"] = 1
        df.loc[gbt == -1, "Signal"] = -1

    df["Signal"] = df["Signal"].shift(1).fillna(0)
    return df


def compute_sharpe(returns, annualize_factor=None):
    returns = pd.Series(returns).dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    sharpe = returns.mean() / returns.std()
    if annualize_factor:
        sharpe *= np.sqrt(annualize_factor)
    return sharpe


def compute_max_drawdown(equity_series):
    equity_series = pd.Series(equity_series).dropna()
    if len(equity_series) < 2:
        return 0.0, pd.Series(dtype=float)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max * 100
    return float(drawdown.min()), drawdown


def compute_win_rate(logs_df):
    if logs_df is None or logs_df.empty or "PnL" not in logs_df.columns:
        return 0.0, 0
    closed = logs_df.dropna(subset=["PnL"])
    if closed.empty:
        return 0.0, 0
    wins = (closed["PnL"] > 0).sum()
    return (wins / len(closed)) * 100, len(closed)
