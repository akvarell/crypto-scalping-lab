from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from src.data import fetch_ohlcv
from src.indicators import add_indicators
from src.strategy import generate_signals
from src.backtest import run_backtest

st.set_page_config(page_title="Crypto Scalping Lab", page_icon="📈", layout="wide")

st.title("Crypto Scalping Lab v0.1")
st.caption("Research and backtesting dashboard. No order execution and no exchange API keys required.")

with st.sidebar:
    st.header("Market")
    symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"], index=0)
    timeframe = st.selectbox("Timeframe", ["1m", "3m", "5m", "15m", "30m", "1h"], index=2)
    limit = st.slider("Candles", 200, 1500, 600, 100)

    st.header("Strategy")
    ema_fast = st.number_input("Fast EMA", min_value=2, max_value=100, value=9, step=1)
    ema_slow = st.number_input("Slow EMA", min_value=3, max_value=300, value=21, step=1)
    rsi_period = st.number_input("RSI period", min_value=2, max_value=100, value=14, step=1)
    rsi_long = st.slider("Long RSI minimum", 1, 99, 52)
    rsi_short = st.slider("Short RSI maximum", 1, 99, 48)
    atr_period = st.number_input("ATR period", min_value=2, max_value=100, value=14, step=1)

    st.header("Backtest")
    fee_bps = st.number_input("Fee per side, bps", min_value=0.0, max_value=100.0, value=4.0, step=0.5)
    slippage_bps = st.number_input("Slippage per side, bps", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
    start_cash = st.number_input("Starting capital", min_value=100.0, value=10000.0, step=100.0)

    refresh = st.button("Load / Refresh", type="primary", use_container_width=True)

if ema_fast >= ema_slow:
    st.warning("Fast EMA must be lower than Slow EMA.")
    st.stop()

@st.cache_data(ttl=30, show_spinner=False)
def load_market(symbol: str, timeframe: str, limit: int):
    return fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)

try:
    with st.spinner("Loading market data..."):
        df = load_market(symbol, timeframe, limit)
except Exception as exc:
    st.error(f"Could not load market data: {exc}")
    st.stop()

if refresh:
    load_market.clear()
    st.rerun()

params = {
    "ema_fast": int(ema_fast),
    "ema_slow": int(ema_slow),
    "rsi_period": int(rsi_period),
    "rsi_long": int(rsi_long),
    "rsi_short": int(rsi_short),
    "atr_period": int(atr_period),
}

df = add_indicators(df, **params)
df = generate_signals(df, rsi_long=int(rsi_long), rsi_short=int(rsi_short))
results = run_backtest(
    df,
    start_cash=float(start_cash),
    fee_bps=float(fee_bps),
    slippage_bps=float(slippage_bps),
)

metrics = results["metrics"]
trades = results["trades"]
equity = results["equity"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Net return", f"{metrics['net_return_pct']:.2f}%")
m2.metric("Trades", int(metrics["trades"]))
m3.metric("Win rate", f"{metrics['win_rate_pct']:.1f}%")
m4.metric("Profit factor", "∞" if metrics["profit_factor"] == float("inf") else f"{metrics['profit_factor']:.2f}")
m5.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.2f}%")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="Price",
))
fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], mode="lines", name=f"EMA {ema_fast}"))
fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], mode="lines", name=f"EMA {ema_slow}"))

longs = df[df["signal"] == 1]
shorts = df[df["signal"] == -1]
if not longs.empty:
    fig.add_trace(go.Scatter(
        x=longs.index,
        y=longs["low"] * 0.999,
        mode="markers",
        marker_symbol="triangle-up",
        marker_size=10,
        name="Long signal",
    ))
if not shorts.empty:
    fig.add_trace(go.Scatter(
        x=shorts.index,
        y=shorts["high"] * 1.001,
        mode="markers",
        marker_symbol="triangle-down",
        marker_size=10,
        name="Short signal",
    ))

fig.update_layout(
    height=650,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=35, b=10),
    title=f"{symbol} · {timeframe}",
)
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Equity curve")
    st.line_chart(equity.set_index("timestamp")["equity"], use_container_width=True)
with c2:
    st.subheader("Current snapshot")
    latest = df.iloc[-1]
    st.metric("Last price", f"{latest['close']:,.4f}")
    st.metric("RSI", f"{latest['rsi']:.1f}")
    st.metric("ATR", f"{latest['atr']:.4f}")
    signal_text = {1: "LONG", -1: "SHORT", 0: "FLAT"}[int(latest["signal"])]
    st.metric("Signal", signal_text)

st.subheader("Trades")
if trades.empty:
    st.info("No completed trades for the selected period and parameters.")
else:
    display_cols = ["entry_time", "exit_time", "side", "entry_price", "exit_price", "pnl", "return_pct"]
    st.dataframe(trades[display_cols].sort_values("exit_time", ascending=False), use_container_width=True, hide_index=True)

with st.expander("How the v0.1 signal works"):
    st.write(
        "Long: fast EMA crosses above slow EMA and RSI is above the selected long threshold. "
        "Short: fast EMA crosses below slow EMA and RSI is below the selected short threshold. "
        "A position exits on the opposite signal. Fees and slippage are applied on entry and exit."
    )
