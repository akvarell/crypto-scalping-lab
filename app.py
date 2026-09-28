from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.indicators import add_indicators
from src.strategy import generate_signals

st.set_page_config(page_title="Crypto Scalping Lab", page_icon="📈", layout="wide")


def pf_text(value: float) -> str:
    return "∞" if value == float("inf") else f"{value:.2f}"


def render_metrics(result: dict, label: str) -> None:
    metrics = result["metrics"]
    st.caption(label)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Net return", f"{metrics['net_return_pct']:.2f}%")
    c2.metric("Trades", metrics["trades"])
    c3.metric("Win rate", f"{metrics['win_rate_pct']:.1f}%")
    c4.metric("Profit factor", pf_text(metrics["profit_factor"]))
    c5.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    c6.metric("Avg trade", f"{metrics['avg_trade_pct']:.3f}%")


def render_side_stats(result: dict) -> None:
    st.subheader("LONG vs SHORT")
    long_stats = result["long"]
    short_stats = result["short"]
    table = pd.DataFrame(
        [
            {
                "Side": "LONG",
                "Trades": long_stats["trades"],
                "Win rate %": round(long_stats["win_rate_pct"], 1),
                "P&L": round(long_stats["pnl"], 2),
                "Profit factor": pf_text(long_stats["profit_factor"]),
            },
            {
                "Side": "SHORT",
                "Trades": short_stats["trades"],
                "Win rate %": round(short_stats["win_rate_pct"], 1),
                "P&L": round(short_stats["pnl"], 2),
                "Profit factor": pf_text(short_stats["profit_factor"]),
            },
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_trades(result: dict) -> None:
    trades = result["trades"]
    if trades.empty:
        st.info("No completed trades for this period.")
        return

    display_cols = [
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "exit_reason",
        "duration_bars",
        "pnl",
        "return_pct",
    ]
    shown = trades[display_cols].sort_values("exit_time", ascending=False).copy()
    shown["entry_price"] = shown["entry_price"].round(4)
    shown["exit_price"] = shown["exit_price"].round(4)
    shown["pnl"] = shown["pnl"].round(2)
    shown["return_pct"] = shown["return_pct"].round(3)
    st.dataframe(shown, use_container_width=True, hide_index=True)


st.title("Crypto Scalping Lab v0.2")
st.caption(
    "Research/backtesting only · Public Kraken market data · "
    "No API keys and no real order execution."
)

with st.sidebar:
    st.header("Market")
    symbol = st.selectbox(
        "Symbol",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
        index=0,
    )
    timeframe = st.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "30m", "1h"],
        index=1,
    )
    limit = st.slider("Candles", 300, 700, 600, 50)

    st.header("Entry strategy")
    ema_fast = st.number_input("Fast EMA", min_value=2, max_value=100, value=9, step=1)
    ema_slow = st.number_input("Slow EMA", min_value=3, max_value=300, value=21, step=1)
    rsi_period = st.number_input("RSI period", min_value=2, max_value=100, value=14, step=1)
    rsi_long = st.slider("Long RSI minimum", 1, 99, 52)
    rsi_short = st.slider("Short RSI maximum", 1, 99, 48)

    st.subheader("Filters")
    use_trend_filter = st.checkbox("Trend EMA filter", value=True)
    trend_ema = st.number_input("Trend EMA", min_value=20, max_value=400, value=100, step=10)
    use_volume_filter = st.checkbox("Volume filter", value=False)
    min_volume_ratio = st.slider(
        "Minimum volume / average",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        disabled=not use_volume_filter,
    )

    st.header("Risk / exits")
    atr_period = st.number_input("ATR period", min_value=2, max_value=100, value=14, step=1)
    use_stop_loss = st.checkbox("ATR Stop Loss", value=True)
    stop_atr = st.slider(
        "Stop distance, ATR",
        min_value=0.25,
        max_value=5.0,
        value=1.0,
        step=0.25,
        disabled=not use_stop_loss,
    )
    use_take_profit = st.checkbox("ATR Take Profit", value=True)
    take_atr = st.slider(
        "Take distance, ATR",
        min_value=0.25,
        max_value=8.0,
        value=1.5,
        step=0.25,
        disabled=not use_take_profit,
    )

    st.header("Backtest")
    fee_bps = st.number_input(
        "Fee per side, bps",
        min_value=0.0,
        max_value=100.0,
        value=4.0,
        step=0.5,
    )
    slippage_bps = st.number_input(
        "Slippage per side, bps",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.5,
    )
    start_cash = st.number_input(
        "Starting capital",
        min_value=100.0,
        value=10000.0,
        step=100.0,
    )

    st.header("Validation")
    use_split = st.checkbox("Train / test split", value=True)
    train_pct = st.slider(
        "Train share",
        min_value=50,
        max_value=85,
        value=70,
        step=5,
        disabled=not use_split,
    )

    refresh = st.button("Load / Refresh", type="primary", use_container_width=True)

if ema_fast >= ema_slow:
    st.warning("Fast EMA must be lower than Slow EMA.")
    st.stop()

if use_stop_loss and use_take_profit:
    rr = take_atr / stop_atr
else:
    rr = None


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

df = add_indicators(
    df,
    ema_fast=int(ema_fast),
    ema_slow=int(ema_slow),
    trend_ema=int(trend_ema),
    rsi_period=int(rsi_period),
    atr_period=int(atr_period),
)
df = generate_signals(
    df,
    rsi_long=int(rsi_long),
    rsi_short=int(rsi_short),
    use_trend_filter=bool(use_trend_filter),
    use_volume_filter=bool(use_volume_filter),
    min_volume_ratio=float(min_volume_ratio),
)

bt_kwargs = {
    "start_cash": float(start_cash),
    "fee_bps": float(fee_bps),
    "slippage_bps": float(slippage_bps),
    "use_stop_loss": bool(use_stop_loss),
    "stop_atr": float(stop_atr),
    "use_take_profit": bool(use_take_profit),
    "take_atr": float(take_atr),
}

full_result = run_backtest(df, **bt_kwargs)
train_result = None
test_result = None
split_time = None

if use_split:
    split_idx = max(1, min(len(df) - 1, int(len(df) * train_pct / 100)))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    split_time = df.index[split_idx]

    train_result = run_backtest(train_df, **bt_kwargs)
    test_result = run_backtest(test_df, **bt_kwargs)
    headline_result = test_result
    headline_label = f"Out-of-sample TEST · final {100 - train_pct}% of candles"
else:
    headline_result = full_result
    headline_label = "Full selected period"

render_metrics(headline_result, headline_label)

status_cols = st.columns(4)
status_cols[0].metric("Signals", int((df["signal"] != 0).sum()))
status_cols[1].metric("Current RSI", f"{df['rsi'].iloc[-1]:.1f}")
status_cols[2].metric("Current ATR", f"{df['atr'].iloc[-1]:.2f}")
status_cols[3].metric("Reward / risk", f"{rr:.2f}R" if rr is not None else "Custom exits")

fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
    )
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["ema_fast"],
        mode="lines",
        name=f"EMA {ema_fast}",
    )
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["ema_slow"],
        mode="lines",
        name=f"EMA {ema_slow}",
    )
)
if use_trend_filter:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["trend_ema"],
            mode="lines",
            name=f"Trend EMA {trend_ema}",
        )
    )

longs = df[df["signal"] == 1]
shorts = df[df["signal"] == -1]
if not longs.empty:
    fig.add_trace(
        go.Scatter(
            x=longs.index,
            y=longs["low"] * 0.999,
            mode="markers",
            marker_symbol="triangle-up",
            marker_size=10,
            name="Long signal",
        )
    )
if not shorts.empty:
    fig.add_trace(
        go.Scatter(
            x=shorts.index,
            y=shorts["high"] * 1.001,
            mode="markers",
            marker_symbol="triangle-down",
            marker_size=10,
            name="Short signal",
        )
    )

if split_time is not None:
    fig.add_vline(x=split_time, line_dash="dash", annotation_text="TEST starts")

fig.update_layout(
    height=650,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=35, b=10),
    title=f"{symbol} · {timeframe}",
)
st.plotly_chart(fig, use_container_width=True)

if use_split:
    test_tab, train_tab, full_tab = st.tabs(["TEST", "TRAIN", "FULL"])
    with test_tab:
        render_metrics(test_result, "Out-of-sample test")
        render_side_stats(test_result)
        st.subheader("Equity curve")
        if not test_result["equity"].empty:
            st.line_chart(
                test_result["equity"].set_index("timestamp")["equity"],
                use_container_width=True,
            )
        st.subheader("Test trades")
        render_trades(test_result)

    with train_tab:
        render_metrics(train_result, "Training sample")
        render_side_stats(train_result)
        st.subheader("Equity curve")
        if not train_result["equity"].empty:
            st.line_chart(
                train_result["equity"].set_index("timestamp")["equity"],
                use_container_width=True,
            )
        st.subheader("Train trades")
        render_trades(train_result)

    with full_tab:
        render_metrics(full_result, "Entire loaded period")
        render_side_stats(full_result)
        st.subheader("Full-period trades")
        render_trades(full_result)
else:
    render_side_stats(full_result)
    st.subheader("Equity curve")
    if not full_result["equity"].empty:
        st.line_chart(
            full_result["equity"].set_index("timestamp")["equity"],
            use_container_width=True,
        )
    st.subheader("Trades")
    render_trades(full_result)

st.subheader("Current market snapshot")
latest = df.iloc[-1]
snap1, snap2, snap3, snap4 = st.columns(4)
snap1.metric("Last price", f"{latest['close']:,.4f}")
snap2.metric(
    "Trend",
    "Above trend EMA" if latest["close"] > latest["trend_ema"] else "Below trend EMA",
)
snap3.metric("Volume / avg", f"{latest['volume_ratio']:.2f}x")
signal_text = {1: "LONG", -1: "SHORT", 0: "FLAT"}[int(latest["signal"])]
snap4.metric("Latest signal", signal_text)

with st.expander("v0.2 logic and backtest assumptions"):
    st.write(
        "Entry signals use EMA crosses plus RSI. Optional filters require price to be on the "
        "correct side of the trend EMA and/or volume to exceed its rolling average. "
        "Stops and targets are fixed from ATR at entry. If both stop and target are touched "
        "inside the same candle, the backtest conservatively counts the stop first. "
        "Fees and slippage are applied to every completed trade. The TEST tab is kept separate "
        "from the earlier TRAIN candles to reduce the risk of judging settings only on the same sample."
    )
