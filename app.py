from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.alt_basket_study import run_alt_basket_study
from src.backtest import run_backtest
from src.counter_regime_holdout import run_counter_regime_holdout
from src.data import fetch_ohlcv
from src.direction_regime_study import run_direction_regime_study
from src.dynamic_scalping import run_dynamic_universe_scalping
from src.edge_diagnostics import evaluate_edge_diagnostics
from src.entry_quality_holdout import run_frozen_1m_entry_holdout
from src.entry_quality_lab import run_entry_quality_lab
from src.event_edge_study import run_event_edge_study
from src.execution_reality import run_execution_reality_check
from src.exit_surface import run_exit_surface
from src.family_benchmark import benchmark_families
from src.indicators import add_indicators
from src.mean_reversion_lab import evaluate_mean_reversion_variants
from src.one_shot_lab import run_one_shot_lab
from src.optimizer import optimize_quick
from src.relative_event_lab import run_relative_event_lab
from src.rolling_universe import run_rolling_universe_validation
from src.scalping_edge_map import run_scalping_edge_map
from src.strategy import generate_signals
from src.universe import build_universe_screener
from src.walkforward import evaluate_walk_forward

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


st.title("Crypto Scalping Lab v2.3")
st.caption(
    "Research/backtesting only · Public Coinbase market data · "
    "No API keys and no real order execution."
)

with st.sidebar:
    st.header("Market")
    symbol = st.selectbox(
        "Symbol",
        ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"],
        index=0,
    )
    timeframe = st.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h"],
        index=1,
    )
    limit = st.slider("Candles", 1000, 5000, 3000, 500)

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


@st.cache_data(ttl=300, show_spinner=False)
def load_market(symbol: str, timeframe: str, limit: int):
    return fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)


try:
    with st.spinner("Loading market data..."):
        raw_df = load_market(symbol, timeframe, limit)
except Exception as exc:
    st.error(f"Could not load market data: {exc}")
    st.stop()

if refresh:
    load_market.clear()
    st.rerun()

if len(raw_df) < int(limit * 0.8):
    st.warning(
        f"Requested {limit} candles but received only {len(raw_df)}. "
        "Results may use a smaller sample than expected."
    )

df = add_indicators(
    raw_df,
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
    split_x = split_time.isoformat()
    fig.add_shape(
        type="line",
        x0=split_x,
        x1=split_x,
        y0=0,
        y1=1,
        yref="paper",
        line={"dash": "dash"},
    )

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

st.subheader("Market Universe Screener · v0.9")
st.caption(
    "Select the research universe first: liquidity + volatility + activity. "
    "BTC correlation is displayed as context and is not used in the research score. "
    "This is a market-research screener, not a recommendation to trade a specific coin."
)

us1, us2, us3 = st.columns(3)
with us1:
    universe_size = st.slider(
        "Liquid symbols to inspect",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        key="universe_size",
    )
with us2:
    universe_lookback = st.select_slider(
        "Screener lookback",
        options=[168, 336, 720],
        value=336,
        format_func=lambda x: {168: "7 days", 336: "14 days", 720: "30 days"}[x],
        key="universe_lookback",
    )
with us3:
    min_turnover_m = st.number_input(
        "Minimum 24h turnover, $M",
        min_value=1.0,
        max_value=500.0,
        value=10.0,
        step=5.0,
        key="min_turnover_m",
    )

run_universe = st.button(
    "Build market universe",
    use_container_width=True,
    key="run_universe",
)

universe_signature = (
    int(universe_size),
    int(universe_lookback),
    float(min_turnover_m),
)

if run_universe:
    with st.spinner("Scanning liquid USDT pairs and calculating volatility/correlation..."):
        try:
            universe_df = build_universe_screener(
                max_symbols=int(universe_size),
                lookback_hours=int(universe_lookback),
                min_quote_volume_usd=float(min_turnover_m) * 1_000_000.0,
            )
            st.session_state["universe_df"] = universe_df
            st.session_state["universe_signature"] = universe_signature
        except Exception as exc:
            st.error(f"Universe screener could not load Binance public data: {exc}")

universe_df = st.session_state.get("universe_df")
saved_universe_signature = st.session_state.get("universe_signature")

if universe_df is not None:
    if saved_universe_signature != universe_signature:
        st.info("Screener settings changed. Build the market universe again.")
    else:
        filter1, filter2, filter3 = st.columns(3)
        with filter1:
            min_atr_pct = st.slider(
                "Minimum ATR %",
                min_value=0.0,
                max_value=10.0,
                value=0.5,
                step=0.1,
                key="min_atr_pct",
            )
        with filter2:
            min_realized_vol = st.slider(
                "Minimum dailyized realized vol %",
                min_value=0.0,
                max_value=20.0,
                value=1.5,
                step=0.1,
                key="min_realized_vol",
            )
        with filter3:
            corr_abs_max = st.slider(
                "Maximum |BTC correlation|",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                key="corr_abs_max",
            )

        shown_universe = universe_df[
            (universe_df["ATR %"] >= float(min_atr_pct))
            & (universe_df["Dailyized realized vol %"] >= float(min_realized_vol))
            & (universe_df["BTC corr"].abs() <= float(corr_abs_max))
        ].copy()

        for col in [
            "24h turnover $",
            "7d avg turnover $",
            "24h change %",
            "Dailyized realized vol %",
            "ATR %",
            "Avg abs 1h move %",
            "BTC corr",
            "7d avg trades/day",
            "Research score",
        ]:
            if col in shown_universe:
                shown_universe[col] = shown_universe[col].round(2)

        display_cols = [
            "Symbol",
            "24h turnover $",
            "7d avg turnover $",
            "24h change %",
            "Dailyized realized vol %",
            "ATR %",
            "Avg abs 1h move %",
            "BTC corr",
            "24h trades",
            "7d avg trades/day",
            "Research score",
        ]
        st.dataframe(
            shown_universe[display_cols],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Research score = 30% liquidity + 35% realized volatility + 25% ATR rank + 10% activity. "
            "BTC correlation is intentionally separate. For a historical strategy test, the universe "
            "must be re-selected at each past date using only information available before that date."
        )

st.subheader("Rolling Universe Validation · v1.1")
st.caption(
    "Long-distance validation of the screener itself. At each historical rebalance date, "
    "the candidate pool is rebuilt from historical trailing turnover at that date, then ranked "
    "with only the preceding lookback window and measured on the following period. "
    "This removes the v1.0 current-volume look-ahead before we attach a trading strategy."
)

ru1, ru2, ru3 = st.columns(3)
with ru1:
    rolling_horizon = st.select_slider(
        "Validation horizon",
        options=[60, 90, 180],
        value=90,
        format_func=lambda x: f"{x} days",
        key="rolling_horizon",
    )
with ru2:
    rolling_lookback = st.select_slider(
        "Selection lookback",
        options=[7, 14, 30],
        value=14,
        format_func=lambda x: f"{x} days",
        key="rolling_lookback",
    )
with ru3:
    rolling_forward = st.select_slider(
        "Forward evaluation window",
        options=[3, 7],
        value=7,
        format_func=lambda x: f"{x} days",
        key="rolling_forward",
    )

ru4, ru5, ru6 = st.columns(3)
with ru4:
    rolling_pool = st.slider(
        "Candidate pool size",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        key="rolling_pool",
    )
with ru5:
    rolling_top_n = st.slider(
        "Select top N each period",
        min_value=2,
        max_value=8,
        value=5,
        step=1,
        key="rolling_top_n",
    )
with ru6:
    rolling_min_turnover_m = st.number_input(
        "Historical min daily turnover, $M",
        min_value=1.0,
        max_value=200.0,
        value=5.0,
        step=5.0,
        key="rolling_min_turnover_m",
    )

rolling_signature = (
    int(rolling_horizon),
    int(rolling_lookback),
    int(rolling_forward),
    int(rolling_pool),
    int(rolling_top_n),
    float(rolling_min_turnover_m),
)

run_rolling = st.button(
    "Run rolling universe validation",
    use_container_width=True,
    key="run_rolling_universe",
)

if run_rolling:
    with st.spinner(
        "Downloading historical hourly data and rebuilding the universe at each past date..."
    ):
        try:
            rolling_periods, rolling_details, rolling_summary = run_rolling_universe_validation(
                horizon_days=int(rolling_horizon),
                lookback_days=int(rolling_lookback),
                forward_days=int(rolling_forward),
                pool_size=int(rolling_pool),
                select_top_n=int(rolling_top_n),
                min_daily_turnover_usd=float(rolling_min_turnover_m) * 1_000_000.0,
            )
            st.session_state["rolling_periods"] = rolling_periods
            st.session_state["rolling_details"] = rolling_details
            st.session_state["rolling_summary"] = rolling_summary
            st.session_state["rolling_signature"] = rolling_signature
        except Exception as exc:
            st.error(f"Rolling validation failed: {exc}")

rolling_periods = st.session_state.get("rolling_periods")
rolling_details = st.session_state.get("rolling_details")
rolling_summary = st.session_state.get("rolling_summary")
saved_rolling_signature = st.session_state.get("rolling_signature")

if rolling_summary is not None:
    if saved_rolling_signature != rolling_signature:
        st.info("Rolling validation settings changed. Run it again to refresh the results.")
    else:
        periods_count = int(rolling_summary.get("periods", 0))
        positive_count = int(rolling_summary.get("positive_vol_uplift_periods", 0))

        rs1, rs2, rs3, rs4, rs5 = st.columns(5)
        rs1.metric("Periods", periods_count)
        rs2.metric(
            "Positive vol uplift",
            f"{positive_count}/{periods_count}" if periods_count else "0/0",
        )
        rs3.metric(
            "Avg vol uplift",
            f"{rolling_summary.get('avg_vol_uplift_pct', 0.0):.1f}%",
        )
        rs4.metric(
            "Median vol uplift",
            f"{rolling_summary.get('median_vol_uplift_pct', 0.0):.1f}%",
        )
        rs5.metric(
            "Avg move uplift",
            f"{rolling_summary.get('avg_move_uplift_pct', 0.0):.1f}%",
        )

        st.caption(
            "Positive uplift means the coins selected using only past data were, on average, "
            "more volatile/move-rich in the following window than the eligible comparison universe. "
            "This validates opportunity selection, not profitability."
        )

        if rolling_periods is not None and not rolling_periods.empty:
            shown_periods = rolling_periods.copy()
            for col in [
                "Selected forward vol %",
                "Universe forward vol %",
                "Vol uplift %",
                "Selected abs 1h move %",
                "Universe abs 1h move %",
                "Move uplift %",
                "Turnover uplift %",
            ]:
                if col in shown_periods:
                    shown_periods[col] = shown_periods[col].round(2)

            st.dataframe(
                shown_periods,
                use_container_width=True,
                hide_index=True,
            )

        if rolling_details is not None and not rolling_details.empty:
            with st.expander("Selected-symbol details by period"):
                shown_details = rolling_details.copy()
                for col in [
                    "Selection score",
                    "Lookback vol %",
                    "Lookback ATR %",
                    "Forward vol %",
                    "Forward ATR %",
                    "Forward abs 1h move %",
                ]:
                    if col in shown_details:
                        shown_details[col] = shown_details[col].round(2)
                st.dataframe(
                    shown_details,
                    use_container_width=True,
                    hide_index=True,
                )

        st.warning(
            "v1.1 removes the current-volume look-ahead from v1.0: historical candidate pools are "
            "formed using turnover known at each past date. One research-stage limitation remains: "
            "the master symbol list contains pairs that are trading today, so delisted assets are not "
            "yet reconstructed and survivorship bias is not fully removed."
        )

st.subheader("Dynamic Universe Scalping Backtest · v1.2")
st.caption(
    "This is the first end-to-end test of the intended workflow: historical screener selection -> "
    "next-period 5-minute data -> fixed scalping rules -> fees/slippage -> equal-weight weekly portfolio. "
    "It does not use the future week to choose the coins."
)

if rolling_details is None or rolling_details.empty:
    st.info("Run Rolling Universe Validation first. The scalping backtest uses its historical selections.")
else:
    ds1, ds2, ds3 = st.columns(3)
    with ds1:
        scalp_periods_to_test = st.select_slider(
            "Periods to backtest",
            options=[4, 8, 12],
            value=12,
            key="scalp_periods_to_test",
        )
    with ds2:
        scalp_fee_bps = st.number_input(
            "Scalping fee per side, bps",
            min_value=0.0,
            max_value=20.0,
            value=float(fee_bps),
            step=0.5,
            key="scalp_fee_bps",
        )
    with ds3:
        scalp_slippage_bps = st.number_input(
            "Scalping slippage per side, bps",
            min_value=0.0,
            max_value=20.0,
            value=max(float(slippage_bps), 2.0),
            step=0.5,
            key="scalp_slippage_bps",
        )

    scalp_signature = (
        saved_rolling_signature,
        int(rolling_forward),
        int(scalp_periods_to_test),
        float(scalp_fee_bps),
        float(scalp_slippage_bps),
    )

    run_scalping = st.button(
        "Run dynamic-universe 5m scalping backtest",
        use_container_width=True,
        key="run_dynamic_scalping",
    )

    if run_scalping:
        with st.spinner(
            "Downloading the selected coins' 5-minute candles and testing fixed strategies..."
        ):
            try:
                scalp_summary, scalp_periods, scalp_details = run_dynamic_universe_scalping(
                    rolling_details,
                    forward_days=int(rolling_forward),
                    fee_bps=float(scalp_fee_bps),
                    slippage_bps=float(scalp_slippage_bps),
                    max_periods=int(scalp_periods_to_test),
                )
                st.session_state["scalp_summary"] = scalp_summary
                st.session_state["scalp_periods"] = scalp_periods
                st.session_state["scalp_details"] = scalp_details
                st.session_state["scalp_signature"] = scalp_signature
                st.session_state["scalp_failed_pairs"] = int(
                    scalp_summary.attrs.get("failed_pairs", 0)
                )
            except Exception as exc:
                st.error(f"Dynamic scalping backtest failed: {exc}")

    scalp_summary = st.session_state.get("scalp_summary")
    scalp_periods = st.session_state.get("scalp_periods")
    scalp_details = st.session_state.get("scalp_details")
    saved_scalp_signature = st.session_state.get("scalp_signature")

    if scalp_summary is not None:
        if saved_scalp_signature != scalp_signature:
            st.info("Scalping settings changed. Run the dynamic-universe backtest again.")
        else:
            shown_scalp = scalp_summary.copy()
            for col in [
                "Avg gross return %",
                "Avg net return %",
                "Median net return %",
                "Compounded gross %",
                "Compounded net %",
                "Avg cost drag pp",
                "Worst period %",
                "Best period %",
                "Avg net PF",
            ]:
                if col in shown_scalp:
                    shown_scalp[col] = shown_scalp[col].round(2)

            st.dataframe(shown_scalp, use_container_width=True, hide_index=True)
            failed_pairs = int(st.session_state.get("scalp_failed_pairs", 0))
            st.caption(
                "Weekly portfolio return is the equal-weight average of the selected coins' returns. "
                "Gross uses zero fees/slippage; Net uses the costs entered above. "
                f"Skipped symbol-periods with insufficient 5m data: {failed_pairs}."
            )

            if scalp_periods is not None and not scalp_periods.empty:
                with st.expander("Weekly portfolio results"):
                    shown_periods = scalp_periods.copy()
                    for col in [
                        "Gross portfolio return %",
                        "Net portfolio return %",
                        "Cost drag pp",
                        "Avg win rate %",
                        "Avg max DD %",
                    ]:
                        if col in shown_periods:
                            shown_periods[col] = shown_periods[col].round(2)
                    st.dataframe(
                        shown_periods,
                        use_container_width=True,
                        hide_index=True,
                    )

            if scalp_details is not None and not scalp_details.empty:
                with st.expander("Per-coin 5m results"):
                    shown_details = scalp_details.copy()
                    for col in [
                        "Gross return %",
                        "Net return %",
                        "Cost drag pp",
                        "Gross PF",
                        "Net PF",
                        "Win rate %",
                        "Max DD %",
                        "Long P&L",
                        "Short P&L",
                    ]:
                        if col in shown_details:
                            shown_details[col] = shown_details[col].round(2)
                    st.dataframe(
                        shown_details,
                        use_container_width=True,
                        hide_index=True,
                    )

st.subheader("Execution Reality Check · v1.3")
st.caption(
    "Signals are now acted on only at the NEXT 5-minute bar open. "
    "This removes the optimistic same-bar-close execution used in earlier research. "
    "The table also separates LONG/SHORT and tests simple cooldowns to reduce turnover."
)

if rolling_details is None or rolling_details.empty:
    st.info("Run Rolling Universe Validation first. This check uses the same historical selections.")
else:
    er1, er2, er3 = st.columns(3)
    with er1:
        reality_strategy = st.selectbox(
            "Strategy to audit",
            ["Mean Reversion", "Momentum Pullback", "Vol+Volume Breakout"],
            index=0,
            key="reality_strategy",
        )
    with er2:
        reality_periods = st.select_slider(
            "Periods to audit",
            options=[4, 8, 12],
            value=12,
            key="reality_periods",
        )
    with er3:
        st.metric(
            "Round-trip cost assumption",
            f"{2 * (float(scalp_fee_bps) + float(scalp_slippage_bps)):.1f} bps",
        )

    reality_signature = (
        saved_rolling_signature,
        reality_strategy,
        int(rolling_forward),
        int(reality_periods),
        float(scalp_fee_bps),
        float(scalp_slippage_bps),
    )

    run_reality = st.button(
        "Run execution reality check",
        use_container_width=True,
        key="run_execution_reality",
    )

    if run_reality:
        with st.spinner(
            "Re-running the selected strategy with next-bar execution, side splits and cooldowns..."
        ):
            try:
                reality_summary, reality_period_table, reality_details = run_execution_reality_check(
                    rolling_details,
                    strategy_name=reality_strategy,
                    forward_days=int(rolling_forward),
                    fee_bps=float(scalp_fee_bps),
                    slippage_bps=float(scalp_slippage_bps),
                    max_periods=int(reality_periods),
                )
                st.session_state["reality_summary"] = reality_summary
                st.session_state["reality_period_table"] = reality_period_table
                st.session_state["reality_details"] = reality_details
                st.session_state["reality_signature"] = reality_signature
            except Exception as exc:
                st.error(f"Execution reality check failed: {exc}")

    reality_summary = st.session_state.get("reality_summary")
    reality_period_table = st.session_state.get("reality_period_table")
    reality_details = st.session_state.get("reality_details")
    saved_reality_signature = st.session_state.get("reality_signature")

    if reality_summary is not None:
        if saved_reality_signature != reality_signature:
            st.info("Reality-check settings changed. Run it again to refresh the results.")
        else:
            shown_reality = reality_summary.copy()
            for col in [
                "Avg gross return %",
                "Avg net return %",
                "Median net return %",
                "Compounded net %",
                "Avg cost drag pp",
                "Worst period %",
                "Best period %",
                "Avg trade %",
                "Avg net PF",
            ]:
                if col in shown_reality:
                    shown_reality[col] = shown_reality[col].round(3)

            st.dataframe(
                shown_reality,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Interpretation: LONG/SHORT rows diagnose directional asymmetry. "
                "Cooldown rows show whether fewer, more selective trades improve net results. "
                "Gross vs net still separates signal edge from execution costs."
            )

            if reality_period_table is not None and not reality_period_table.empty:
                with st.expander("Reality-check weekly portfolio results"):
                    shown_reality_periods = reality_period_table.copy()
                    for col in [
                        "Gross portfolio return %",
                        "Net portfolio return %",
                        "Cost drag pp",
                        "Avg trade %",
                    ]:
                        if col in shown_reality_periods:
                            shown_reality_periods[col] = shown_reality_periods[col].round(3)
                    st.dataframe(
                        shown_reality_periods,
                        use_container_width=True,
                        hide_index=True,
                    )

            if reality_details is not None and not reality_details.empty:
                with st.expander("Reality-check per-coin results"):
                    shown_reality_details = reality_details.copy()
                    for col in [
                        "Gross return %",
                        "Net return %",
                        "Cost drag pp",
                        "Gross PF",
                        "Net PF",
                        "Win rate %",
                        "Max DD %",
                        "Avg trade %",
                    ]:
                        if col in shown_reality_details:
                            shown_reality_details[col] = shown_reality_details[col].round(3)
                    st.dataframe(
                        shown_reality_details,
                        use_container_width=True,
                        hide_index=True,
                    )

st.subheader("One-Shot Signal Lab · v1.4")
st.caption(
    "The v1.3 audit showed that Mean Reversion still has positive gross edge but excessive turnover. "
    "This lab removes repeated entries during the same extreme move: a new signal appears only when "
    "price first crosses into the extreme zone. Additional variants require a volume spike and/or a "
    "large candle relative to ATR. All entries still execute at the next 5-minute bar open."
)

if rolling_details is None or rolling_details.empty:
    st.info("Run Rolling Universe Validation first. The one-shot lab uses the same historical selections.")
else:
    os1, os2, os3 = st.columns(3)
    with os1:
        one_shot_periods = st.select_slider(
            "Periods to test",
            options=[4, 8, 12],
            value=12,
            key="one_shot_periods",
        )
    with os2:
        st.metric("Fee per side", f"{float(scalp_fee_bps):.1f} bps")
    with os3:
        st.metric("Slippage per side", f"{float(scalp_slippage_bps):.1f} bps")

    one_shot_signature = (
        saved_rolling_signature,
        int(rolling_forward),
        int(one_shot_periods),
        float(scalp_fee_bps),
        float(scalp_slippage_bps),
    )

    run_one_shot = st.button(
        "Run one-shot signal lab",
        use_container_width=True,
        key="run_one_shot_lab",
    )

    if run_one_shot:
        with st.spinner(
            "Testing one entry per extreme impulse, with volume and capitulation filters..."
        ):
            try:
                one_shot_summary, one_shot_period_table, one_shot_details = run_one_shot_lab(
                    rolling_details,
                    forward_days=int(rolling_forward),
                    fee_bps=float(scalp_fee_bps),
                    slippage_bps=float(scalp_slippage_bps),
                    max_periods=int(one_shot_periods),
                )
                st.session_state["one_shot_summary"] = one_shot_summary
                st.session_state["one_shot_period_table"] = one_shot_period_table
                st.session_state["one_shot_details"] = one_shot_details
                st.session_state["one_shot_signature"] = one_shot_signature
            except Exception as exc:
                st.error(f"One-shot signal lab failed: {exc}")

    one_shot_summary = st.session_state.get("one_shot_summary")
    one_shot_period_table = st.session_state.get("one_shot_period_table")
    one_shot_details = st.session_state.get("one_shot_details")
    saved_one_shot_signature = st.session_state.get("one_shot_signature")

    if one_shot_summary is not None:
        if saved_one_shot_signature != one_shot_signature:
            st.info("One-shot settings changed. Run the lab again to refresh the results.")
        else:
            shown_one_shot = one_shot_summary.copy()
            for col in [
                "Avg gross return %",
                "Avg net return %",
                "Median net return %",
                "Compounded net %",
                "Avg cost drag pp",
                "Worst period %",
                "Best period %",
                "Avg trade %",
                "Avg net PF",
            ]:
                if col in shown_one_shot:
                    shown_one_shot[col] = shown_one_shot[col].round(3)

            st.dataframe(
                shown_one_shot,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "The key comparison is turnover versus retained gross edge. A useful one-shot variant "
                "should cut trades and cost drag sharply without destroying gross return."
            )

            if one_shot_period_table is not None and not one_shot_period_table.empty:
                with st.expander("One-shot weekly portfolio results"):
                    shown_one_shot_periods = one_shot_period_table.copy()
                    for col in [
                        "Gross portfolio return %",
                        "Net portfolio return %",
                        "Cost drag pp",
                        "Avg trade %",
                    ]:
                        if col in shown_one_shot_periods:
                            shown_one_shot_periods[col] = shown_one_shot_periods[col].round(3)
                    st.dataframe(
                        shown_one_shot_periods,
                        use_container_width=True,
                        hide_index=True,
                    )

            if one_shot_details is not None and not one_shot_details.empty:
                with st.expander("One-shot per-coin results"):
                    shown_one_shot_details = one_shot_details.copy()
                    for col in [
                        "Gross return %",
                        "Net return %",
                        "Cost drag pp",
                        "Gross PF",
                        "Net PF",
                        "Win rate %",
                        "Max DD %",
                        "Avg trade %",
                    ]:
                        if col in shown_one_shot_details:
                            shown_one_shot_details[col] = shown_one_shot_details[col].round(3)
                    st.dataframe(
                        shown_one_shot_details,
                        use_container_width=True,
                        hide_index=True,
                    )

st.subheader("BTC-Relative Event Lab · v1.5")
st.caption(
    "Instead of trading every indicator extreme, this lab looks for rarer events in the already-selected "
    "high-opportunity universe: abnormal volume, large 5-minute displacement, relative movement versus BTC, "
    "and low-correlation breakouts. Entries still execute on the NEXT 5-minute bar open."
)

if rolling_details is None or rolling_details.empty:
    st.info("Run Rolling Universe Validation first. The event lab uses those historical selections.")
else:
    rel1, rel2, rel3 = st.columns(3)
    with rel1:
        relative_periods = st.select_slider(
            "Periods to test",
            options=[4, 8, 12],
            value=12,
            key="relative_periods",
        )
    with rel2:
        st.metric("Round-trip cost", f"{2 * (float(scalp_fee_bps) + float(scalp_slippage_bps)):.1f} bps")
    with rel3:
        st.metric("Execution", "Next 5m open")

    relative_signature = (
        saved_rolling_signature,
        int(rolling_forward),
        int(relative_periods),
        float(scalp_fee_bps),
        float(scalp_slippage_bps),
    )

    run_relative = st.button(
        "Run BTC-relative event lab",
        use_container_width=True,
        key="run_relative_event_lab",
    )

    if run_relative:
        with st.spinner(
            "Testing volume shocks, BTC-relative momentum, decorrelation breakouts and capitulation..."
        ):
            try:
                relative_summary, relative_period_table, relative_details = run_relative_event_lab(
                    rolling_details,
                    forward_days=int(rolling_forward),
                    fee_bps=float(scalp_fee_bps),
                    slippage_bps=float(scalp_slippage_bps),
                    max_periods=int(relative_periods),
                )
                st.session_state["relative_summary"] = relative_summary
                st.session_state["relative_period_table"] = relative_period_table
                st.session_state["relative_details"] = relative_details
                st.session_state["relative_signature"] = relative_signature
            except Exception as exc:
                st.error(f"BTC-relative event lab failed: {exc}")

    relative_summary = st.session_state.get("relative_summary")
    relative_period_table = st.session_state.get("relative_period_table")
    relative_details = st.session_state.get("relative_details")
    saved_relative_signature = st.session_state.get("relative_signature")

    if relative_summary is not None:
        if saved_relative_signature != relative_signature:
            st.info("Event-lab settings changed. Run it again to refresh the results.")
        else:
            shown_relative = relative_summary.copy()
            for col in [
                "Avg gross return %",
                "Avg net return %",
                "Median net return %",
                "Compounded net %",
                "Gross avg trade bps",
                "Net avg trade bps",
                "Round-trip cost bps",
                "Gross edge / cost",
                "Avg net PF",
                "Worst period %",
                "Best period %",
            ]:
                if col in shown_relative:
                    shown_relative[col] = shown_relative[col].round(3)

            st.dataframe(
                shown_relative,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Gross avg trade bps is the key diagnostic. With the current cost assumption, a signal whose "
                "gross expectancy per trade does not comfortably exceed round-trip cost is unlikely to survive "
                "fees and slippage. Gross edge / cost above 1 only means breakeven-before-variance, not robustness."
            )

            if relative_period_table is not None and not relative_period_table.empty:
                with st.expander("Event-lab weekly portfolio results"):
                    shown_relative_periods = relative_period_table.copy()
                    for col in [
                        "Gross portfolio return %",
                        "Net portfolio return %",
                        "Cost drag pp",
                        "Gross avg trade bps",
                        "Net avg trade bps",
                    ]:
                        if col in shown_relative_periods:
                            shown_relative_periods[col] = shown_relative_periods[col].round(3)
                    st.dataframe(
                        shown_relative_periods,
                        use_container_width=True,
                        hide_index=True,
                    )

            if relative_details is not None and not relative_details.empty:
                with st.expander("Event-lab per-coin results"):
                    shown_relative_details = relative_details.copy()
                    for col in [
                        "Gross return %",
                        "Net return %",
                        "Cost drag pp",
                        "Gross PF",
                        "Net PF",
                        "Gross avg trade bps",
                        "Net avg trade bps",
                        "Win rate %",
                        "Max DD %",
                    ]:
                        if col in shown_relative_details:
                            shown_relative_details[col] = shown_relative_details[col].round(3)
                    st.dataframe(
                        shown_relative_details,
                        use_container_width=True,
                        hide_index=True,
                    )

st.subheader("Event Edge Study · v1.6")
st.caption(
    "This study isolates the raw predictive value of BTC-relative breakout events. "
    "It uses next-bar entry, a 60-minute per-symbol cooldown, and fixed exits after 15/30/60 minutes. "
    "The goal is to see whether stronger event quality actually produces gross expectancy above trading costs."
)

if rolling_details is None or rolling_details.empty:
    st.info("Run Rolling Universe Validation first. The event study uses those historical selections.")
else:
    ee1, ee2, ee3 = st.columns(3)
    with ee1:
        edge_study_periods = st.select_slider(
            "Periods to study",
            options=[4, 8, 12],
            value=12,
            key="edge_study_periods",
        )
    with ee2:
        st.metric("Round-trip cost", f"{2 * (float(scalp_fee_bps) + float(scalp_slippage_bps)):.1f} bps")
    with ee3:
        st.metric("Event cooldown", "60 min")

    edge_study_signature = (
        saved_rolling_signature,
        int(rolling_forward),
        int(edge_study_periods),
        float(scalp_fee_bps),
        float(scalp_slippage_bps),
    )

    run_edge_study = st.button(
        "Run event edge study",
        use_container_width=True,
        key="run_event_edge_study",
    )

    if run_edge_study:
        with st.spinner(
            "Measuring next-bar forward returns after base, strong and extreme breakout events..."
        ):
            try:
                event_edge_summary, event_edge_details = run_event_edge_study(
                    rolling_details,
                    forward_days=int(rolling_forward),
                    fee_bps=float(scalp_fee_bps),
                    slippage_bps=float(scalp_slippage_bps),
                    max_periods=int(edge_study_periods),
                )
                st.session_state["event_edge_summary"] = event_edge_summary
                st.session_state["event_edge_details"] = event_edge_details
                st.session_state["event_edge_signature"] = edge_study_signature
            except Exception as exc:
                st.error(f"Event edge study failed: {exc}")

    event_edge_summary = st.session_state.get("event_edge_summary")
    event_edge_details = st.session_state.get("event_edge_details")
    saved_event_edge_signature = st.session_state.get("event_edge_signature")

    if event_edge_summary is not None:
        if saved_event_edge_signature != edge_study_signature:
            st.info("Event-study settings changed. Run it again to refresh the results.")
        else:
            shown_event_edge = event_edge_summary.copy()
            for col in [
                "Gross avg bps",
                "Gross median bps",
                "Net avg bps",
                "Net positive events %",
                "LONG gross avg bps",
                "SHORT gross avg bps",
                "Round-trip cost bps",
                "Gross edge / cost",
            ]:
                if col in shown_event_edge:
                    shown_event_edge[col] = shown_event_edge[col].round(2)

            st.dataframe(
                shown_event_edge,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "The key question is whether stricter event tiers raise gross average bps above the cost line. "
                "A small number of events can look impressive by chance, so event count and positive periods matter too."
            )

            if event_edge_details is not None and not event_edge_details.empty:
                with st.expander("Event-level details"):
                    shown_event_details = event_edge_details.copy()
                    for col in [
                        "Gross bps",
                        "Net bps",
                        "Volume ratio",
                        "Range / ATR",
                        "BTC corr",
                        "Relative move / ATR",
                    ]:
                        if col in shown_event_details:
                            shown_event_details[col] = shown_event_details[col].round(2)
                    st.dataframe(
                        shown_event_details,
                        use_container_width=True,
                        hide_index=True,
                    )

st.subheader("Direction × BTC Regime Study · v1.7")
st.caption(
    "The v1.6 event study showed a clear LONG/SHORT asymmetry. This study keeps the same fixed breakout event "
    "definition and splits results by trade direction and BTC regime. BTC regime is classified from trailing "
    "24h return plus 5m EMA structure; no future bars are used for regime classification."
)

if rolling_details is None or rolling_details.empty:
    st.info("Run Rolling Universe Validation first. The regime study uses those historical selections.")
else:
    dr1, dr2, dr3 = st.columns(3)
    with dr1:
        regime_periods = st.select_slider(
            "Periods to study",
            options=[4, 8, 12],
            value=12,
            key="regime_periods",
        )
    with dr2:
        st.metric("Round-trip cost", f"{2 * (float(scalp_fee_bps) + float(scalp_slippage_bps)):.1f} bps")
    with dr3:
        st.metric("Event cooldown", "60 min")

    regime_signature = (
        saved_rolling_signature,
        int(rolling_forward),
        int(regime_periods),
        float(scalp_fee_bps),
        float(scalp_slippage_bps),
    )

    run_regime = st.button(
        "Run direction × BTC regime study",
        use_container_width=True,
        key="run_direction_regime_study",
    )

    if run_regime:
        with st.spinner("Splitting breakout events by LONG/SHORT and BTC regime..."):
            try:
                regime_summary, regime_details = run_direction_regime_study(
                    rolling_details,
                    forward_days=int(rolling_forward),
                    fee_bps=float(scalp_fee_bps),
                    slippage_bps=float(scalp_slippage_bps),
                    max_periods=int(regime_periods),
                )
                st.session_state["regime_summary"] = regime_summary
                st.session_state["regime_details"] = regime_details
                st.session_state["regime_signature"] = regime_signature
            except Exception as exc:
                st.error(f"Direction × BTC regime study failed: {exc}")

    regime_summary = st.session_state.get("regime_summary")
    regime_details = st.session_state.get("regime_details")
    saved_regime_signature = st.session_state.get("regime_signature")

    if regime_summary is not None:
        if saved_regime_signature != regime_signature:
            st.info("Regime-study settings changed. Run it again to refresh the results.")
        else:
            shown_regime = regime_summary.copy()
            for col in [
                "Gross avg bps",
                "Gross median bps",
                "Net avg bps",
                "Net positive events %",
                "Round-trip cost bps",
                "Gross edge / cost",
            ]:
                if col in shown_regime:
                    shown_regime[col] = shown_regime[col].round(2)

            st.dataframe(
                shown_regime,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Because the previous 12 weeks led us to investigate direction/regime asymmetry, these rows are "
                "development diagnostics rather than a final holdout result. A promising pattern must later be "
                "tested on a different historical window."
            )

            if regime_details is not None and not regime_details.empty:
                with st.expander("Direction/regime event details"):
                    shown_regime_details = regime_details.copy()
                    for col in [
                        "Gross bps",
                        "Net bps",
                        "BTC corr",
                        "Volume ratio",
                        "Range / ATR",
                        "Relative move / ATR",
                    ]:
                        if col in shown_regime_details:
                            shown_regime_details[col] = shown_regime_details[col].round(2)
                    st.dataframe(
                        shown_regime_details,
                        use_container_width=True,
                        hide_index=True,
                    )

st.subheader("Alt-Basket Relative Study · v1.8")
st.caption(
    "BTC is no longer used as the market regime reference here. For each coin, context is built from "
    "the OTHER selected active alts in that historical period: basket breadth, basket 60-minute return, "
    "basket activity and the coin's relative move versus that alt basket."
)

if rolling_details is None or rolling_details.empty:
    st.info("Run Rolling Universe Validation first. The alt-basket study uses those historical selections.")
else:
    ab1, ab2, ab3 = st.columns(3)
    with ab1:
        alt_basket_periods = st.select_slider(
            "Periods to study",
            options=[4, 8, 12],
            value=12,
            key="alt_basket_periods",
        )
    with ab2:
        st.metric("Round-trip cost", f"{2 * (float(scalp_fee_bps) + float(scalp_slippage_bps)):.1f} bps")
    with ab3:
        st.metric("Reference market", "Selected alt basket")

    alt_basket_signature = (
        saved_rolling_signature,
        int(rolling_forward),
        int(alt_basket_periods),
        float(scalp_fee_bps),
        float(scalp_slippage_bps),
    )

    run_alt_basket = st.button(
        "Run alt-basket relative study",
        use_container_width=True,
        key="run_alt_basket_study",
    )

    if run_alt_basket:
        with st.spinner(
            "Building cross-sectional alt breadth and measuring breakout events relative to the alt basket..."
        ):
            try:
                alt_basket_summary, alt_basket_details = run_alt_basket_study(
                    rolling_details,
                    forward_days=int(rolling_forward),
                    fee_bps=float(scalp_fee_bps),
                    slippage_bps=float(scalp_slippage_bps),
                    max_periods=int(alt_basket_periods),
                )
                st.session_state["alt_basket_summary"] = alt_basket_summary
                st.session_state["alt_basket_details"] = alt_basket_details
                st.session_state["alt_basket_signature"] = alt_basket_signature
            except Exception as exc:
                st.error(f"Alt-basket relative study failed: {exc}")

    alt_basket_summary = st.session_state.get("alt_basket_summary")
    alt_basket_details = st.session_state.get("alt_basket_details")
    saved_alt_basket_signature = st.session_state.get("alt_basket_signature")

    if alt_basket_summary is not None:
        if saved_alt_basket_signature != alt_basket_signature:
            st.info("Alt-basket study settings changed. Run it again to refresh the results.")
        else:
            shown_alt_basket = alt_basket_summary.copy()
            for col in [
                "Gross avg bps",
                "Gross median bps",
                "Net avg bps",
                "Net positive events %",
                "Round-trip cost bps",
                "Gross edge / cost",
            ]:
                if col in shown_alt_basket:
                    shown_alt_basket[col] = shown_alt_basket[col].round(2)

            st.dataframe(
                shown_alt_basket,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "This is the cleaner test for your idea: the coin is judged versus other currently active alts, "
                "not versus BTC. A potentially useful row should have enough events across multiple periods and "
                "gross average bps comfortably above the cost line."
            )

            if alt_basket_details is not None and not alt_basket_details.empty:
                with st.expander("Alt-basket event details"):
                    shown_alt_details = alt_basket_details.copy()
                    for col in [
                        "Gross bps",
                        "Net bps",
                        "Breadth",
                        "Basket 60m return %",
                        "Basket volume ratio",
                        "Coin volume ratio",
                        "Range / ATR",
                        "Relative move / ATR",
                    ]:
                        if col in shown_alt_details:
                            shown_alt_details[col] = shown_alt_details[col].round(2)
                    st.dataframe(
                        shown_alt_details,
                        use_container_width=True,
                        hide_index=True,
                    )

st.subheader("Frozen Historical Holdout · v1.9")
st.caption(
    "We freeze the pattern observed in v1.8 and test it on an older, non-overlapping window. "
    "No thresholds or horizons are optimized inside this block. Fixed rules: LONG when a coin breaks up "
    "against an alt-BEAR basket and exit after 30m; SHORT when a coin breaks down against an alt-BULL basket "
    "and exit after 15m."
)

ho1, ho2, ho3 = st.columns(3)
with ho1:
    holdout_horizon = st.select_slider(
        "Older holdout length",
        options=[120, 180],
        value=180,
        format_func=lambda x: f"{x} days",
        key="holdout_horizon",
    )
with ho2:
    st.metric("Gap from development window", "90 days")
with ho3:
    st.metric("Rules", "Frozen")

hc1, hc2 = st.columns(2)
with hc1:
    holdout_fee_bps = st.number_input(
        "Holdout fee per side, bps",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.5,
        key="holdout_fee_bps",
    )
with hc2:
    holdout_slippage_bps = st.number_input(
        "Holdout slippage per side, bps",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.5,
        key="holdout_slippage_bps",
    )

holdout_signature = (
    int(holdout_horizon),
    int(rolling_lookback),
    int(rolling_forward),
    int(rolling_pool),
    int(rolling_top_n),
    float(rolling_min_turnover_m),
    float(holdout_fee_bps),
    float(holdout_slippage_bps),
)

run_holdout = st.button(
    "Run frozen historical holdout",
    use_container_width=True,
    key="run_counter_regime_holdout",
)

if run_holdout:
    with st.spinner(
        "Rebuilding the older historical universe and testing the frozen counter-regime rules..."
    ):
        try:
            holdout_summary, holdout_events, holdout_periods, holdout_universe_summary = run_counter_regime_holdout(
                horizon_days=int(holdout_horizon),
                end_offset_days=90,
                lookback_days=int(rolling_lookback),
                forward_days=int(rolling_forward),
                pool_size=int(rolling_pool),
                select_top_n=int(rolling_top_n),
                min_daily_turnover_usd=float(rolling_min_turnover_m) * 1_000_000.0,
                fee_bps=float(holdout_fee_bps),
                slippage_bps=float(holdout_slippage_bps),
            )
            st.session_state["holdout_summary"] = holdout_summary
            st.session_state["holdout_events"] = holdout_events
            st.session_state["holdout_periods"] = holdout_periods
            st.session_state["holdout_universe_summary"] = holdout_universe_summary
            st.session_state["holdout_signature"] = holdout_signature
        except Exception as exc:
            st.error(f"Frozen historical holdout failed: {exc}")

holdout_summary = st.session_state.get("holdout_summary")
holdout_events = st.session_state.get("holdout_events")
holdout_periods = st.session_state.get("holdout_periods")
holdout_universe_summary = st.session_state.get("holdout_universe_summary")
saved_holdout_signature = st.session_state.get("holdout_signature")

if holdout_summary is not None:
    if saved_holdout_signature != holdout_signature:
        st.info("Holdout settings changed. Run the frozen holdout again.")
    else:
        if holdout_universe_summary is not None:
            h1, h2, h3 = st.columns(3)
            holdout_period_count = int(holdout_universe_summary.get("periods", 0))
            holdout_positive = int(
                holdout_universe_summary.get("positive_vol_uplift_periods", 0)
            )
            h1.metric("Holdout universe periods", holdout_period_count)
            h2.metric(
                "Positive universe vol uplift",
                f"{holdout_positive}/{holdout_period_count}" if holdout_period_count else "0/0",
            )
            h3.metric(
                "Universe avg vol uplift",
                f"{holdout_universe_summary.get('avg_vol_uplift_pct', 0.0):.1f}%",
            )

        shown_holdout = holdout_summary.copy()
        for col in [
            "Gross avg bps",
            "Gross median bps",
            "Trimmed gross avg bps",
            "Net avg bps",
            "Net median bps",
            "Net positive events %",
            "Worst period avg bps",
            "Best period avg bps",
            "Round-trip cost bps",
            "Gross edge / cost",
        ]:
            if col in shown_holdout:
                shown_holdout[col] = shown_holdout[col].round(2)

        st.dataframe(
            shown_holdout,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "This is the important robustness check. Because the rules were fixed before this older window was "
            "examined, these results are much more informative than another optimization on the recent 12 weeks. "
            "Trimmed gross average removes the most extreme 10% tails on each side."
        )

        if holdout_events is not None and not holdout_events.empty:
            with st.expander("Historical holdout event details"):
                shown_holdout_events = holdout_events.copy()
                for col in [
                    "Gross bps",
                    "Net bps",
                    "Breadth",
                    "Coin volume ratio",
                    "Range / ATR",
                    "Relative move / ATR",
                ]:
                    if col in shown_holdout_events:
                        shown_holdout_events[col] = shown_holdout_events[col].round(2)
                st.dataframe(
                    shown_holdout_events,
                    use_container_width=True,
                    hide_index=True,
                )

st.subheader("Scalping Edge Map · v2.0")
st.caption(
    "Multi-timeframe research: the Rolling Universe chooses active alts, 5m builds the alt-basket context, "
    "and 1m is used for execution and forward-move measurement. A 5m event is only tradable after that "
    "5m candle closes; entry is therefore at a later 1m open, not inside the signal candle."
)

if rolling_details is None or rolling_details.empty:
    st.info("Run Rolling Universe Validation first. The 1m edge map uses those historical selections.")
else:
    em1, em2, em3 = st.columns(3)
    with em1:
        edge_map_periods = st.select_slider(
            "Periods to map",
            options=[2, 4, 8, 12],
            value=4,
            key="edge_map_periods",
        )
    with em2:
        edge_map_fee_bps = st.number_input(
            "1m fee per side, bps",
            min_value=0.0,
            max_value=20.0,
            value=4.0,
            step=0.5,
            key="edge_map_fee_bps",
        )
    with em3:
        edge_map_slippage_bps = st.number_input(
            "1m slippage per side, bps",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5,
            key="edge_map_slippage_bps",
        )

    st.caption(
        "Start with 4 periods: 1m history is much heavier than 5m. After the first result, "
        "increase to 8 or 12 periods for a longer-distance check. Raw 1m downloads are cached in-process."
    )

    edge_map_signature = (
        saved_rolling_signature,
        int(rolling_forward),
        int(edge_map_periods),
        float(edge_map_fee_bps),
        float(edge_map_slippage_bps),
    )

    run_edge_map = st.button(
        "Run 1m Scalping Edge Map",
        use_container_width=True,
        key="run_scalping_edge_map",
    )

    if run_edge_map:
        with st.spinner(
            "Downloading 1m candles, rebuilding 5m alt context and measuring 1/3/5/10/15/30m forward edge..."
        ):
            try:
                edge_map_summary, edge_map_details = run_scalping_edge_map(
                    rolling_details,
                    forward_days=int(rolling_forward),
                    fee_bps=float(edge_map_fee_bps),
                    slippage_bps=float(edge_map_slippage_bps),
                    max_periods=int(edge_map_periods),
                )
                st.session_state["edge_map_summary"] = edge_map_summary
                st.session_state["edge_map_details"] = edge_map_details
                st.session_state["edge_map_signature"] = edge_map_signature
            except Exception as exc:
                st.error(f"1m Scalping Edge Map failed: {exc}")

    edge_map_summary = st.session_state.get("edge_map_summary")
    edge_map_details = st.session_state.get("edge_map_details")
    saved_edge_map_signature = st.session_state.get("edge_map_signature")

    if edge_map_summary is not None:
        if saved_edge_map_signature != edge_map_signature:
            st.info("1m Edge Map settings changed. Run it again to refresh the results.")
        else:
            ef1, ef2, ef3 = st.columns(3)
            with ef1:
                entry_mode_filter = st.multiselect(
                    "Entry mode",
                    options=edge_map_summary["Entry mode"].dropna().unique().tolist(),
                    default=edge_map_summary["Entry mode"].dropna().unique().tolist(),
                    key="edge_map_entry_filter",
                )
            with ef2:
                side_filter = st.multiselect(
                    "Side",
                    options=edge_map_summary["Side"].dropna().unique().tolist(),
                    default=edge_map_summary["Side"].dropna().unique().tolist(),
                    key="edge_map_side_filter",
                )
            with ef3:
                regime_filter = st.multiselect(
                    "Alt regime",
                    options=edge_map_summary["Alt regime"].dropna().unique().tolist(),
                    default=edge_map_summary["Alt regime"].dropna().unique().tolist(),
                    key="edge_map_regime_filter",
                )

            shown_edge_map = edge_map_summary[
                edge_map_summary["Entry mode"].isin(entry_mode_filter)
                & edge_map_summary["Side"].isin(side_filter)
                & edge_map_summary["Alt regime"].isin(regime_filter)
            ].copy()

            horizon_order = {"1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "30m": 30}
            shown_edge_map["_h"] = shown_edge_map["Horizon"].map(horizon_order)
            shown_edge_map = shown_edge_map.sort_values(
                ["Entry mode", "Side", "Alt regime", "_h"]
            ).drop(columns=["_h"])

            for col in [
                "Gross avg bps",
                "Gross median bps",
                "Trimmed gross avg bps",
                "Net avg bps",
                "Net positive events %",
                "Avg MFE bps",
                "Avg MAE bps",
                "Round-trip cost bps",
                "Gross edge / cost",
            ]:
                if col in shown_edge_map:
                    shown_edge_map[col] = shown_edge_map[col].round(2)

            st.dataframe(
                shown_edge_map,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Immediate next 1m enters at the first 1m open after the 5m context candle has closed. "
                "1m volume confirm waits up to 3 completed 1m candles for same-direction movement with "
                "volume >= 1.5× its 20-minute average, then enters at the following 1m open. "
                "MFE/MAE show favorable/adverse excursion after entry."
            )

            if edge_map_details is not None and not edge_map_details.empty:
                with st.expander("1m event-level details"):
                    shown_edge_details = edge_map_details.copy()
                    for col in [
                        "Gross bps",
                        "Net bps",
                        "MFE bps",
                        "MAE bps",
                        "Breadth",
                        "Coin volume ratio 5m",
                        "Range / ATR 5m",
                        "Relative move / ATR 5m",
                    ]:
                        if col in shown_edge_details:
                            shown_edge_details[col] = shown_edge_details[col].round(2)
                    st.dataframe(
                        shown_edge_details,
                        use_container_width=True,
                        hide_index=True,
                    )

st.subheader("1m Exit Surface · v2.1")
st.caption(
    "The v2.0 12-period result kept some LONG/BULL edge, but fixed 10-minute exits were unstable. "
    "This development study freezes the signal itself and varies only a coarse 1m stop/take grid. "
    "Candidate signal: LONG + alt-BULL + 1m volume confirmation. Maximum holding time stays fixed at 10 minutes."
)

if rolling_details is None or rolling_details.empty:
    st.info("Run Rolling Universe Validation first. The exit study uses the same historical selections.")
else:
    xs1, xs2, xs3 = st.columns(3)
    with xs1:
        exit_surface_periods = st.select_slider(
            "Periods to study",
            options=[4, 8, 12],
            value=12,
            key="exit_surface_periods",
        )
    with xs2:
        exit_surface_fee = st.number_input(
            "Exit-study fee per side, bps",
            min_value=0.0,
            max_value=20.0,
            value=4.0,
            step=0.5,
            key="exit_surface_fee",
        )
    with xs3:
        exit_surface_slippage = st.number_input(
            "Exit-study slippage per side, bps",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5,
            key="exit_surface_slippage",
        )

    exit_surface_signature = (
        saved_rolling_signature,
        int(rolling_forward),
        int(exit_surface_periods),
        float(exit_surface_fee),
        float(exit_surface_slippage),
    )

    run_exit_surface_button = st.button(
        "Run 1m exit surface",
        use_container_width=True,
        key="run_exit_surface",
    )

    if run_exit_surface_button:
        with st.spinner(
            "Reusing the fixed LONG/BULL 1m-confirm entries and testing a coarse stop/take grid..."
        ):
            try:
                exit_surface_summary, exit_surface_details = run_exit_surface(
                    rolling_details,
                    forward_days=int(rolling_forward),
                    fee_bps=float(exit_surface_fee),
                    slippage_bps=float(exit_surface_slippage),
                    max_periods=int(exit_surface_periods),
                )
                st.session_state["exit_surface_summary"] = exit_surface_summary
                st.session_state["exit_surface_details"] = exit_surface_details
                st.session_state["exit_surface_signature"] = exit_surface_signature
            except Exception as exc:
                st.error(f"1m Exit Surface failed: {exc}")

    exit_surface_summary = st.session_state.get("exit_surface_summary")
    exit_surface_details = st.session_state.get("exit_surface_details")
    saved_exit_surface_signature = st.session_state.get("exit_surface_signature")

    if exit_surface_summary is not None:
        if saved_exit_surface_signature != exit_surface_signature:
            st.info("Exit-study settings changed. Run it again to refresh the results.")
        else:
            shown_exit_surface = exit_surface_summary.copy()

            for col in [
                "Gross avg bps",
                "Gross median bps",
                "Trimmed gross avg bps",
                "Net avg bps",
                "Net median bps",
                "Net positive events %",
                "Worst period avg bps",
                "Best period avg bps",
                "Take exits %",
                "Stop exits %",
                "Time exits %",
                "Avg hold min",
                "Round-trip cost bps",
            ]:
                if col in shown_exit_surface:
                    shown_exit_surface[col] = shown_exit_surface[col].round(2)

            baseline = shown_exit_surface[shown_exit_surface["Config"] == "Fixed 10m"]
            grid = shown_exit_surface[shown_exit_surface["Config"] != "Fixed 10m"].sort_values(
                ["Net avg bps", "Trimmed gross avg bps"],
                ascending=[False, False],
            )
            shown_exit_surface = pd.concat([baseline, grid], ignore_index=True)

            st.dataframe(
                shown_exit_surface,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "This is still development data, so the top row in the stop/take grid is not a validated strategy. "
                "We are looking for a broad, stable region: positive net expectancy, positive trimmed gross, "
                "reasonable period consistency, and neighboring stop/take settings with similar behavior. "
                "If one isolated parameter pair looks exceptional, treat it as likely overfit."
            )

            if exit_surface_details is not None and not exit_surface_details.empty:
                with st.expander("Exit-surface event details"):
                    shown_exit_details = exit_surface_details.copy()
                    for col in ["Gross bps", "Net bps"]:
                        if col in shown_exit_details:
                            shown_exit_details[col] = shown_exit_details[col].round(2)
                    st.dataframe(
                        shown_exit_details,
                        use_container_width=True,
                        hide_index=True,
                    )

st.subheader("1m Entry Quality Lab · v2.2")
st.caption(
    "v2.1 showed that changing exits did not rescue the signal. This lab freezes the 10-minute exit "
    "and asks a cleaner question: do stronger pre-entry conditions improve expectancy? "
    "All quality filters use only information known before the 1m entry."
)

if rolling_details is None or rolling_details.empty:
    st.info("Run Rolling Universe Validation first. The entry-quality lab uses the same historical selections.")
else:
    eq1, eq2, eq3 = st.columns(3)
    with eq1:
        entry_quality_periods = st.select_slider(
            "Periods to study",
            options=[4, 8, 12],
            value=12,
            key="entry_quality_periods",
        )
    with eq2:
        entry_quality_fee = st.number_input(
            "Entry-quality fee per side, bps",
            min_value=0.0,
            max_value=20.0,
            value=4.0,
            step=0.5,
            key="entry_quality_fee",
        )
    with eq3:
        entry_quality_slippage = st.number_input(
            "Entry-quality slippage per side, bps",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5,
            key="entry_quality_slippage",
        )

    entry_quality_signature = (
        saved_rolling_signature,
        int(rolling_forward),
        int(entry_quality_periods),
        float(entry_quality_fee),
        float(entry_quality_slippage),
    )

    run_entry_quality = st.button(
        "Run 1m entry-quality lab",
        use_container_width=True,
        key="run_entry_quality_lab",
    )

    if run_entry_quality:
        with st.spinner(
            "Testing baseline, strong 1m confirmation, strong 5m event and dual-strong subsets..."
        ):
            try:
                entry_quality_summary, entry_quality_details = run_entry_quality_lab(
                    rolling_details,
                    forward_days=int(rolling_forward),
                    fee_bps=float(entry_quality_fee),
                    slippage_bps=float(entry_quality_slippage),
                    max_periods=int(entry_quality_periods),
                )
                st.session_state["entry_quality_summary"] = entry_quality_summary
                st.session_state["entry_quality_details"] = entry_quality_details
                st.session_state["entry_quality_signature"] = entry_quality_signature
            except Exception as exc:
                st.error(f"1m Entry Quality Lab failed: {exc}")

    entry_quality_summary = st.session_state.get("entry_quality_summary")
    entry_quality_details = st.session_state.get("entry_quality_details")
    saved_entry_quality_signature = st.session_state.get("entry_quality_signature")

    if entry_quality_summary is not None:
        if saved_entry_quality_signature != entry_quality_signature:
            st.info("Entry-quality settings changed. Run it again to refresh the results.")
        else:
            shown_entry_quality = entry_quality_summary.copy()
            for col in [
                "Gross avg bps",
                "Gross median bps",
                "Trimmed gross avg bps",
                "Net avg bps",
                "Net median bps",
                "Net positive events %",
                "Worst period avg bps",
                "Best period avg bps",
                "Avg 1m confirm vol",
                "Avg 1m body bps",
                "Avg 5m event vol",
                "Avg 5m relative / ATR",
                "Round-trip cost bps",
            ]:
                if col in shown_entry_quality:
                    shown_entry_quality[col] = shown_entry_quality[col].round(2)

            st.dataframe(
                shown_entry_quality,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Do not choose a tier only because it has the highest mean. We want enough events across many periods, "
                "positive trimmed gross above costs, and a median/period profile that improves together. "
                "If a tiny Dual-strong sample looks spectacular, treat it as exploratory until an older holdout confirms it."
            )

            if entry_quality_details is not None and not entry_quality_details.empty:
                with st.expander("Entry-quality event details"):
                    shown_entry_details = entry_quality_details.copy()
                    for col in [
                        "Gross bps",
                        "Net bps",
                        "1m confirm volume ratio",
                        "1m confirm body bps",
                        "5m volume ratio",
                        "5m range / ATR",
                        "5m relative move / ATR",
                        "Breadth",
                    ]:
                        if col in shown_entry_details:
                            shown_entry_details[col] = shown_entry_details[col].round(2)
                    st.dataframe(
                        shown_entry_details,
                        use_container_width=True,
                        hide_index=True,
                    )

st.subheader("Frozen 1m Entry Holdout · v2.3")
st.caption(
    "The v2.2 development sample produced two stronger entry-quality subsets. "
    "This block freezes those rules and tests them on an older window that ends 270 days before today, "
    "separate from both the recent 90-day development sample and the older v1.9 window. "
    "Signal, thresholds and 10-minute exit are not retuned inside this holdout."
)

fh1, fh2, fh3 = st.columns(3)
with fh1:
    frozen_holdout_days = st.select_slider(
        "Frozen holdout length",
        options=[90, 120, 180],
        value=120,
        format_func=lambda x: f"{x} days",
        key="frozen_1m_holdout_days",
    )
with fh2:
    st.metric("Window ends", "270 days ago")
with fh3:
    st.metric("Exit", "Fixed 10m")

fc1, fc2 = st.columns(2)
with fc1:
    frozen_fee = st.number_input(
        "Frozen holdout fee per side, bps",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.5,
        key="frozen_1m_fee",
    )
with fc2:
    frozen_slippage = st.number_input(
        "Frozen holdout slippage per side, bps",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.5,
        key="frozen_1m_slippage",
    )

frozen_1m_signature = (
    int(frozen_holdout_days),
    int(rolling_lookback),
    int(rolling_forward),
    int(rolling_pool),
    int(rolling_top_n),
    float(rolling_min_turnover_m),
    float(frozen_fee),
    float(frozen_slippage),
)

run_frozen_1m = st.button(
    "Run frozen 1m entry holdout",
    use_container_width=True,
    key="run_frozen_1m_entry_holdout",
)

if run_frozen_1m:
    with st.spinner(
        "Rebuilding an older rolling universe and downloading its 1m data. This is the heaviest validation run so far..."
    ):
        try:
            frozen_1m_summary, frozen_1m_details, frozen_1m_periods, frozen_1m_universe_summary = run_frozen_1m_entry_holdout(
                horizon_days=int(frozen_holdout_days),
                end_offset_days=270,
                lookback_days=int(rolling_lookback),
                forward_days=int(rolling_forward),
                pool_size=int(rolling_pool),
                select_top_n=int(rolling_top_n),
                min_daily_turnover_usd=float(rolling_min_turnover_m) * 1_000_000.0,
                fee_bps=float(frozen_fee),
                slippage_bps=float(frozen_slippage),
            )
            st.session_state["frozen_1m_summary"] = frozen_1m_summary
            st.session_state["frozen_1m_details"] = frozen_1m_details
            st.session_state["frozen_1m_periods"] = frozen_1m_periods
            st.session_state["frozen_1m_universe_summary"] = frozen_1m_universe_summary
            st.session_state["frozen_1m_signature"] = frozen_1m_signature
        except Exception as exc:
            st.error(f"Frozen 1m holdout failed: {exc}")

frozen_1m_summary = st.session_state.get("frozen_1m_summary")
frozen_1m_details = st.session_state.get("frozen_1m_details")
frozen_1m_universe_summary = st.session_state.get("frozen_1m_universe_summary")
saved_frozen_1m_signature = st.session_state.get("frozen_1m_signature")

if frozen_1m_summary is not None:
    if saved_frozen_1m_signature != frozen_1m_signature:
        st.info("Frozen 1m holdout settings changed. Run it again.")
    else:
        if frozen_1m_universe_summary is not None:
            fu1, fu2, fu3 = st.columns(3)
            fp = int(frozen_1m_universe_summary.get("periods", 0))
            fpos = int(frozen_1m_universe_summary.get("positive_vol_uplift_periods", 0))
            fu1.metric("Holdout universe periods", fp)
            fu2.metric(
                "Positive universe vol uplift",
                f"{fpos}/{fp}" if fp else "0/0",
            )
            fu3.metric(
                "Universe avg vol uplift",
                f"{frozen_1m_universe_summary.get('avg_vol_uplift_pct', 0.0):.1f}%",
            )

        shown_frozen_1m = frozen_1m_summary.copy()
        for col in [
            "Gross avg bps",
            "Gross median bps",
            "Trimmed gross avg bps",
            "Net avg bps",
            "Net median bps",
            "Net positive events %",
            "Worst period avg bps",
            "Best period avg bps",
            "Avg 1m confirm vol",
            "Avg 1m body bps",
            "Avg 5m event vol",
            "Avg 5m relative / ATR",
            "Round-trip cost bps",
        ]:
            if col in shown_frozen_1m:
                shown_frozen_1m[col] = shown_frozen_1m[col].round(2)

        st.dataframe(
            shown_frozen_1m,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "The Baseline row is a control. Strong 1m confirm and Dual strong are the frozen v2.2 hypotheses. "
            "Do not tune thresholds from this table; if a frozen subset remains positive here, the next step is "
            "an execution/risk model built around that fixed signal rather than another search on this window."
        )

        if frozen_1m_details is not None and not frozen_1m_details.empty:
            with st.expander("Frozen 1m holdout event details"):
                shown_frozen_details = frozen_1m_details.copy()
                for col in [
                    "Gross bps",
                    "Net bps",
                    "1m confirm volume ratio",
                    "1m confirm body bps",
                    "5m volume ratio",
                    "5m range / ATR",
                    "5m relative move / ATR",
                    "Breadth",
                ]:
                    if col in shown_frozen_details:
                        shown_frozen_details[col] = shown_frozen_details[col].round(2)
                st.dataframe(
                    shown_frozen_details,
                    use_container_width=True,
                    hide_index=True,
                )

st.subheader("Strategy family benchmark · v0.6")
st.caption(
    "Four fixed reference strategies are compared on the same sequential future windows. "
    "No family is tuned or re-ranked using these future folds."
)

fam1, fam2, fam3 = st.columns(3)
with fam1:
    family_folds = st.slider(
        "Benchmark folds",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        key="family_folds",
    )
with fam2:
    family_calibration_pct = st.slider(
        "Benchmark calibration share",
        min_value=25,
        max_value=60,
        value=40,
        step=5,
        key="family_calibration_pct",
    )
with fam3:
    family_min_trades = st.number_input(
        "Minimum trades per benchmark fold",
        min_value=3,
        max_value=50,
        value=8,
        step=1,
        key="family_min_trades",
    )

family_signature = (
    symbol,
    timeframe,
    int(limit),
    int(rsi_period),
    int(atr_period),
    float(start_cash),
    float(fee_bps),
    float(slippage_bps),
    int(family_folds),
    int(family_calibration_pct),
    int(family_min_trades),
)

run_family_benchmark = st.button(
    "Run strategy family benchmark",
    use_container_width=True,
    key="run_family_benchmark",
)

if run_family_benchmark:
    with st.spinner("Comparing fixed strategy families across future folds..."):
        family_summary, family_details = benchmark_families(
            raw_df,
            folds=int(family_folds),
            calibration_pct=int(family_calibration_pct),
            rsi_period=int(rsi_period),
            atr_period=int(atr_period),
            start_cash=float(start_cash),
            fee_bps=float(fee_bps),
            slippage_bps=float(slippage_bps),
            min_fold_trades=int(family_min_trades),
        )
        st.session_state["family_summary"] = family_summary
        st.session_state["family_details"] = family_details
        st.session_state["family_signature"] = family_signature

family_summary = st.session_state.get("family_summary")
family_details = st.session_state.get("family_details")
saved_family_signature = st.session_state.get("family_signature")

if family_summary is not None:
    if saved_family_signature != family_signature:
        st.info("Benchmark settings changed. Run the strategy family benchmark again.")
    else:
        shown_family = family_summary.copy()
        for col in [
            "Median return %",
            "Average return %",
            "Worst fold %",
            "Best fold %",
            "Average PF",
            "Worst DD %",
        ]:
            if col in shown_family:
                shown_family[col] = shown_family[col].round(2)

        st.dataframe(shown_family, use_container_width=True, hide_index=True)
        st.caption(
            "Interpret the table as a robustness check: repeated positive folds with adequate trade counts "
            "are more informative than one isolated positive period."
        )

        if family_details is not None:
            with st.expander("Strategy family fold details"):
                shown_family_details = family_details.copy()
                for col in ["Return %", "PF", "DD %"]:
                    if col in shown_family_details:
                        shown_family_details[col] = shown_family_details[col].round(2)
                st.dataframe(shown_family_details, use_container_width=True, hide_index=True)

st.subheader("Mean Reversion Lab · v0.7")
st.caption(
    "Mean Reversion was the only v0.6 family with 2/3 positive folds and PF above 1 on average, "
    "but one bad fold erased most of the gain. These four fixed variants diagnose whether waiting "
    "for a reclaim and avoiding strong-trend/high-volatility regimes improves robustness. "
    "Because v0.6 already influenced this choice, these folds are development data, not a final untouched holdout."
)

mr1, mr2, mr3 = st.columns(3)
with mr1:
    mr_folds = st.slider(
        "MR folds",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        key="mr_folds",
    )
with mr2:
    mr_calibration_pct = st.slider(
        "MR calibration share",
        min_value=25,
        max_value=60,
        value=40,
        step=5,
        key="mr_calibration_pct",
    )
with mr3:
    mr_min_trades = st.number_input(
        "Minimum MR trades per fold",
        min_value=3,
        max_value=50,
        value=8,
        step=1,
        key="mr_min_trades",
    )

mr_signature = (
    symbol,
    timeframe,
    int(limit),
    int(rsi_period),
    int(atr_period),
    float(start_cash),
    float(fee_bps),
    float(slippage_bps),
    int(mr_folds),
    int(mr_calibration_pct),
    int(mr_min_trades),
)

run_mr_lab = st.button(
    "Run Mean Reversion Lab",
    use_container_width=True,
    key="run_mr_lab",
)

if run_mr_lab:
    with st.spinner("Testing fixed Mean Reversion variants across sequential folds..."):
        mr_summary, mr_details = evaluate_mean_reversion_variants(
            raw_df,
            folds=int(mr_folds),
            calibration_pct=int(mr_calibration_pct),
            rsi_period=int(rsi_period),
            atr_period=int(atr_period),
            start_cash=float(start_cash),
            fee_bps=float(fee_bps),
            slippage_bps=float(slippage_bps),
            min_fold_trades=int(mr_min_trades),
        )
        st.session_state["mr_summary"] = mr_summary
        st.session_state["mr_details"] = mr_details
        st.session_state["mr_signature"] = mr_signature

mr_summary = st.session_state.get("mr_summary")
mr_details = st.session_state.get("mr_details")
saved_mr_signature = st.session_state.get("mr_signature")

if mr_summary is not None:
    if saved_mr_signature != mr_signature:
        st.info("Mean Reversion Lab settings changed. Run it again to refresh the table.")
    else:
        shown_mr = mr_summary.copy()
        for col in [
            "Median return %",
            "Average return %",
            "Worst fold %",
            "Best fold %",
            "Average PF",
            "Worst DD %",
            "Long P&L",
            "Short P&L",
        ]:
            if col in shown_mr:
                shown_mr[col] = shown_mr[col].round(2)

        st.dataframe(shown_mr, use_container_width=True, hide_index=True)
        st.caption(
            "Focus on repeatability: multiple positive folds, enough trades, and whether one side "
            "(LONG or SHORT) is responsible for most of the losses."
        )

        if mr_details is not None:
            with st.expander("Mean Reversion fold details"):
                shown_mr_details = mr_details.copy()
                for col in ["Return %", "PF", "DD %", "Long P&L", "Short P&L"]:
                    if col in shown_mr_details:
                        shown_mr_details[col] = shown_mr_details[col].round(2)
                st.dataframe(shown_mr_details, use_container_width=True, hide_index=True)

st.subheader("Edge Diagnostics · v0.8")
st.caption(
    "This diagnostic does not optimize parameters. It keeps the fixed v0.7 Extreme-entry "
    "Mean Reversion rule and asks two questions: does the signal have any edge before costs, "
    "and is the weakness concentrated in LONG or SHORT trades?"
)

ed1, ed2, ed3 = st.columns(3)
with ed1:
    edge_folds = st.slider(
        "Diagnostic folds",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        key="edge_folds",
    )
with ed2:
    edge_calibration_pct = st.slider(
        "Diagnostic calibration share",
        min_value=25,
        max_value=60,
        value=40,
        step=5,
        key="edge_calibration_pct",
    )
with ed3:
    edge_min_trades = st.number_input(
        "Minimum trades per diagnostic fold",
        min_value=3,
        max_value=50,
        value=8,
        step=1,
        key="edge_min_trades",
    )

edge_signature = (
    symbol,
    timeframe,
    int(limit),
    int(rsi_period),
    int(atr_period),
    float(start_cash),
    float(fee_bps),
    float(slippage_bps),
    int(edge_folds),
    int(edge_calibration_pct),
    int(edge_min_trades),
)

run_edge = st.button(
    "Run edge diagnostics",
    use_container_width=True,
    key="run_edge_diagnostics",
)

if run_edge:
    with st.spinner("Separating gross edge, trading costs, LONG and SHORT..."):
        edge_summary, edge_details = evaluate_edge_diagnostics(
            raw_df,
            folds=int(edge_folds),
            calibration_pct=int(edge_calibration_pct),
            rsi_period=int(rsi_period),
            atr_period=int(atr_period),
            start_cash=float(start_cash),
            fee_bps=float(fee_bps),
            slippage_bps=float(slippage_bps),
            min_fold_trades=int(edge_min_trades),
        )
        st.session_state["edge_summary"] = edge_summary
        st.session_state["edge_details"] = edge_details
        st.session_state["edge_signature"] = edge_signature

edge_summary = st.session_state.get("edge_summary")
edge_details = st.session_state.get("edge_details")
saved_edge_signature = st.session_state.get("edge_signature")

if edge_summary is not None:
    if saved_edge_signature != edge_signature:
        st.info("Edge diagnostic settings changed. Run the diagnostics again.")
    else:
        shown_edge = edge_summary.copy()
        for col in [
            "Gross avg return %",
            "Net avg return %",
            "Net median return %",
            "Cost drag pp",
            "Gross avg PF",
            "Net avg PF",
            "Net worst fold %",
            "Net worst DD %",
        ]:
            if col in shown_edge:
                shown_edge[col] = shown_edge[col].round(2)

        st.dataframe(shown_edge, use_container_width=True, hide_index=True)
        st.caption(
            "If gross results are already negative, the entry rule itself lacks evidence of an edge. "
            "If gross is positive but net turns negative, costs/execution are a major part of the problem."
        )

        if edge_details is not None:
            with st.expander("Edge diagnostic fold details"):
                shown_edge_details = edge_details.copy()
                for col in [
                    "Gross return %",
                    "Net return %",
                    "Cost drag pp",
                    "Gross PF",
                    "Net PF",
                    "Net DD %",
                ]:
                    if col in shown_edge_details:
                        shown_edge_details[col] = shown_edge_details[col].round(2)
                st.dataframe(shown_edge_details, use_container_width=True, hide_index=True)

st.subheader("Parameter optimizer")
st.caption(
    "Quick search ranks combinations using TRAIN only, then reports TEST results without "
    "using TEST to choose the ranking. This is a coarse research tool, not a guarantee of future performance."
)

opt1, opt2 = st.columns([1, 1])
with opt1:
    optimizer_objective = st.selectbox(
        "TRAIN objective",
        ["Balanced", "Net return", "Profit factor"],
        index=0,
        key="optimizer_objective",
    )
with opt2:
    optimizer_top_n = st.slider(
        "Top TRAIN combinations to test",
        min_value=3,
        max_value=12,
        value=8,
        step=1,
        key="optimizer_top_n",
    )

sample1, sample2 = st.columns(2)
with sample1:
    min_train_trades = st.number_input(
        "Minimum TRAIN trades",
        min_value=5,
        max_value=200,
        value=25,
        step=5,
    )
with sample2:
    min_test_trades = st.number_input(
        "Minimum TEST trades",
        min_value=3,
        max_value=100,
        value=10,
        step=1,
    )

candidate_count = 324 if use_trend_filter else 108
st.caption(
    f"Quick grid: {candidate_count} combinations · EMA + RSI thresholds + ATR stop/take"
    + (" + Trend EMA." if use_trend_filter else ".")
    + f" TRAIN rows below {int(min_train_trades)} trades are heavily penalized; "
    + f"TEST rows below {int(min_test_trades)} trades are marked LOW SAMPLE."
)

optimizer_signature = (
    symbol,
    timeframe,
    int(limit),
    int(train_pct) if use_split else None,
    optimizer_objective,
    int(rsi_period),
    int(atr_period),
    bool(use_trend_filter),
    bool(use_volume_filter),
    float(min_volume_ratio),
    float(start_cash),
    float(fee_bps),
    float(slippage_bps),
    int(optimizer_top_n),
    int(min_train_trades),
    int(min_test_trades),
)

run_optimizer = st.button(
    "Run quick optimizer",
    type="primary",
    disabled=not use_split,
    use_container_width=True,
)

if not use_split:
    st.info("Enable Train / test split in the sidebar to use the optimizer.")

if run_optimizer:
    with st.spinner(f"Testing {candidate_count} TRAIN combinations..."):
        opt_results = optimize_quick(
            raw_df,
            split_idx,
            objective=optimizer_objective,
            rsi_period=int(rsi_period),
            atr_period=int(atr_period),
            use_trend_filter=bool(use_trend_filter),
            use_volume_filter=bool(use_volume_filter),
            min_volume_ratio=float(min_volume_ratio),
            start_cash=float(start_cash),
            fee_bps=float(fee_bps),
            slippage_bps=float(slippage_bps),
            min_train_trades=int(min_train_trades),
            min_test_trades=int(min_test_trades),
            top_n=int(optimizer_top_n),
        )
        st.session_state["optimizer_results"] = opt_results
        st.session_state["optimizer_signature"] = optimizer_signature

stored_results = st.session_state.get("optimizer_results")
stored_signature = st.session_state.get("optimizer_signature")

if stored_results is not None:
    if stored_signature != optimizer_signature:
        st.info("Optimizer settings changed. Run the optimizer again to refresh this table.")
    else:
        shown_opt = stored_results.copy()
        pct_cols = [
            "Train return %",
            "Train DD %",
            "Test return %",
            "Test DD %",
            "Test win %",
        ]
        for col in pct_cols:
            if col in shown_opt:
                shown_opt[col] = shown_opt[col].round(2)
        for col in ["Train PF", "Test PF", "Score"]:
            if col in shown_opt:
                shown_opt[col] = shown_opt[col].round(2)

        for col in ["Train PF", "Test PF"]:
            if col in shown_opt:
                shown_opt[col] = shown_opt[col].map(
                    lambda x: "∞" if x == float("inf") else f"{x:.2f}"
                )

        st.dataframe(shown_opt, use_container_width=True, hide_index=True)
        st.caption(
            "Rows stay ordered by TRAIN score. TEST columns are shown only as an out-of-sample check; "
            "they are not used to reorder the candidates."
        )

        first = stored_results.iloc[0]
        st.markdown(
            "**Top TRAIN-ranked parameters:** "
            f"EMA {int(first['EMA fast'])}/{int(first['EMA slow'])}, "
            f"Trend EMA {int(first['Trend EMA'])}, "
            f"RSI {int(first['RSI long'])}/{int(first['RSI short'])}, "
            f"Stop {first['Stop ATR']:.2f} ATR, Take {first['Take ATR']:.2f} ATR."
        )

        st.subheader("Walk-forward validation · v0.5")
        st.caption(
            "The same TRAIN-ranked candidates are checked across several sequential future windows. "
            "Their original TRAIN order is preserved; walk-forward results do not re-rank them."
        )

        wf1, wf2, wf3 = st.columns(3)
        with wf1:
            wf_folds = st.slider(
                "Future folds",
                min_value=2,
                max_value=5,
                value=3,
                step=1,
                key="wf_folds",
            )
        with wf2:
            wf_calibration_pct = st.slider(
                "Initial calibration share",
                min_value=25,
                max_value=60,
                value=40,
                step=5,
                key="wf_calibration_pct",
            )
        with wf3:
            wf_min_trades = st.number_input(
                "Minimum trades per fold",
                min_value=3,
                max_value=50,
                value=8,
                step=1,
                key="wf_min_trades",
            )

        wf_signature = (
            optimizer_signature,
            int(wf_folds),
            int(wf_calibration_pct),
            int(wf_min_trades),
        )

        run_wf = st.button(
            "Run walk-forward validation",
            use_container_width=True,
            key="run_walk_forward",
        )

        if run_wf:
            with st.spinner(f"Checking {len(stored_results)} candidates across {wf_folds} future folds..."):
                wf_summary, wf_details = evaluate_walk_forward(
                    raw_df,
                    stored_results,
                    folds=int(wf_folds),
                    calibration_pct=int(wf_calibration_pct),
                    rsi_period=int(rsi_period),
                    atr_period=int(atr_period),
                    use_trend_filter=bool(use_trend_filter),
                    use_volume_filter=bool(use_volume_filter),
                    min_volume_ratio=float(min_volume_ratio),
                    start_cash=float(start_cash),
                    fee_bps=float(fee_bps),
                    slippage_bps=float(slippage_bps),
                    min_fold_trades=int(wf_min_trades),
                )
                st.session_state["wf_summary"] = wf_summary
                st.session_state["wf_details"] = wf_details
                st.session_state["wf_signature"] = wf_signature

        wf_summary = st.session_state.get("wf_summary")
        wf_details = st.session_state.get("wf_details")
        saved_wf_signature = st.session_state.get("wf_signature")

        if wf_summary is not None:
            if saved_wf_signature != wf_signature:
                st.info("Walk-forward settings changed. Run validation again to refresh the results.")
            else:
                shown_wf = wf_summary.copy()
                for col in [
                    "Median return %",
                    "Average return %",
                    "Worst fold %",
                    "Best fold %",
                    "Average PF",
                    "Worst DD %",
                ]:
                    if col in shown_wf:
                        shown_wf[col] = shown_wf[col].round(2)

                st.dataframe(shown_wf, use_container_width=True, hide_index=True)
                st.caption(
                    "A stronger research result would show positive performance across multiple folds "
                    "with enough trades in each fold, rather than one unusually good window."
                )

                if wf_details is not None:
                    with st.expander("Walk-forward fold details"):
                        shown_details = wf_details.copy()
                        for col in ["Return %", "PF", "DD %"]:
                            if col in shown_details:
                                shown_details[col] = shown_details[col].round(2)
                        st.dataframe(shown_details, use_container_width=True, hide_index=True)

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

with st.expander("v2.3 logic and backtest assumptions"):
    st.write(
        "Entry signals use EMA crosses plus RSI. Optional filters require price to be on the "
        "correct side of the trend EMA and/or volume to exceed its rolling average. "
        "Stops and targets are fixed from ATR at entry. If both stop and target are touched "
        "inside the same candle, the backtest conservatively counts the stop first. "
        "Fees and slippage are applied to every completed trade. The TEST tab is kept separate "
        "from the earlier TRAIN candles to reduce the risk of judging settings only on the same sample. "
        "The v0.6 family benchmark uses fixed reference rules for EMA Cross, Trend Pullback, "
        "Donchian Breakout, and Mean Reversion so different entry hypotheses can be compared before tuning."
    )
