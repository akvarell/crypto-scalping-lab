from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest import run_backtest
from src.data import fetch_ohlcv
from src.family_benchmark import benchmark_families
from src.indicators import add_indicators
from src.mean_reversion_lab import evaluate_mean_reversion_variants
from src.optimizer import optimize_quick
from src.strategy import generate_signals
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


st.title("Crypto Scalping Lab v0.7")
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

with st.expander("v0.7 logic and backtest assumptions"):
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
