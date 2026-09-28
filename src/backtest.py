from __future__ import annotations

import math
import pandas as pd


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    dd = (equity / peaks - 1.0) * 100.0
    return float(dd.min())


def _side_metrics(trades: pd.DataFrame, side: str) -> dict:
    if trades.empty:
        return {"trades": 0, "win_rate_pct": 0.0, "pnl": 0.0, "profit_factor": 0.0}
    side_df = trades[trades["side"] == side]
    if side_df.empty:
        return {"trades": 0, "win_rate_pct": 0.0, "pnl": 0.0, "profit_factor": 0.0}
    wins = side_df[side_df["pnl"] > 0]
    losses = side_df[side_df["pnl"] < 0]
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
    return {
        "trades": int(len(side_df)),
        "win_rate_pct": float((side_df["pnl"] > 0).mean() * 100.0),
        "pnl": float(side_df["pnl"].sum()),
        "profit_factor": float(pf),
    }


def run_backtest(
    df: pd.DataFrame,
    start_cash: float = 10000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 1.0,
    use_stop_loss: bool = True,
    stop_atr: float = 1.0,
    use_take_profit: bool = True,
    take_atr: float = 1.5,
) -> dict:
    fee = float(fee_bps) / 10000.0
    slip = float(slippage_bps) / 10000.0

    cash = float(start_cash)
    position = 0
    entry_price = 0.0
    entry_time = None
    entry_atr = 0.0
    entry_bar = -1

    equity_rows = []
    trade_rows = []
    rows = list(df.iterrows())
    last_i = len(rows) - 1

    for i, (ts, row) in enumerate(rows):
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        signal = int(row["signal"])

        exit_reason = None
        raw_exit = None

        if position != 0:
            stop_price = None
            take_price = None

            if use_stop_loss and entry_atr > 0:
                stop_price = entry_price - position * float(stop_atr) * entry_atr
            if use_take_profit and entry_atr > 0:
                take_price = entry_price + position * float(take_atr) * entry_atr

            if position == 1:
                stop_hit = stop_price is not None and low <= stop_price
                take_hit = take_price is not None and high >= take_price
            else:
                stop_hit = stop_price is not None and high >= stop_price
                take_hit = take_price is not None and low <= take_price

            # Conservative rule: if both levels are touched in one candle, count the stop first.
            if stop_hit:
                raw_exit = float(stop_price)
                exit_reason = "STOP"
            elif take_hit:
                raw_exit = float(take_price)
                exit_reason = "TAKE"
            elif signal == -position:
                raw_exit = price
                exit_reason = "OPPOSITE"
            elif i == last_i:
                raw_exit = price
                exit_reason = "END"

        if position != 0 and raw_exit is not None:
            exit_price = raw_exit * (1 - slip if position == 1 else 1 + slip)
            gross_return = position * (exit_price / entry_price - 1.0)
            net_return = gross_return - 2 * fee
            pnl = cash * net_return
            cash += pnl

            trade_rows.append(
                {
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "return_pct": net_return * 100.0,
                    "exit_reason": exit_reason,
                    "duration_bars": max(1, i - entry_bar),
                    "atr_at_entry": entry_atr,
                }
            )

            position = 0
            entry_price = 0.0
            entry_time = None
            entry_atr = 0.0
            entry_bar = -1

        if position == 0 and signal != 0 and i < last_i:
            atr_now = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
            if atr_now > 0:
                position = signal
                entry_price = price * (1 + slip if signal == 1 else 1 - slip)
                entry_time = ts
                entry_atr = atr_now
                entry_bar = i

        marked_equity = cash
        if position != 0 and entry_price > 0:
            marked_equity = cash * (1 + position * (price / entry_price - 1.0) - fee)
        equity_rows.append({"timestamp": ts, "equity": marked_equity})

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)

    if equity.empty:
        net_return_pct = 0.0
        max_dd = 0.0
    else:
        final_equity = float(equity["equity"].iloc[-1])
        net_return_pct = (final_equity / start_cash - 1.0) * 100.0
        max_dd = _max_drawdown_pct(equity["equity"])

    if trades.empty:
        wins = 0
        losses = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pct = 0.0
    else:
        wins_df = trades[trades["pnl"] > 0]
        losses_df = trades[trades["pnl"] < 0]
        wins = len(wins_df)
        losses = len(losses_df)
        win_rate = wins / len(trades) * 100.0
        gross_profit = float(wins_df["pnl"].sum()) if wins else 0.0
        gross_loss = abs(float(losses_df["pnl"].sum())) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
        avg_trade_pct = float(trades["return_pct"].mean())

    metrics = {
        "net_return_pct": float(net_return_pct),
        "trades": int(len(trades)),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate_pct": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": float(max_dd),
        "avg_trade_pct": float(avg_trade_pct),
    }

    return {
        "metrics": metrics,
        "long": _side_metrics(trades, "LONG"),
        "short": _side_metrics(trades, "SHORT"),
        "trades": trades,
        "equity": equity,
    }
