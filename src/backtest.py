from __future__ import annotations

import math
import pandas as pd


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    dd = (equity / peaks - 1.0) * 100.0
    return float(dd.min())


def run_backtest(
    df: pd.DataFrame,
    start_cash: float = 10000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 1.0,
) -> dict:
    fee = fee_bps / 10000.0
    slip = slippage_bps / 10000.0

    cash = float(start_cash)
    position = 0
    entry_price = 0.0
    entry_time = None
    equity_rows = []
    trade_rows = []

    for ts, row in df.iterrows():
        price = float(row["close"])
        signal = int(row["signal"])

        if position != 0 and signal == -position:
            exit_price = price * (1 - slip if position == 1 else 1 + slip)
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
                }
            )
            position = 0
            entry_price = 0.0
            entry_time = None

        if position == 0 and signal != 0:
            position = signal
            entry_price = price * (1 + slip if signal == 1 else 1 - slip)
            entry_time = ts

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
    else:
        wins_df = trades[trades["pnl"] > 0]
        losses_df = trades[trades["pnl"] < 0]
        wins = len(wins_df)
        losses = len(losses_df)
        win_rate = wins / len(trades) * 100.0
        gross_profit = float(wins_df["pnl"].sum()) if wins else 0.0
        gross_loss = abs(float(losses_df["pnl"].sum())) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)

    metrics = {
        "net_return_pct": float(net_return_pct),
        "trades": int(len(trades)),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate_pct": float(win_rate),
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": float(max_dd),
    }
    return {"metrics": metrics, "trades": trades, "equity": equity}
