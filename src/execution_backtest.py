from __future__ import annotations

import math
import pandas as pd


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    dd = (equity / peaks - 1.0) * 100.0
    return float(dd.min())


def run_strict_backtest(
    df: pd.DataFrame,
    *,
    start_cash: float = 10000.0,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
    stop_atr: float = 1.0,
    take_atr: float = 1.5,
    side: str = "BOTH",
    cooldown_bars: int = 0,
) -> dict:
    """Backtest signals with next-bar execution.

    A signal observed at bar t close can only be acted on at bar t+1 open.
    Intrabar stop/take checks then use the t+1 high/low. If both stop and target
    are touched in one bar, the stop is counted first (conservative).
    """
    if df.empty:
        return {
            "metrics": {
                "net_return_pct": 0.0,
                "trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "avg_trade_pct": 0.0,
            },
            "trades": pd.DataFrame(),
            "equity": pd.DataFrame(),
        }

    fee = float(fee_bps) / 10000.0
    slip = float(slippage_bps) / 10000.0
    side = str(side).upper()

    cash = float(start_cash)
    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_time = None
    entry_bar = -1
    last_exit_bar = -10_000

    equity_rows = []
    trade_rows = []
    rows = list(df.iterrows())

    for i, (ts, row) in enumerate(rows):
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        prev_signal = int(rows[i - 1][1]["signal"]) if i > 0 else 0
        prev_atr = float(rows[i - 1][1]["atr"]) if i > 0 and pd.notna(rows[i - 1][1]["atr"]) else 0.0

        if side == "LONG":
            prev_signal = 1 if prev_signal > 0 else 0
        elif side == "SHORT":
            prev_signal = -1 if prev_signal < 0 else 0

        # Signal-based exit executes at the next bar open.
        if position != 0 and prev_signal == -position:
            exit_price = open_price * (1 - slip if position == 1 else 1 + slip)
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
                    "return_pct": net_return * 100.0,
                    "pnl": pnl,
                    "exit_reason": "OPPOSITE_NEXT_OPEN",
                    "duration_bars": max(1, i - entry_bar),
                }
            )
            position = 0
            entry_price = 0.0
            entry_atr = 0.0
            entry_time = None
            last_exit_bar = i

        # New entries also execute at next bar open.
        if (
            position == 0
            and prev_signal != 0
            and prev_atr > 0
            and (i - last_exit_bar) > int(cooldown_bars)
        ):
            position = prev_signal
            entry_price = open_price * (1 + slip if position == 1 else 1 - slip)
            entry_atr = prev_atr
            entry_time = ts
            entry_bar = i

        # Intrabar risk management after the open execution.
        if position != 0:
            stop_price = entry_price - position * float(stop_atr) * entry_atr
            take_price = entry_price + position * float(take_atr) * entry_atr

            if position == 1:
                stop_hit = low <= stop_price
                take_hit = high >= take_price
            else:
                stop_hit = high >= stop_price
                take_hit = low <= take_price

            raw_exit = None
            reason = None
            if stop_hit:
                raw_exit = stop_price
                reason = "STOP"
            elif take_hit:
                raw_exit = take_price
                reason = "TAKE"

            if raw_exit is not None:
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
                        "return_pct": net_return * 100.0,
                        "pnl": pnl,
                        "exit_reason": reason,
                        "duration_bars": max(1, i - entry_bar + 1),
                    }
                )
                position = 0
                entry_price = 0.0
                entry_atr = 0.0
                entry_time = None
                last_exit_bar = i

        # Close any remaining position on the final close.
        if i == len(rows) - 1 and position != 0:
            exit_price = close * (1 - slip if position == 1 else 1 + slip)
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
                    "return_pct": net_return * 100.0,
                    "pnl": pnl,
                    "exit_reason": "END",
                    "duration_bars": max(1, i - entry_bar + 1),
                }
            )
            position = 0

        marked_equity = cash
        if position != 0 and entry_price > 0:
            marked_equity = cash * (
                1.0 + position * (close / entry_price - 1.0) - fee
            )
        equity_rows.append({"timestamp": ts, "equity": marked_equity})

    trades = pd.DataFrame(trade_rows)
    equity = pd.DataFrame(equity_rows)

    if trades.empty:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0
    else:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]
        win_rate = float((trades["pnl"] > 0).mean() * 100.0)
        gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
        gross_loss = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (math.inf if gross_profit > 0 else 0.0)
        )
        avg_trade = float(trades["return_pct"].mean())

    final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else start_cash
    metrics = {
        "net_return_pct": (final_equity / start_cash - 1.0) * 100.0,
        "trades": int(len(trades)),
        "win_rate_pct": win_rate,
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": _max_drawdown_pct(equity["equity"]) if not equity.empty else 0.0,
        "avg_trade_pct": avg_trade,
    }
    return {"metrics": metrics, "trades": trades, "equity": equity}
