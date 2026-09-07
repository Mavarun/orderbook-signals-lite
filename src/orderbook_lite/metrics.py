"""Directional accuracy, costed PnL, and Sharpe helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def hit_rate(positions: np.ndarray, fwd_returns: np.ndarray) -> float:
    """Fraction of non-flat bars where position sign matches return sign."""
    pos = np.asarray(positions, dtype=float).ravel()
    r = np.asarray(fwd_returns, dtype=float).ravel()
    mask = pos != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(pos[mask]) == np.sign(r[mask])))


def strategy_pnl(
    positions: np.ndarray,
    fwd_returns: np.ndarray,
    cost_bps: float = 0.0,
) -> np.ndarray:
    """
    Per-bar strategy returns after transaction costs.

    Cost is charged on absolute position change (spread+slippage proxy)
    in decimal: cost_bps / 1e4 * |delta position|.
    """
    pos = np.asarray(positions, dtype=float).ravel()
    r = np.asarray(fwd_returns, dtype=float).ravel()
    prev = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - prev)
    cost = (cost_bps / 1e4) * turnover
    return pos * r - cost


def sharpe_ratio(
    bar_returns: np.ndarray,
    periods_per_year: float = 252.0 * 390.0,
) -> float:
    """Annualize assuming ~1-minute equity bars by default (252*390)."""
    x = np.asarray(bar_returns, dtype=float).ravel()
    if len(x) < 2:
        return float("nan")
    mu = np.mean(x)
    sd = np.std(x, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * mu / sd)


def summarize_pnl(
    positions: np.ndarray,
    fwd_returns: np.ndarray,
    cost_bps: float = 0.0,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    periods_per_year: float = 252.0 * 390.0,
) -> dict[str, float]:
    pnl = strategy_pnl(positions, fwd_returns, cost_bps=cost_bps)
    out: dict[str, float] = {
        "n": float(len(pnl)),
        "total_pnl": float(np.sum(pnl)),
        "mean_pnl": float(np.mean(pnl)) if len(pnl) else float("nan"),
        "sharpe": sharpe_ratio(pnl, periods_per_year=periods_per_year),
        "hit_rate": hit_rate(positions, fwd_returns),
        "cost_bps": float(cost_bps),
        "avg_abs_position": float(np.mean(np.abs(positions)))
        if len(positions)
        else float("nan"),
    }
    if y_true is not None and y_pred is not None:
        out["dir_accuracy"] = directional_accuracy(y_true, y_pred)
    return out
