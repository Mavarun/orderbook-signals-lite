"""L1 features: mid, imbalance, microprice, spread \u2014 leakage-safe labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "mid",
    "imbalance",
    "microprice",
    "spread",
    "microprice_delta",
    "imbalance_lag1",
    "spread_lag1",
]


def book_imbalance(bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
    """(bid_size - ask_size) / (bid_size + ask_size), clipped to [-1, 1]."""
    denom = bid_size.astype(float) + ask_size.astype(float)
    raw = (bid_size.astype(float) - ask_size.astype(float)) / denom.replace(0.0, np.nan)
    return raw.clip(-1.0, 1.0)


def microprice(
    bid: pd.Series,
    ask: pd.Series,
    bid_size: pd.Series,
    ask_size: pd.Series,
) -> pd.Series:
    """Size-weighted mid: (ask * bid_size + bid * ask_size) / (bid_size + ask_size)."""
    denom = bid_size.astype(float) + ask_size.astype(float)
    return (
        ask.astype(float) * bid_size.astype(float)
        + bid.astype(float) * ask_size.astype(float)
    ) / denom.replace(0.0, np.nan)


def build_feature_matrix(
    book: pd.DataFrame,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build contemporaneous L1 features at t and next-mid-return labels at t+horizon.

    Features use only information available at bar t (bid/ask/sizes at t and
    lagged features ending at t). Label is sign of mid return from t to t+horizon.
    Forward mid return is also returned for PnL evaluation.

    Returns
    -------
    X : DataFrame of features
    y_dir : Series of {0, 1} (1 = next mid return > 0)
    fwd_ret : Series of next-horizon mid simple returns
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    required = {"bid", "ask", "bid_size", "ask_size"}
    missing = required - set(book.columns)
    if missing:
        raise ValueError(f"book missing columns: {sorted(missing)}")

    bid = book["bid"].astype(float)
    ask = book["ask"].astype(float)
    bs = book["bid_size"].astype(float)
    asz = book["ask_size"].astype(float)
    mid = 0.5 * (bid + ask)
    spr = ask - bid
    imb = book_imbalance(bs, asz)
    mp = microprice(bid, ask, bs, asz)

    cols: dict[str, pd.Series] = {
        "mid": mid,
        "imbalance": imb,
        "microprice": mp,
        "spread": spr,
        # deltas / lags \u2014 past only (shift >= 1)
        "microprice_delta": mp - mp.shift(1),
        "imbalance_lag1": imb.shift(1),
        "spread_lag1": spr.shift(1),
    }
    X = pd.DataFrame(cols, index=book.index)

    # Label: mid return over next `horizon` bars \u2014 never used as a feature
    fwd_ret = mid.pct_change(horizon).shift(-horizon)
    y_dir = (fwd_ret > 0).astype(int)

    valid = X.notna().all(axis=1) & fwd_ret.notna()
    X = X.loc[valid]
    y_dir = y_dir.loc[valid]
    fwd_ret = fwd_ret.loc[valid]
    return X, y_dir, fwd_ret
