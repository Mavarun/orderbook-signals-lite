"""Synthetic L1 order-book samples (default) with clear provenance note.

No free public L1 tape is bundled. When a public sample path is unavailable,
we generate a reproducible synthetic top-of-book series in-process.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def make_synthetic_l1(
    n_bars: int = 4000,
    seed: int = 42,
    mid0: float = 100.0,
    tick: float = 0.01,
    base_spread_ticks: int = 1,
    size_mean: float = 50.0,
    size_std: float = 15.0,
    signal_strength: float = 0.35,
    noise_vol: float = 0.00025,
    start: str = "2024-01-01",
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Synthetic L1 (top-of-book) bars with mild imbalance → next-mid-return link.

    Generative story
    ----------------
    - Latent imbalance innovation drives a small next-bar mid drift
      (`signal_strength * imbalance * noise_vol` scale).
    - Bid/ask sizes are log-normal around `size_mean`; spread is a few ticks.
    - Mid follows a random walk plus the imbalance-linked drift.

    Citation: generated in-process for reproducible offline research;
    **not** exchange market data. Prefer a public L1 tape when available;
    until then synthetic L1 is the documented default.
    """
    if n_bars < 10:
        raise ValueError("n_bars must be >= 10")
    rng = np.random.default_rng(seed)

    # Latent imbalance in [-1, 1] with mild AR(1) persistence
    eps = rng.normal(0.0, 0.4, size=n_bars)
    imb = np.empty(n_bars, dtype=float)
    imb[0] = np.clip(eps[0], -1.0, 1.0)
    for t in range(1, n_bars):
        imb[t] = np.clip(0.55 * imb[t - 1] + eps[t], -1.0, 1.0)

    # Next-bar mid return driven partly by *current* imbalance (predictive)
    noise = rng.normal(0.0, noise_vol, size=n_bars)
    mid_ret = np.empty(n_bars, dtype=float)
    mid_ret[0] = noise[0]
    for t in range(1, n_bars):
        mid_ret[t] = signal_strength * imb[t - 1] * noise_vol * 4.0 + noise[t]

    mid = mid0 * np.cumprod(1.0 + mid_ret)

    # Spread in price units
    spread_ticks = base_spread_ticks + (rng.random(n_bars) < 0.15).astype(int)
    spread = spread_ticks * tick
    bid = mid - 0.5 * spread
    ask = mid + 0.5 * spread

    # Sizes consistent with latent imbalance: more bid size when imb > 0
    total = np.clip(rng.normal(size_mean * 2.0, size_std, size=n_bars), 10.0, None)
    bid_size = np.clip(total * (0.5 + 0.5 * imb), 1.0, None)
    ask_size = np.clip(total - bid_size, 1.0, None)

    idx = pd.date_range(start=start, periods=n_bars, freq=freq)
    return pd.DataFrame(
        {
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "mid": mid,
            "spread": spread,
            "imbalance_latent": imb,
            "mid_return": mid_ret,
        },
        index=idx,
    )


def load_public_l1_sample(path: str) -> pd.DataFrame:
    """
    Load a public/cited L1 CSV if the user supplies one.

    Expected columns: bid, ask, bid_size, ask_size (optional: mid, timestamp index).
    Raises if required columns are missing.
    """
    df = pd.read_csv(path)
    required = {"bid", "ask", "bid_size", "ask_size"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"public L1 sample missing columns: {sorted(missing)}")
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"]))
    df = df.copy()
    df["mid"] = 0.5 * (df["bid"].astype(float) + df["ask"].astype(float))
    df["spread"] = df["ask"].astype(float) - df["bid"].astype(float)
    df["mid_return"] = df["mid"].pct_change().fillna(0.0)
    return df


def load_l1(
    source: str = "synthetic",
    n_bars: int = 4000,
    seed: int = 42,
    path: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load L1 book bars; default is synthetic with an explicit provenance note.

    Returns (frame, note).
    """
    if source == "public":
        if not path:
            raise ValueError("source='public' requires path= to a CSV sample")
        df = load_public_l1_sample(path)
        note = (
            f"public/cited L1 CSV path={path} n={len(df)} "
            "(user-supplied; cite the original data source in papers)"
        )
        return df, note

    df = make_synthetic_l1(n_bars=n_bars, seed=seed)
    note = (
        f"synthetic L1 top-of-book n={len(df)} seed={seed} "
        "(in-process; not exchange tape \u2014 no free public L1 sample bundled)"
    )
    return df, note
