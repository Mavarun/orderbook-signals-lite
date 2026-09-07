"""Feature shape and no look-ahead checks."""

from __future__ import annotations

import numpy as np

from orderbook_lite.data import make_synthetic_l1
from orderbook_lite.features import FEATURE_COLS, build_feature_matrix, book_imbalance, microprice


def test_feature_shapes_and_alignment():
    book = make_synthetic_l1(n_bars=500, seed=0)
    X, y, fwd = build_feature_matrix(book, horizon=1)
    assert len(X) == len(y) == len(fwd)
    assert list(X.columns) == FEATURE_COLS
    assert X.shape[1] == len(FEATURE_COLS)
    assert set(y.unique()).issubset({0, 1})
    # first row dropped due to lag-1 features; last dropped for forward label
    assert len(X) == len(book) - 2


def test_no_lookahead_in_features():
    """Features at t use only book state at t (and lags); label is future mid ret."""
    book = make_synthetic_l1(n_bars=120, seed=1)
    X, y, fwd = build_feature_matrix(book, horizon=1)
    mid = 0.5 * (book["bid"] + book["ask"])
    imb_series = book_imbalance(book["bid_size"], book["ask_size"])
    for t in X.index:
        loc = book.index.get_loc(t)
        prev = book.index[loc - 1]
        nxt = book.index[loc + 1]
        # contemporaneous mid/imbalance match book at t
        assert np.isclose(X.loc[t, "mid"], mid.loc[t])
        assert np.isclose(X.loc[t, "imbalance"], imb_series.loc[t])
        # lag1 is strictly past
        assert np.isclose(X.loc[t, "imbalance_lag1"], imb_series.loc[prev])
        # forward return is mid[t+1]/mid[t] - 1 \u2014 label only
        assert np.isclose(fwd.loc[t], mid.loc[nxt] / mid.loc[t] - 1.0)
        # y uses only the sign of that forward return
        assert int(y.loc[t]) == int(fwd.loc[t] > 0)
        # no feature column equals the forward return (leakage check)
        for col in X.columns:
            assert not np.isclose(X.loc[t, col], fwd.loc[t], atol=0.0, rtol=0.0) or col == "spread"
        # microprice_delta uses mp[t] - mp[t-1], never mp[t+1]
        mp = microprice(book["bid"], book["ask"], book["bid_size"], book["ask_size"])
        assert np.isclose(X.loc[t, "microprice_delta"], mp.loc[t] - mp.loc[prev])


def test_microprice_between_bid_ask():
    book = make_synthetic_l1(n_bars=80, seed=2)
    mp = microprice(book["bid"], book["ask"], book["bid_size"], book["ask_size"])
    assert (mp >= book["bid"] - 1e-9).all()
    assert (mp <= book["ask"] + 1e-9).all()
