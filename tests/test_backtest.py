"""Cost drag, time-split barrier, and walk-forward integrity."""

from __future__ import annotations

import numpy as np
import pytest

from orderbook_lite.backtest import (
    TimeSplitConfig,
    WalkForwardConfig,
    _barrier_check,
    signals_from_pred,
    time_split_eval,
    walk_forward,
)
from orderbook_lite.data import make_synthetic_l1
from orderbook_lite.features import build_feature_matrix
from orderbook_lite.metrics import strategy_pnl, summarize_pnl


def test_costs_reduce_pnl_vs_zero_cost():
    rng = np.random.default_rng(7)
    n = 200
    pos = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)])
    fwd = rng.normal(0.0005, 0.01, size=n)
    pnl0 = strategy_pnl(pos, fwd, cost_bps=0.0)
    pnl_c = strategy_pnl(pos, fwd, cost_bps=10.0)
    assert np.sum(pnl_c) < np.sum(pnl0)
    s0 = summarize_pnl(pos, fwd, cost_bps=0.0)
    sc = summarize_pnl(pos, fwd, cost_bps=10.0)
    assert sc["total_pnl"] < s0["total_pnl"]


def test_barrier_rejects_overlap():
    _barrier_check(100, 100)
    with pytest.raises(ValueError, match="barrier"):
        _barrier_check(100, 99)


def test_time_split_train_before_test():
    book = make_synthetic_l1(n_bars=800, seed=3)
    X, y, fwd = build_feature_matrix(book)
    cfg = TimeSplitConfig(train_frac=0.7, cost_bps=5.0, baseline_kind="logistic")
    res = time_split_eval(
        X.values.astype(float),
        y.values.astype(int),
        fwd.values.astype(float),
        cfg=cfg,
    )
    assert len(res.folds) == 1
    f = res.folds[0]
    assert f.test_start == f.train_end
    assert f.test_end > f.test_start
    # OOS dir accuracy should be finite; IS often >= OOS on planted signal
    assert 0.0 <= res.oos_costed["dir_accuracy"] <= 1.0
    assert 0.0 <= res.is_costed["dir_accuracy"] <= 1.0
    assert res.oos_costed["total_pnl"] <= res.oos_uncosted["total_pnl"] + 1e-12


def test_walk_forward_train_test_barrier():
    book = make_synthetic_l1(n_bars=2500, seed=4)
    X, y, fwd = build_feature_matrix(book)
    cfg = WalkForwardConfig(
        train_size=800,
        test_size=200,
        step=200,
        cost_bps=5.0,
        baseline_kind="ridge",
    )
    res = walk_forward(
        X.values.astype(float),
        y.values.astype(int),
        fwd.values.astype(float),
        cfg=cfg,
    )
    assert len(res.folds) >= 2
    for f in res.folds:
        assert f.test_start == f.train_end
        assert f.test_end > f.test_start


def test_signals_modes():
    pred = np.array([1, 0, 1, 0])
    assert list(signals_from_pred(pred, "long_flat")) == [1.0, 0.0, 1.0, 0.0]
    assert list(signals_from_pred(pred, "long_short")) == [1.0, -1.0, 1.0, -1.0]
