"""Leakage-safe time split / walk-forward backtest with spread costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from orderbook_lite.metrics import summarize_pnl
from orderbook_lite.models import BaselineKind, build_baseline

Mode = Literal["long_flat", "long_short"]


def signals_from_pred(
    y_pred: np.ndarray,
    mode: Mode = "long_flat",
) -> np.ndarray:
    """Map class predictions {0,1} to positions."""
    y = np.asarray(y_pred).ravel().astype(int)
    if mode == "long_flat":
        return y.astype(float)
    if mode == "long_short":
        return np.where(y == 1, 1.0, -1.0)
    raise ValueError(f"unknown mode: {mode}")


def _barrier_check(train_end: int, test_start: int) -> None:
    """Enforce strict train/test barrier (no overlap)."""
    if test_start < train_end:
        raise ValueError(
            f"look-ahead barrier violated: test_start={test_start} < train_end={train_end}"
        )


@dataclass
class TimeSplitConfig:
    train_frac: float = 0.70
    mode: Mode = "long_flat"
    cost_bps: float = 5.0
    baseline_kind: BaselineKind = "logistic"
    periods_per_year: float = 252.0 * 390.0


@dataclass
class WalkForwardConfig:
    train_size: int = 1500
    test_size: int = 400
    step: int = 400
    mode: Mode = "long_flat"
    cost_bps: float = 5.0
    baseline_kind: BaselineKind = "logistic"
    periods_per_year: float = 252.0 * 390.0


@dataclass
class FoldResult:
    fold: int
    train_end: int
    test_start: int
    test_end: int
    metrics_costed: dict[str, float]
    metrics_uncosted: dict[str, float]
    is_metrics_costed: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalResult:
    model: str
    split: str
    folds: list[FoldResult]
    oos_costed: dict[str, float]
    oos_uncosted: dict[str, float]
    is_costed: dict[str, float]
    is_uncosted: dict[str, float]


def time_split_eval(
    X: np.ndarray,
    y: np.ndarray,
    fwd: np.ndarray,
    cfg: Optional[TimeSplitConfig] = None,
) -> EvalResult:
    """
    Single leakage-safe chronological split: train on past bars only.

    Train = [0, train_end), test = [train_end, n). Features/labels already
    exclude look-ahead inside the feature builder; this split ensures the
    classifier never sees future rows during fit.
    """
    cfg = cfg or TimeSplitConfig()
    n = len(X)
    if n < 50:
        raise ValueError(f"need at least 50 rows, got {n}")
    train_end = int(n * cfg.train_frac)
    if train_end < 20 or train_end >= n:
        raise ValueError(f"invalid train_end={train_end} for n={n}")
    _barrier_check(train_end, train_end)

    X_tr, y_tr, fwd_tr = X[:train_end], y[:train_end], fwd[:train_end]
    X_te, y_te, fwd_te = X[train_end:], y[train_end:], fwd[train_end:]

    pipe = build_baseline(kind=cfg.baseline_kind)
    pipe.fit(X_tr, y_tr)
    pred_tr = pipe.predict(X_tr)
    pred_te = pipe.predict(X_te)
    pos_tr = signals_from_pred(pred_tr, mode=cfg.mode)
    pos_te = signals_from_pred(pred_te, mode=cfg.mode)

    fold = FoldResult(
        fold=0,
        train_end=train_end,
        test_start=train_end,
        test_end=n,
        metrics_costed=summarize_pnl(
            pos_te,
            fwd_te,
            cost_bps=cfg.cost_bps,
            y_true=y_te,
            y_pred=pred_te,
            periods_per_year=cfg.periods_per_year,
        ),
        metrics_uncosted=summarize_pnl(
            pos_te,
            fwd_te,
            cost_bps=0.0,
            y_true=y_te,
            y_pred=pred_te,
            periods_per_year=cfg.periods_per_year,
        ),
        is_metrics_costed=summarize_pnl(
            pos_tr,
            fwd_tr,
            cost_bps=cfg.cost_bps,
            y_true=y_tr,
            y_pred=pred_tr,
            periods_per_year=cfg.periods_per_year,
        ),
    )
    return EvalResult(
        model=cfg.baseline_kind,
        split="time_split",
        folds=[fold],
        oos_costed=fold.metrics_costed,
        oos_uncosted=fold.metrics_uncosted,
        is_costed=fold.is_metrics_costed,
        is_uncosted=summarize_pnl(
            pos_tr,
            fwd_tr,
            cost_bps=0.0,
            y_true=y_tr,
            y_pred=pred_tr,
            periods_per_year=cfg.periods_per_year,
        ),
    )


def walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    fwd: np.ndarray,
    cfg: Optional[WalkForwardConfig] = None,
) -> EvalResult:
    """
    Rolling walk-forward: train on [0, train_end), test on
    [train_end, train_end + test_size) with contiguous barrier (no leak).
    """
    cfg = cfg or WalkForwardConfig()
    n = len(X)
    if n < cfg.train_size + cfg.test_size:
        raise ValueError(
            f"not enough rows ({n}) for train={cfg.train_size} + test={cfg.test_size}"
        )

    folds: list[FoldResult] = []
    oos_pos: list[np.ndarray] = []
    oos_fwd: list[np.ndarray] = []
    oos_y: list[np.ndarray] = []
    oos_pred: list[np.ndarray] = []

    fold_i = 0
    train_end = cfg.train_size
    last_pos_tr = last_fwd_tr = last_y_tr = last_pred_tr = None

    while train_end + cfg.test_size <= n:
        test_start = train_end
        test_end = train_end + cfg.test_size
        _barrier_check(train_end, test_start)

        X_tr, y_tr, fwd_tr = X[:train_end], y[:train_end], fwd[:train_end]
        X_te, y_te, fwd_te = X[test_start:test_end], y[test_start:test_end], fwd[test_start:test_end]

        pipe = build_baseline(kind=cfg.baseline_kind)
        pipe.fit(X_tr, y_tr)
        pred_te = pipe.predict(X_te)
        pred_tr = pipe.predict(X_tr)
        pos_te = signals_from_pred(pred_te, mode=cfg.mode)
        pos_tr = signals_from_pred(pred_tr, mode=cfg.mode)

        m_c = summarize_pnl(
            pos_te,
            fwd_te,
            cost_bps=cfg.cost_bps,
            y_true=y_te,
            y_pred=pred_te,
            periods_per_year=cfg.periods_per_year,
        )
        m_u = summarize_pnl(
            pos_te,
            fwd_te,
            cost_bps=0.0,
            y_true=y_te,
            y_pred=pred_te,
            periods_per_year=cfg.periods_per_year,
        )
        m_is = summarize_pnl(
            pos_tr,
            fwd_tr,
            cost_bps=cfg.cost_bps,
            y_true=y_tr,
            y_pred=pred_tr,
            periods_per_year=cfg.periods_per_year,
        )

        folds.append(
            FoldResult(
                fold=fold_i,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                metrics_costed=m_c,
                metrics_uncosted=m_u,
                is_metrics_costed=m_is,
            )
        )
        oos_pos.append(pos_te)
        oos_fwd.append(fwd_te)
        oos_y.append(y_te)
        oos_pred.append(pred_te)
        last_pos_tr, last_fwd_tr, last_y_tr, last_pred_tr = pos_tr, fwd_tr, y_tr, pred_tr

        fold_i += 1
        train_end += cfg.step

    if not folds:
        raise RuntimeError("no walk-forward folds produced")

    pos_o = np.concatenate(oos_pos)
    fwd_o = np.concatenate(oos_fwd)
    y_o = np.concatenate(oos_y)
    pred_o = np.concatenate(oos_pred)
    assert last_pos_tr is not None

    return EvalResult(
        model=cfg.baseline_kind,
        split="walk_forward",
        folds=folds,
        oos_costed=summarize_pnl(
            pos_o,
            fwd_o,
            cost_bps=cfg.cost_bps,
            y_true=y_o,
            y_pred=pred_o,
            periods_per_year=cfg.periods_per_year,
        ),
        oos_uncosted=summarize_pnl(
            pos_o,
            fwd_o,
            cost_bps=0.0,
            y_true=y_o,
            y_pred=pred_o,
            periods_per_year=cfg.periods_per_year,
        ),
        is_costed=summarize_pnl(
            last_pos_tr,
            last_fwd_tr,
            cost_bps=cfg.cost_bps,
            y_true=last_y_tr,
            y_pred=last_pred_tr,
            periods_per_year=cfg.periods_per_year,
        ),
        is_uncosted=summarize_pnl(
            last_pos_tr,
            last_fwd_tr,
            cost_bps=0.0,
            y_true=last_y_tr,
            y_pred=last_pred_tr,
            periods_per_year=cfg.periods_per_year,
        ),
    )


def result_to_dict(res: EvalResult) -> dict[str, Any]:
    return {
        "model": res.model,
        "split": res.split,
        "n_folds": len(res.folds),
        "oos_costed": res.oos_costed,
        "oos_uncosted": res.oos_uncosted,
        "is_costed": res.is_costed,
        "is_uncosted": res.is_uncosted,
        "fold_oos_dir_acc": [
            f.metrics_costed.get("dir_accuracy", float("nan")) for f in res.folds
        ],
        "fold_oos_sharpe_costed": [
            f.metrics_costed.get("sharpe", float("nan")) for f in res.folds
        ],
    }
