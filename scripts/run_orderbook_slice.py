#!/usr/bin/env python3
"""Run mid/imbalance/microprice research slice; dump IS/OOS costed metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orderbook_lite.backtest import (
    TimeSplitConfig,
    WalkForwardConfig,
    result_to_dict,
    time_split_eval,
    walk_forward,
)
from orderbook_lite.data import load_l1
from orderbook_lite.features import build_feature_matrix


def _fmt(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x:.4f}"


def print_table(rows: list[dict]) -> None:
    headers = [
        "model",
        "split",
        "eval",
        "costed",
        "dir_acc",
        "hit_rate",
        "sharpe",
        "total_pnl",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(r["model"]),
                    str(r["split"]),
                    str(r["eval"]),
                    str(r["costed"]),
                    _fmt(r["dir_acc"]),
                    _fmt(r["hit_rate"]),
                    _fmt(r["sharpe"]),
                    _fmt(r["total_pnl"]),
                ]
            )
            + " |"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["synthetic", "public"], default="synthetic")
    p.add_argument("--path", default=None, help="CSV path when --source public")
    p.add_argument("--n-bars", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--train-size", type=int, default=1500)
    p.add_argument("--test-size", type=int, default=400)
    p.add_argument("--step", type=int, default=400)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--mode", choices=["long_flat", "long_short"], default="long_flat")
    p.add_argument("--baseline", default="logistic", choices=["logistic", "ridge"])
    p.add_argument(
        "--eval",
        choices=["time_split", "walk_forward", "both"],
        default="both",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    df, note = load_l1(
        source=args.source,
        n_bars=args.n_bars,
        seed=args.seed,
        path=args.path,
    )
    print(f"Data: {note}")
    X, y, fwd = build_feature_matrix(df, horizon=args.horizon)
    print(
        f"Features: n={len(X)} cols={list(X.columns)} "
        f"pos_rate={y.mean():.3f}"
    )

    Xa = X.values.astype(np.float64)
    ya = y.values.astype(np.int64)
    fa = fwd.values.astype(np.float64)

    all_results = {}
    table_rows = []

    def add_rows(name: str, split_label: str, d: dict) -> None:
        for eval_name, key, costed in [
            ("IS", "is_costed", True),
            ("IS", "is_uncosted", False),
            ("OOS", "oos_costed", True),
            ("OOS", "oos_uncosted", False),
        ]:
            m = d[key]
            table_rows.append(
                {
                    "model": name,
                    "split": split_label,
                    "eval": eval_name,
                    "costed": "yes" if costed else "no",
                    "dir_acc": m.get("dir_accuracy", float("nan")),
                    "hit_rate": m.get("hit_rate", float("nan")),
                    "sharpe": m.get("sharpe", float("nan")),
                    "total_pnl": m.get("total_pnl", float("nan")),
                }
            )

    if args.eval in ("time_split", "both"):
        cfg = TimeSplitConfig(
            train_frac=args.train_frac,
            mode=args.mode,
            cost_bps=args.cost_bps,
            baseline_kind=args.baseline,
        )
        print(f"\n=== time_split: {args.baseline} ===")
        res = time_split_eval(Xa, ya, fa, cfg=cfg)
        d = result_to_dict(res)
        all_results["time_split"] = d
        print(
            f"OOS dir_acc={d['oos_costed'].get('dir_accuracy', float('nan')):.4f} "
            f"IS dir_acc={d['is_costed'].get('dir_accuracy', float('nan')):.4f} "
            f"OOS sharpe costed={d['oos_costed']['sharpe']:.4f} "
            f"uncosted={d['oos_uncosted']['sharpe']:.4f}"
        )
        add_rows(args.baseline, "time_split", d)

    if args.eval in ("walk_forward", "both"):
        cfg_wf = WalkForwardConfig(
            train_size=args.train_size,
            test_size=args.test_size,
            step=args.step,
            mode=args.mode,
            cost_bps=args.cost_bps,
            baseline_kind=args.baseline,
        )
        print(f"\n=== walk_forward: {args.baseline} ===")
        res_wf = walk_forward(Xa, ya, fa, cfg=cfg_wf)
        d_wf = result_to_dict(res_wf)
        all_results["walk_forward"] = d_wf
        print(
            f"folds={d_wf['n_folds']} "
            f"OOS dir_acc={d_wf['oos_costed'].get('dir_accuracy', float('nan')):.4f} "
            f"OOS sharpe costed={d_wf['oos_costed']['sharpe']:.4f} "
            f"uncosted={d_wf['oos_uncosted']['sharpe']:.4f}"
        )
        add_rows(args.baseline, "walk_forward", d_wf)

    print("\n## Metrics table")
    print_table(table_rows)
    print(
        "\nDisclaimer: research slice only \u2014 not live PnL, not a claim of alpha. "
        "Synthetic L1 is not an exchange tape. OOS after costs is the score that matters."
    )

    payload = {
        "data_note": note,
        "config": {
            "source": args.source,
            "n_bars": args.n_bars,
            "seed": args.seed,
            "horizon": args.horizon,
            "train_frac": args.train_frac,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "step": args.step,
            "cost_bps": args.cost_bps,
            "mode": args.mode,
            "baseline": args.baseline,
            "eval": args.eval,
        },
        "results": all_results,
        "table": table_rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=float))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
