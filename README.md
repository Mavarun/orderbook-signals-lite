# orderbook-signals-lite

Research slice: **mid price, bid-ask imbalance, and microprice** features from
synthetic L1 (top-of-book) samples, scored with a **leakage-safe time split /
walk-forward** and **spread costs**. Not live PnL. Not a claim of alpha.

## Hypothesis

1. Mid price, bid-ask imbalance, and simple microprice features from public/synthetic L1 order-book samples have short-horizon predictive power for next mid return.
2. A leakage-safe time split (train features only on past bars) will show weaker OOS accuracy than in-sample \u2014 report both.
3. After spread costs, any naive signal PnL shrinks or vanishes.

## Method

- **Data (default):** synthetic L1 top-of-book generated in-process (`seed=42`,
  `n_bars=4000`, 1-minute bars). Latent imbalance is AR(1) and mildly drives
  next-bar mid drift. **No free public L1 tape is bundled**; use `--source public
  --path ...` for a cited CSV with columns `bid,ask,bid_size,ask_size`.
- **Features (at t only):** `mid`, `imbalance=(bid_sz-ask_sz)/(bid_sz+ask_sz)`,
  `microprice`, `spread`, `microprice_delta`, `imbalance_lag1`, `spread_lag1`.
  Label = sign of next mid return. Forward return is never a feature.
- **Models:** `StandardScaler` + logistic (default) or ridge classifier
  (scikit-learn). No PyTorch.
- **Backtest:** chronological `train_frac=0.70` time split and rolling
  walk-forward (`train_size=1500`, `test_size=400`, `step=400`). Positions:
  long/flat (default) or long/short. Costs: `cost_bps` on `|\u0394position|` each bar.
- **Metrics:** directional accuracy, hit rate, total PnL, annualized Sharpe
  (1-min scale `252*390`) \u2014 **IS vs OOS**, **costed vs uncosted**.

## Metrics (local synthetic run)

Command: `python scripts/run_orderbook_slice.py --source synthetic`  
Defaults above; model `logistic`.

| model | split | eval | costed | dir_acc | hit_rate | sharpe | total_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic | time_split | IS | yes | 0.7091 | 0.7016 | -93.6352 | -0.2640 |
| logistic | time_split | IS | no | 0.7091 | 0.7016 | 104.3370 | 0.1885 |
| logistic | time_split | OOS | yes | 0.6858 | 0.6818 | -106.5607 | -0.1293 |
| logistic | time_split | OOS | no | 0.6858 | 0.6818 | 98.2862 | 0.0717 |
| logistic | walk_forward | IS | yes | 0.7040 | 0.6999 | -87.3238 | -0.3064 |
| logistic | walk_forward | IS | no | 0.7040 | 0.6999 | 104.2018 | 0.2341 |
| logistic | walk_forward | OOS | yes | 0.6992 | 0.6990 | -98.8934 | -0.2368 |
| logistic | walk_forward | OOS | no | 0.6992 | 0.6990 | 100.8191 | 0.1517 |

**Reading:** On this planted synthetic L1, imbalance/microprice features show
clear short-horizon directional edge (OOS dir_acc \u2248 0.69 vs ~0.5 chance).
**IS > OOS** accuracy (0.709 vs 0.686 on the time split) matches hypothesis (2).
**Uncosted** OOS Sharpe/PnL is strongly positive, but **5 bps turnover costs
flip Sharpe deeply negative** and wipe cumulative PnL \u2014 hypothesis (3). This is
a research illustration on synthetic tape, not tradable alpha.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt
pytest
python scripts/run_orderbook_slice.py
python scripts/run_orderbook_slice.py --json --out artifacts/metrics.json
# optional user-supplied public L1 CSV:
# python scripts/run_orderbook_slice.py --source public --path /path/to/l1.csv
```

## Package layout

```
src/orderbook_lite/   data, features, models, metrics, backtest
scripts/              run_orderbook_slice.py
tests/                feature shapes, no look-ahead, cost drag, barriers
```

## Assumptions / limits

- Synthetic L1 is **not** exchange market data; signal strength is planted for
  a reproducible research demo.
- Flat bps cost ignores queue position, adverse selection, and latency.
- Long/flat on 1-minute bars with frequent flips pays turnover heavily.
- Single-name L1; no multi-level book, no cancellations, no venue fees.
- No hyperparameter search; logistic/ridge only.
- **Never** claim live PnL or production alpha from this slice.
