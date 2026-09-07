"""Synthetic L1 data smoke tests."""

from __future__ import annotations

from orderbook_lite.data import load_l1, make_synthetic_l1


def test_synthetic_l1_columns_and_length():
    df = make_synthetic_l1(n_bars=100, seed=42)
    assert len(df) == 100
    for col in ("bid", "ask", "bid_size", "ask_size", "mid", "spread"):
        assert col in df.columns
    assert (df["ask"] >= df["bid"]).all()
    assert (df["spread"] > 0).all()


def test_load_l1_synthetic_note():
    df, note = load_l1(source="synthetic", n_bars=50, seed=1)
    assert len(df) == 50
    assert "synthetic" in note.lower()
