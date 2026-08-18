"""
core/data.py
============
Data acquisition, caching, and walk-forward CV utilities.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.stattools import adfuller, kpss

from core.models import (
    stationarity_analysis, find_best_arima_order,
    generate_walk_forward_windows, compute_metrics,
    rolling_one_step, SARIMAXWrapper,
    walk_forward_split
)
from statsmodels.tsa.statespace.sarimax import SARIMAX


def load_data(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Download and clean stock data via yfinance."""
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if len(df) < 100:
                continue
            # ffill only — bfill would pull future prices backward into gaps
            df = df.ffill()
            data[sym] = df
        except Exception:
            pass
    return data


def load_market_index(start: str, end: str,
                      ticker: str = "^GSPC") -> Optional[pd.Series]:
    """
    Download S&P 500 (or any index) log returns.
    Returns a pd.Series of log returns already shifted by 1 day (t-1)
    so it can be safely used as a feature at time t without leakage.

    The caller receives the lagged series; compute_features aligns it to
    the stock DataFrame index using reindex + ffill.
    """
    try:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        price_col = "Close" if "Close" in df.columns else df.columns[0]
        price = df[price_col].ffill().dropna()
        log_ret = np.log(price / price.shift(1)).dropna()
        # Shift by 1: at time t we know yesterday's SPY return (t-1) → no leakage
        return log_ret.shift(1)
    except Exception:
        return None


def prepare_splits(
    data: Dict[str, pd.DataFrame],
    selected_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, Dict[str, pd.Series]]:
    splits = {}
    for sym, df in data.items():
        price = df[selected_col].dropna()
        train, val, test = walk_forward_split(price, train_ratio, val_ratio)
        splits[sym] = {"train": train, "val": val, "test": test, "full": price}
    return splits


def prepare_order_info(splits: Dict, selected_col: str) -> Dict:
    order_info = {}
    for sym, sp in splits.items():
        series = np.log(sp["train"]).diff().dropna()
        d, is_stat, stat_info = stationarity_analysis(series, sym)
        order = find_best_arima_order(series, d)
        order_info[sym] = {
            "d": d, "order": order,
            "is_stationary": is_stat,
            "stat_info": stat_info
        }
    return order_info


def run_walk_forward_cv(
    sym: str,
    price: pd.Series,
    order_info: Dict,
    n_splits: int = 5,
    refit_every: int = 50,
    confidence: float = 0.95,
) -> List[Dict]:
    """
    Expanding-window walk-forward CV for ARIMA (fast reference model).
    Returns list of per-fold metric dicts.
    """
    windows = generate_walk_forward_windows(price, n_splits=n_splits)
    fold_results = []

    arima_order = order_info[sym]["order"]
    d = order_info[sym]["d"]

    for fold_idx, (train, test) in enumerate(windows):
        if len(train) < 50 or len(test) < 5:
            continue

        def make_fn(ord_=arima_order):
            def fn(hist):
                m = SARIMAX(hist, order=ord_, seasonal_order=(0,0,0,0),
                            enforce_stationarity=False, enforce_invertibility=False)
                return SARIMAXWrapper(m.fit(disp=False, maxiter=200))
            return fn

        r = rolling_one_step(train, test, make_fn(), refit_every, confidence=confidence)
        if r:
            m = compute_metrics(test, r["pred"])
            fold_results.append({
                "fold": fold_idx + 1,
                "train_size": len(train),
                "test_size": len(test),
                **m
            })

    return fold_results