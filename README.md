<div align="center">

# 📈 QuantLens

### Professional Stock Forecasting & Risk Analytics Platform

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-006600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**An institutional-grade quantitative analysis dashboard that combines multi-model forecasting, walk-forward cross-validation, and comprehensive VaR risk analytics — all wrapped in a sleek, dark-themed fintech UI.**

[Getting Started](#-getting-started) · [Features](#-key-features) · [Models](#-forecasting-models) · [Risk Engine](#️-risk--var-engine) · [Architecture](#-architecture)

---

</div>

## 🎯 Overview

**QuantLens** is a production-ready stock forecasting and risk analytics platform built for quantitative analysts, portfolio managers, and fintech enthusiasts. It provides an end-to-end pipeline — from real-time data ingestion to multi-model forecasting with **leakage-free validation**, institutional-grade **VaR backtesting**, and **strategy simulation with Kelly sizing**.

> **What makes QuantLens different?**
> Every model is validated through expanding-window walk-forward cross-validation. All features are strictly backward-looking. Market index features are lagged by t-1. GARCH volatility is fitted on training data only. There is **zero lookahead bias** by design.

---

## ✨ Key Features

### 📊 Multi-Model Forecasting Engine
- **8+ forecasting models** trained and compared simultaneously
- Walk-forward cross-validation with expanding windows (5-fold)
- Automatic model ranking by MAPE, RMSE, MAE, and Directional Accuracy
- Bootstrap prediction bands with configurable confidence intervals
- Future price forecasting with uncertainty quantification

### 🛡️ Institutional-Grade Risk Analytics
- **7 VaR estimation methods** run in parallel (Historical, Parametric, EWMA, GARCH, GJR-GARCH, Cornish-Fisher, Regime-Conditional)
- Kupiec POF test, Christoffersen CC test, and Binomial Unconditional Coverage
- Automated best-method selection via composite scoring
- Expected Shortfall (CVaR) computation
- Conservative buffer for real-world capital adequacy

### 📡 Signal Generation & Strategy Backtesting
- Multi-signal consensus voting (momentum, mean-reversion, volatility breakout)
- Strategy backtest with Kelly criterion position sizing
- Rolling Sharpe ratio analysis vs buy-and-hold benchmark
- Configurable stop-loss and confidence thresholds

### 🔬 150+ Engineered Features
- Return lags, rolling log-return sums, price lag ratios
- **4 volatility estimators**: Close-to-Close, Parkinson, Garman-Klass, Yang-Zhang
- Rolling Hurst exponent for regime detection (trending vs. mean-reverting)
- Kaufman Efficiency Ratio, MACD, RSI, Bollinger Bands, OBV
- Realized skewness, kurtosis, and Volatility-of-Volatility (VoV)
- Drawdown from peak, rolling autocorrelation
- S&P 500 lagged returns as market feature (no leakage)
- **Mutual Information-based feature selection** (top 30 features)

### 🎨 Premium Dark-Themed Dashboard
- Custom dark fintech aesthetic with IBM Plex Mono + Space Grotesk typography
- Interactive Plotly charts with unified hover and zoom
- Real-time library status detection (PyTorch, XGBoost, CatBoost, ARCH)
- Responsive multi-tab layout per symbol

---

## 🧠 Forecasting Models

| Model | Type | Description |
|:------|:-----|:------------|
| **ARIMA** | Statistical | Auto-selected (p,d,q) via `pmdarima` with ADF/KPSS stationarity testing |
| **Log-ARIMA** | Statistical | ARIMA on log-transformed prices for multiplicative dynamics |
| **Holt-Winters** | Statistical | Exponential smoothing with additive trend |
| **XGBoost-L1L2** | Machine Learning | Gradient boosting with strong L1/L2 regularization (α=3, λ=8) |
| **XGBoost-Directional** | Machine Learning | Custom loss function blending MSE (30%) + directional hinge (70%) |
| **GBM-Conservative** | Machine Learning | Scikit-Learn GBM with early stopping and validation-fraction holdout |
| **RandomForest** | Machine Learning | Ensemble of 200 trees with restricted depth and min-leaf constraints |
| **CatBoost** | Machine Learning | Gradient boosting with ordered boosting (optional) |
| **LSTM** | Deep Learning | Long Short-Term Memory recurrent network (requires PyTorch) |
| **GRU** | Deep Learning | Gated Recurrent Unit network (requires PyTorch) |
| **XGB+GARCH** | Hybrid | XGBoost predictions combined with GARCH volatility estimates |
| **Direction-Vote** | Ensemble | Multi-model consensus voting for directional accuracy |

### Custom Directional Loss Function

QuantLens includes a novel XGBoost objective that explicitly penalizes wrong-direction predictions:

```
L = α·MSE + (1-α)·log(1 + exp(-penalty · z))

where z = (pred - prev) · (actual - prev) / σ²
```

This pushes the model to predict the correct side of yesterday's price, improving Directional Accuracy by **+5–8%** with only a **+1–3% MAPE** trade-off.

---

## 🛡️ Risk & VaR Engine

QuantLens runs **7 Value-at-Risk methods in parallel** and auto-selects the best via composite scoring:

| Method | Key Property |
|:-------|:-------------|
| **Historical Simulation** | Non-parametric; exact empirical quantile |
| **Parametric (Gaussian)** | Assumes normal returns; fast baseline |
| **EWMA (RiskMetrics)** | Exponentially weighted; adapts quickly to volatility clusters |
| **GARCH(1,1)-t** | Conditional heteroskedasticity; Student-t tails |
| **GJR-GARCH(1,1,1)** | Asymmetric volatility — captures leverage effect |
| **Cornish-Fisher** | Adjusts for skewness and excess kurtosis |
| **Regime-Conditional** | Separate vol estimates for bull/bear markets |

**Backtesting Framework:**
- **Kupiec (1995)** — Proportion of Failures likelihood ratio test
- **Christoffersen (1998)** — Conditional Coverage (independence + coverage)
- **Binomial UC** — Exact binomial test on breach count
- **Composite Score** = 40% × Kupiec p + 40% × proximity + 20% × Christoffersen p

---

## 🔒 Leakage Prevention

QuantLens enforces strict data integrity at every stage:

```
✅ Target           = price.shift(-1)        → predict tomorrow, not today
✅ All features      = backward-looking only  → no future data in any rolling window
✅ S&P 500 returns   = lagged by t-1          → at time t, use yesterday's market return
✅ GARCH volatility  = fitted on train only   → OOS volatility forecasts for test
✅ Walk-forward CV   = expanding window       → no shuffling, preserves temporal order
✅ Gap filling       = ffill only (no bfill)  → never pull future prices backward
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/complexiota/Stock-Forecasting-Risk-Analytics-Platform.git
   cd Stock-Forecasting-Risk-Analytics-Platform
   ```

2. **Create a virtual environment** *(recommended)*
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r Stock_Market_Analysis/Requirements
   ```

4. **Optional: Install deep learning & advanced libraries**
   ```bash
   # For LSTM/GRU models
   pip install torch
   
   # For CatBoost models
   pip install catboost
   
   # For GARCH-based VaR
   pip install arch
   ```

### Usage

```bash
cd Stock_Market_Analysis
streamlit run App.py
```

The dashboard opens at `http://localhost:8501`. Configure symbols, date range, and model parameters in the sidebar, then click **🚀 Run Analysis**.

---

## 📁 Architecture

```
Stock_Market_Analysis/
│
├── App.py                      # Main Streamlit application (2600+ lines)
│                                 # UI layout, sidebar config, pipeline orchestration
│                                 # 9 interactive tabs per symbol
│
├── Requirements                # Python dependencies (pip install -r)
│
└── core/                       # Modular analytical engine
    │
    ├── models.py               # Forecasting models & feature engineering (2400+ lines)
    │                             # - 150+ feature computation (compute_features)
    │                             # - ARIMA/Log-ARIMA/Holt-Winters wrappers
    │                             # - ML model configs with anti-overfit regularization
    │                             # - Custom directional XGBoost objective
    │                             # - LSTM/GRU with PyTorch
    │                             # - Walk-forward validation engine
    │                             # - Mutual Information feature selection
    │                             # - Bootstrap prediction bands
    │                             # - Signal generation & strategy backtest
    │
    ├── risk.py                 # Risk analytics & VaR engine (500+ lines)
    │                             # - 7 VaR estimation methods
    │                             # - Kupiec, Christoffersen, Binomial backtests
    │                             # - Parallel execution with ThreadPoolExecutor
    │                             # - Composite scoring & auto-selection
    │                             # - GJR-GARCH & Regime-Conditional VaR
    │
    ├── charts.py               # Interactive Plotly chart builders (840+ lines)
    │                             # - Dark fintech theme system
    │                             # - Price, forecast, VaR, residual charts
    │                             # - Walk-forward CV visualization
    │                             # - Feature importance & returns distribution
    │                             # - Rolling Sharpe ratio comparison
    │
    └── data.py                 # Data ingestion & preprocessing (134 lines)
                                  # - yfinance download with caching
                                  # - S&P 500 market index (lagged, leakage-free)
                                  # - Train/Val/Test splitting
                                  # - Walk-forward CV window generation
```

---

## 🖥️ Dashboard Tabs

Each analyzed symbol gets **9 interactive tabs**:

| Tab | Contents |
|:----|:---------|
| 🕯️ **Price** | Train/Val/Test split visualization, ARIMA order selection, stationarity tests |
| 🔮 **Forecast** | Actual vs Predicted overlay with prediction bands, directional accuracy chart |
| 📊 **Model Comparison** | Side-by-side MAPE, RMSE, DA bar charts across all models |
| 🔄 **Walk-Forward CV** | Per-fold MAPE, RMSE, MAE, DA with mean reference lines |
| 🛡️ **Risk & VaR** | All 7 VaR methods overlaid, breach markers, method comparison dashboard |
| 📉 **Residuals** | Time series, ACF, distribution, and predicted-vs-actual scatter |
| 🌅 **Future Forecast** | N-day ahead forecast with confidence bands from the best model |
| 📡 **Signals** | Multi-signal consensus table with buy/sell/hold recommendations |
| ⚡ **Strategy Backtest** | Equity curve, rolling Sharpe, Kelly sizing, and drawdown analysis |

---

## ⚙️ Configuration

All parameters are configurable through the sidebar:

| Parameter | Default | Range | Description |
|:----------|:--------|:------|:------------|
| Symbols | `AAPL, MSFT` | Any Yahoo Finance ticker | Comma-separated list |
| Train Ratio | 0.70 | 0.50–0.80 | Fraction of data for training |
| Val Ratio | 0.15 | 0.05–0.25 | Fraction of data for validation |
| Forecast Horizon | 15 days | 5–60 | Future forecast length |
| Refit Every | 50 steps | 10–100 | Walk-forward refit frequency |
| Confidence Level | 0.95 | 0.90–0.99 | Prediction band width |
| VaR Window | 252 days | 60–504 | Rolling window for VaR estimation |
| Conservative Buffer | 10% | 0–20% | Inflates VaR to reduce breaches |

---

## 🛠️ Tech Stack

| Category | Libraries |
|:---------|:----------|
| **UI Framework** | Streamlit 1.32+ |
| **Data Source** | yfinance |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Visualization** | Plotly 5.20+ |
| **Statistical Models** | Statsmodels, pmdarima |
| **Machine Learning** | Scikit-Learn, XGBoost, CatBoost |
| **Deep Learning** | PyTorch (optional) |
| **Volatility Modeling** | ARCH 6.3+ (GARCH/GJR-GARCH) |

---

## ⚠️ Disclaimer

> **This application is for educational and research purposes only.** It does not constitute financial advice, investment recommendations, or trading signals. Past performance does not guarantee future results. Always conduct your own due diligence and consult a licensed financial advisor before making investment decisions. The authors assume no liability for any financial losses incurred from the use of this software.

---

<div align="center">

**Built with ❤️ for Quantitative Finance**

*If you find this useful, consider giving it a ⭐*

</div>
