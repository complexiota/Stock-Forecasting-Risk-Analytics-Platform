# 📈 QuantLens: Professional Stock Forecasting Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**QuantLens** is a professional-grade quantitative finance and stock market analysis application. Built with Streamlit, it provides an advanced, dark-themed fintech dashboard for visualizing stock data, running predictive models, and performing in-depth risk analytics.

---

## ✨ Features

*   **Interactive Dashboards**: A sleek, dark-themed UI with custom CSS and typography designed for a premium fintech experience.
*   **Real-Time Data**: Fetch and analyze real-time market data using `yfinance`.
*   **Advanced Charting**: Interactive, high-performance financial charts built with Plotly.
*   **Predictive Modeling**: Incorporates machine learning and statistical models (XGBoost, Scikit-Learn, ARIMA, GARCH) to forecast market trends.
*   **Risk Analytics**: Comprehensive tools to assess portfolio and asset risk metrics.

## 🛠️ Tech Stack

*   **Frontend/UI**: [Streamlit](https://streamlit.io/)
*   **Data Manipulation**: Pandas, NumPy, SciPy
*   **Data Source**: yfinance
*   **Visualization**: Plotly
*   **Machine Learning & Stats**: Scikit-Learn, XGBoost, Statsmodels, ARCH, pmdarima

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/complexiota/Stock-Forecasting-Risk-Analytics-Platform.git
   cd Stock-Forecasting-Risk-Analytics-Platform
   ```

2. **Create a virtual environment (Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r Stock_Market_Analysis/Requirements
   ```

## 💻 Usage

To run the application locally, navigate to the source directory and start the Streamlit server:

```bash
cd Stock_Market_Analysis
streamlit run App.py
```

The application will launch in your default web browser (typically at `http://localhost:8501`).

## 📁 Project Structure

```text
Stock_Market_Analysis/
├── App.py                  # Main Streamlit application & UI layout
├── Requirements            # Project dependencies
└── core/                   # Core analytical modules
    ├── charts.py           # Plotly-based interactive financial charting
    ├── data.py             # Market data ingestion and preprocessing
    ├── models.py           # Forecasting models (ML/Stats)
    └── risk.py             # Risk calculation and analytics
```

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. It does not constitute financial advice. Always perform your own due diligence before making investment decisions.
