"""
core/charts.py
==============
All Plotly chart builders for the Streamlit dashboard.
Follows a dark-themed, professional fintech aesthetic.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

THEME = dict(
    bg       = "#0D1117",
    surface  = "#161B22",
    surface2 = "#21262D",
    border   = "#30363D",
    text     = "#E6EDF3",
    muted    = "#8B949E",
    green    = "#3FB950",
    red      = "#F85149",
    blue     = "#58A6FF",
    orange   = "#D29922",
    purple   = "#BC8CFF",
    cyan     = "#39D5FF",
    yellow   = "#E3B341",
)

MODEL_PALETTE = [
    THEME["blue"], THEME["green"], THEME["orange"],
    THEME["purple"], THEME["cyan"], THEME["yellow"],
    THEME["red"], "#FF7B72", "#56D364", "#A5D6FF",
]

def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert #RRGGBB hex string to rgba() with given alpha."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor = THEME["bg"],
        plot_bgcolor  = THEME["surface"],
        font          = dict(family="'IBM Plex Mono', monospace", color=THEME["text"], size=12),
        xaxis         = dict(gridcolor=THEME["border"], zerolinecolor=THEME["border"]),
        yaxis         = dict(gridcolor=THEME["border"], zerolinecolor=THEME["border"]),
        legend        = dict(bgcolor=THEME["surface2"], bordercolor=THEME["border"],
                             borderwidth=1, font=dict(size=11)),
        margin        = dict(l=60, r=30, t=50, b=40),
        **kwargs
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRICE CHART
# ─────────────────────────────────────────────────────────────────────────────

def chart_price(
    train: pd.Series, val: pd.Series, test: pd.Series,
    symbol: str, selected_col: str
) -> go.Figure:
    fig = go.Figure()
    for series, name, color, alpha in [
        (train, "Train",      THEME["blue"],   0.7),
        (val,   "Validation", THEME["orange"], 0.8),
        (test,  "Test",       THEME["green"],  0.9),
    ]:
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=name,
            line=dict(color=color, width=1.5),
            opacity=alpha,
        ))

    # Add split markers — use add_shape to avoid Plotly datetime _mean() crash
    if len(train) and len(val):
        for split_date, label in [
            (val.index[0], "Train/Val"), (test.index[0], "Val/Test")
        ]:
            x_str = str(split_date)
            fig.add_shape(
                type="line",
                x0=x_str, x1=x_str,
                y0=0, y1=1,
                yref="paper",
                line=dict(color=THEME["muted"], width=1, dash="dash"),
            )
            fig.add_annotation(
                x=x_str, y=0.98,
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(color=THEME["muted"], size=11),
                xanchor="left", yanchor="top",
            )

    fig.update_layout(
        **base_layout(title=f"{symbol} — {selected_col} Price (Train / Val / Test)",
                      hovermode="x unified"),
        xaxis_title="Date", yaxis_title=selected_col,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ACTUAL vs PREDICTED + BANDS
# ─────────────────────────────────────────────────────────────────────────────

def chart_forecast(
    actual: pd.Series, results: Dict[str, Any],
    symbol: str, top_n: int = 3,
    show_bands: bool = True,
    history: Optional[pd.Series] = None,
) -> go.Figure:
    """
    Actual vs predicted for top_n models with optional prediction bands.
    Optionally prepend recent history for context.
    """
    fig = go.Figure()

    # Recent history context
    if history is not None and len(history) > 0:
        context = history.iloc[-60:]
        fig.add_trace(go.Scatter(
            x=context.index, y=context.values, name="History",
            line=dict(color=THEME["muted"], width=1), opacity=0.5,
        ))

    # Actual
    fig.add_trace(go.Scatter(
        x=actual.index, y=actual.values, name="Actual",
        line=dict(color=THEME["text"], width=2),
    ))

    # Sort by MAPE
    sorted_results = sorted(
        [(n, r) for n, r in results.items() if "pred" in r and r["mape"] is not None],
        key=lambda x: x[1]["mape"]
    )[:top_n]

    for i, (name, r) in enumerate(sorted_results):
        color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        mape_str = f"{r['mape']:.2f}%" if not np.isnan(r['mape']) else "N/A"
        da_str   = f"{r.get('da', np.nan):.1f}%" if not np.isnan(r.get('da', np.nan)) else "N/A"
        label    = f"{name} (MAPE={mape_str}, DA={da_str})"

        fig.add_trace(go.Scatter(
            x=r["pred"].index, y=r["pred"].values,
            name=label, line=dict(color=color, width=1.5, dash="dot"),
        ))

        # Prediction bands
        if show_bands and "pred_lower" in r and "pred_upper" in r:
            lower = r["pred_lower"]
            upper = r["pred_upper"]
            if not lower.isna().all():
                fig.add_trace(go.Scatter(
                    x=list(upper.index) + list(lower.index[::-1]),
                    y=list(upper.values) + list(lower.values[::-1]),
                    fill="toself", fillcolor=hex_to_rgba(color, alpha=0.15),
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{name} Band", showlegend=False,
                ))

    fig.update_layout(
        **base_layout(title=f"{symbol} — Forecast vs Actual (Top {top_n} Models)",
                      hovermode="x unified"),
        xaxis_title="Date", yaxis_title="Price",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def chart_model_comparison(results: Dict[str, Any], symbol: str) -> go.Figure:
    rows = []
    for name, r in results.items():
        if r.get("mape") is None:
            continue
        rows.append({
            "Model": name,
            "MAPE %": r.get("mape", np.nan),
            "RMSE":   r.get("rmse", np.nan),
            "MAE":    r.get("mae", np.nan),
            "DA %":   r.get("da", np.nan),
            "Type":   r.get("type", ""),
        })
    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows).sort_values("MAPE %").reset_index(drop=True)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["MAPE % (↓ better)", "RMSE (↓ better)", "Directional Accuracy % (↑ better)"],
        horizontal_spacing=0.08,
    )

    type_colors = {
        "arima":       THEME["blue"],
        "log_arima":   THEME["purple"],
        "holt_winters": THEME["orange"],
        "ml":          THEME["red"],
        "ensemble":    THEME["green"],
        "lstm":        THEME["cyan"],
        "gru":         THEME["yellow"],
    }

    colors = [type_colors.get(t, THEME["muted"]) for t in df["Type"]]

    for col_idx, (metric, ascending) in enumerate([
        ("MAPE %", True), ("RMSE", True), ("DA %", False)
    ], 1):
        df_sorted = df.sort_values(metric, ascending=ascending).dropna(subset=[metric])
        fig.add_trace(go.Bar(
            x=df_sorted[metric], y=df_sorted["Model"],
            orientation="h",
            marker_color=[type_colors.get(t, THEME["muted"]) for t in df_sorted["Type"]],
            showlegend=False, text=df_sorted[metric].round(2).astype(str),
            textposition="auto",
        ), row=1, col=col_idx)

    fig.update_layout(
        **base_layout(title=f"{symbol} — Model Comparison"),
        height=max(350, len(df) * 22 + 100),
    )
    for axis in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
        fig.update_layout(**{axis: dict(gridcolor=THEME["border"])})
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# VaR BREACH CHART
# ─────────────────────────────────────────────────────────────────────────────

def chart_var_breaches(
    returns: pd.Series, var_series: pd.Series,
    breaches: pd.Series, symbol: str,
    confidence: float = 0.95,
    method_name: str = "",
) -> go.Figure:
    """Single-method VaR breach chart (used when only one method is shown)."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=returns.index, y=returns.values,
        name="Log Returns", line=dict(color=THEME["blue"], width=1),
        opacity=0.7,
    ))

    neg_var = -var_series
    label = f"VaR {confidence*100:.0f}% {('· ' + method_name) if method_name else ''}"
    fig.add_trace(go.Scatter(
        x=var_series.index, y=neg_var.values,
        name=label,
        line=dict(color=THEME["orange"], width=1.5, dash="dash"),
    ))

    breach_dates = breaches[breaches == 1].index
    if len(breach_dates):
        fig.add_trace(go.Scatter(
            x=breach_dates, y=returns.loc[breach_dates].values,
            mode="markers", name="Breaches",
            marker=dict(color=THEME["red"], size=7, symbol="x"),
        ))

    fig.update_layout(
        **base_layout(
            title=f"{symbol} — VaR Backtest ({confidence*100:.0f}% CI, "
                  f"{int(breaches.sum())} breaches / {len(breaches)} obs)",
            hovermode="x unified"
        ),
        xaxis_title="Date", yaxis_title="Log Return",
    )
    return fig


def chart_var_all_methods(
    returns: pd.Series,
    all_var_results: Dict[str, Any],
    symbol: str,
    confidence: float = 0.95,
    best_method: str = "",
) -> go.Figure:
    """
    Overlay chart showing all VaR methods simultaneously on one figure.
    Best method is highlighted with a thicker line; others are dimmed.
    Breaches from the best method are marked.
    """
    METHOD_COLORS = {
        "historical":         "#58A6FF",
        "parametric":         "#D29922",
        "ewma":               "#BC8CFF",
        "garch":              "#3FB950",
        "gjr_garch":          "#56D364",
        "cornish_fisher":     "#39D5FF",
        "regime_conditional": "#FF7B72",
    }
    METHOD_LABELS = {
        "historical":         "Historical Sim",
        "parametric":         "Parametric",
        "ewma":               "EWMA",
        "garch":              "GARCH(1,1)-t",
        "gjr_garch":          "GJR-GARCH(1,1,1)",
        "cornish_fisher":     "Cornish-Fisher",
        "regime_conditional": "Regime-Conditional",
    }

    fig = go.Figure()

    # Returns as background
    fig.add_trace(go.Scatter(
        x=returns.index, y=returns.values,
        name="Log Returns",
        line=dict(color=THEME["muted"], width=0.8),
        opacity=0.5,
    ))

    # All VaR lines
    for method, result in all_var_results.items():
        if "error" in result or "var_series" not in result:
            continue
        is_best  = (method == best_method)
        color    = METHOD_COLORS.get(method, THEME["muted"])
        label    = METHOD_LABELS.get(method, method)
        score    = result.get("score", 0.0)
        obs_rate = result.get("obs_rate", 0.0)
        n_breach = result.get("n_breaches", 0)

        suffix = (f" ★ BEST" if is_best else "")
        name   = (f"{label}{suffix} "
                  f"[breach {obs_rate*100:.2f}%, score {score:.2f}]")

        fig.add_trace(go.Scatter(
            x=result["var_series"].index,
            y=(-result["var_series"]).values,
            name=name,
            line=dict(
                color=color,
                width=2.5 if is_best else 1.0,
                dash="solid" if is_best else "dot",
            ),
            opacity=1.0 if is_best else 0.6,
        ))

    # Breach markers from best method
    if best_method and best_method in all_var_results:
        best = all_var_results[best_method]
        if "breaches" in best:
            breach_dates = best["breaches"][best["breaches"] == 1].index
            if len(breach_dates):
                fig.add_trace(go.Scatter(
                    x=breach_dates,
                    y=returns.loc[breach_dates].values,
                    mode="markers", name=f"Breaches ({best_method})",
                    marker=dict(color=THEME["red"], size=7, symbol="x"),
                ))

    fig.update_layout(
        **base_layout(
            title=f"{symbol} — All VaR Methods Overlay "
                  f"({confidence*100:.0f}% CI) · Best: {METHOD_LABELS.get(best_method, best_method)}",
            hovermode="x unified",
        ),
        xaxis_title="Date", yaxis_title="Log Return",
    )
    return fig


def chart_var_method_comparison(
    all_var_results: Dict[str, Any],
    symbol: str,
    confidence: float = 0.95,
    best_method: str = "",
) -> go.Figure:
    """
    Bar chart comparing all VaR methods across: score, breach rate vs expected,
    Kupiec p-value, and Christoffersen p-value.
    """
    from plotly.subplots import make_subplots

    METHOD_COLORS = {
        "historical":         "#58A6FF",
        "parametric":         "#D29922",
        "ewma":               "#BC8CFF",
        "garch":              "#3FB950",
        "gjr_garch":          "#56D364",
        "cornish_fisher":     "#39D5FF",
        "regime_conditional": "#FF7B72",
    }
    METHOD_LABELS = {
        "historical":         "Historical",
        "parametric":         "Parametric",
        "ewma":               "EWMA",
        "garch":              "GARCH",
        "gjr_garch":          "GJR-GARCH",
        "cornish_fisher":     "Cornish-Fisher",
        "regime_conditional": "Regime-Cond.",
    }

    rows = []
    for method, result in all_var_results.items():
        if "error" in result:
            continue
        rows.append({
            "method":   method,
            "label":    METHOD_LABELS.get(method, method),
            "score":    result.get("score", 0.0),
            "obs_rate": result.get("obs_rate", 0.0) * 100,
            "exp_rate": result.get("exp_rate", 1-confidence) * 100,
            "kup_p":    result.get("kupiec", {}).get("p_value", 0.0) or 0.0,
            "chr_p":    result.get("christoffersen", {}).get("p_value_cc", 0.0) or 0.0,
            "n_breach": result.get("n_breaches", 0),
            "color":    METHOD_COLORS.get(method, THEME["muted"]),
            "is_best":  method == best_method,
        })

    if not rows:
        return go.Figure()

    labels  = [r["label"] for r in rows]
    colors  = [r["color"] for r in rows]
    borders = ["#FFFFFF" if r["is_best"] else r["color"] for r in rows]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Composite Score (↑ best)",
            "Breach Rate vs Expected %",
            "Kupiec p-value (↑ better, >0.05 = pass)",
            "Christoffersen p-value (↑ better, >0.05 = pass)",
        ],
        vertical_spacing=0.18, horizontal_spacing=0.10,
    )

    # Score
    fig.add_trace(go.Bar(
        x=labels, y=[r["score"] for r in rows],
        marker_color=colors, name="Score",
        text=[f"{r['score']:.3f}" for r in rows], textposition="auto",
        showlegend=False,
    ), row=1, col=1)

    # Breach rate vs expected
    fig.add_trace(go.Bar(
        x=labels, y=[r["obs_rate"] for r in rows],
        marker_color=colors, name="Obs Rate",
        text=[f"{r['obs_rate']:.2f}%" for r in rows], textposition="auto",
        showlegend=False,
    ), row=1, col=2)
    # Expected line
    exp_val = (1 - confidence) * 100
    fig.add_shape(type="line", x0=-0.5, x1=len(rows)-0.5, y0=exp_val, y1=exp_val,
                  line=dict(color=THEME["red"], dash="dash", width=1.5),
                  row=1, col=2)
    fig.add_annotation(x=len(rows)-1, y=exp_val, text=f"Expected {exp_val:.1f}%",
                       showarrow=False, font=dict(color=THEME["red"], size=10),
                       xanchor="right", row=1, col=2)

    # Kupiec p
    kup_colors = [THEME["green"] if r["kup_p"] >= 0.05 else THEME["red"] for r in rows]
    fig.add_trace(go.Bar(
        x=labels, y=[r["kup_p"] for r in rows],
        marker_color=kup_colors, name="Kupiec p",
        text=[f"{r['kup_p']:.3f}" for r in rows], textposition="auto",
        showlegend=False,
    ), row=2, col=1)
    fig.add_shape(type="line", x0=-0.5, x1=len(rows)-0.5, y0=0.05, y1=0.05,
                  line=dict(color=THEME["orange"], dash="dash", width=1.5),
                  row=2, col=1)

    # Christoffersen p
    chr_colors = [THEME["green"] if r["chr_p"] >= 0.05 else THEME["red"] for r in rows]
    fig.add_trace(go.Bar(
        x=labels, y=[r["chr_p"] for r in rows],
        marker_color=chr_colors, name="Christoffersen p",
        text=[f"{r['chr_p']:.3f}" for r in rows], textposition="auto",
        showlegend=False,
    ), row=2, col=2)
    fig.add_shape(type="line", x0=-0.5, x1=len(rows)-0.5, y0=0.05, y1=0.05,
                  line=dict(color=THEME["orange"], dash="dash", width=1.5),
                  row=2, col=2)

    fig.update_layout(
        **base_layout(title=f"{symbol} — VaR Method Comparison (★ = auto-selected best)"),
        height=520,
    )
    for axis in ["xaxis","xaxis2","xaxis3","xaxis4",
                 "yaxis","yaxis2","yaxis3","yaxis4"]:
        fig.update_layout(**{axis: dict(gridcolor=THEME["border"])})
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD CV CHART
# ─────────────────────────────────────────────────────────────────────────────

def chart_walk_forward(fold_results: List[Dict], symbol: str) -> go.Figure:
    if not fold_results:
        return go.Figure()

    df = pd.DataFrame(fold_results)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["MAPE % per Fold", "RMSE per Fold",
                        "MAE per Fold", "Directional Accuracy %"],
        vertical_spacing=0.18,
    )

    metrics = [
        ("mape", THEME["blue"],   1, 1),
        ("rmse", THEME["orange"], 1, 2),
        ("mae",  THEME["green"],  2, 1),
        ("da",   THEME["purple"], 2, 2),
    ]

    for metric, color, row, col in metrics:
        if metric not in df.columns:
            continue
        fig.add_trace(go.Bar(
            x=df["fold"].astype(str), y=df[metric],
            name=metric.upper(), marker_color=color,
            showlegend=False,
        ), row=row, col=col)
        # Mean reference line
        mean_val = df[metric].mean()
        fig.add_hline(y=mean_val, line_dash="dash",
                      line_color=THEME["muted"], line_width=1,
                      row=row, col=col)

    fig.update_layout(
        **base_layout(title=f"{symbol} — Walk-Forward CV (Expanding Window)"),
        height=500,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTIONAL ACCURACY HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def chart_directional_accuracy(results: Dict[str, Any], symbol: str) -> go.Figure:
    rows = []
    for name, r in results.items():
        da = r.get("da", np.nan)
        mape = r.get("mape", np.nan)
        if not np.isnan(da) and not np.isnan(mape):
            rows.append({"Model": name, "DA %": da, "MAPE %": mape})
    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows).sort_values("DA %", ascending=False)

    fig = go.Figure(go.Bar(
        x=df["DA %"], y=df["Model"],
        orientation="h",
        marker=dict(
            color=df["DA %"],
            colorscale=[[0, THEME["red"]], [0.5, THEME["orange"]], [1, THEME["green"]]],
            cmin=40, cmax=70,
            colorbar=dict(title="DA %", tickfont=dict(color=THEME["text"])),
        ),
        text=df["DA %"].round(1).astype(str) + "%",
        textposition="auto",
    ))

    fig.add_vline(x=50, line_dash="dash", line_color=THEME["muted"])
    fig.add_annotation(
        x=50, y=1, yref="paper",
        text="Random (50%)",
        showarrow=False,
        font=dict(color=THEME["muted"], size=11),
        xanchor="left", yanchor="top",
    )

    fig.update_layout(
        **base_layout(title=f"{symbol} — Directional Accuracy (% correct direction)"),
        xaxis_title="Directional Accuracy %", height=max(300, len(df) * 28 + 80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FUTURE FORECAST CHART
# ─────────────────────────────────────────────────────────────────────────────

def chart_future_forecast(
    full_series: pd.Series, future_pred: pd.Series,
    symbol: str, model_name: str,
    lower: Optional[pd.Series] = None,
    upper: Optional[pd.Series] = None,
) -> go.Figure:
    fig = go.Figure()

    # Last 120 days of history
    hist = full_series.iloc[-120:]
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, name="Historical",
        line=dict(color=THEME["blue"], width=2),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=future_pred.index, y=future_pred.values, name=f"Forecast ({model_name})",
        line=dict(color=THEME["green"], width=2.5, dash="dot"),
        mode="lines+markers", marker=dict(size=5),
    ))

    # Confidence/Prediction band
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=list(upper.index) + list(lower.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill="toself", fillcolor="rgba(63,185,80,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Prediction Band",
        ))

    # Last actual price annotation
    last_price = full_series.iloc[-1]
    fig.add_annotation(
        x=full_series.index[-1], y=last_price,
        text=f"Last: {last_price:.2f}",
        showarrow=True, arrowhead=2,
        font=dict(color=THEME["text"]), arrowcolor=THEME["muted"],
    )

    fig.update_layout(
        **base_layout(title=f"{symbol} — Future Forecast ({len(future_pred)}d) via {model_name}",
                      hovermode="x unified"),
        xaxis_title="Date", yaxis_title="Price",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RETURNS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def chart_returns_dist(returns: pd.Series, symbol: str, var95: float, var99: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Return Distribution", "Q-Q Plot vs Normal"],
                        horizontal_spacing=0.1)

    # Histogram with KDE overlay
    fig.add_trace(go.Histogram(
        x=returns.values, nbinsx=80,
        name="Returns",
        marker_color=THEME["blue"], opacity=0.7,
        histnorm="probability density",
    ), row=1, col=1)

    # Normal overlay
    x_range = np.linspace(returns.min(), returns.max(), 200)
    from scipy.stats import norm
    y_norm = norm.pdf(x_range, returns.mean(), returns.std())
    fig.add_trace(go.Scatter(
        x=x_range, y=y_norm, name="Normal fit",
        line=dict(color=THEME["orange"], width=2),
    ), row=1, col=1)

    # VaR lines
    for var_val, label, color in [(-var95, "VaR 95%", THEME["orange"]),
                                   (-var99, "VaR 99%", THEME["red"])]:
        fig.add_vline(x=var_val, line_dash="dash", line_color=color, row=1, col=1)
        fig.add_annotation(
            x=var_val, y=1, yref="paper",
            text=label, showarrow=False,
            font=dict(color=color, size=11),
            xanchor="left", yanchor="top",
        )

    # Q-Q plot
    from scipy.stats import probplot
    qq = probplot(returns.dropna().values, dist="norm")
    fig.add_trace(go.Scatter(
        x=qq[0][0], y=qq[0][1], mode="markers",
        name="Q-Q", marker=dict(color=THEME["blue"], size=3),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1],
        mode="lines", name="Normal Line",
        line=dict(color=THEME["red"], width=2),
    ), row=1, col=2)

    fig.update_layout(
        **base_layout(title=f"{symbol} — Return Distribution & Normality"),
        height=400, showlegend=True,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def chart_feature_importance(importance: pd.Series, model_name: str, symbol: str, top_n: int = 20) -> go.Figure:
    top = importance.head(top_n)
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker_color=THEME["blue"], opacity=0.85,
    ))
    fig.update_layout(
        **base_layout(title=f"{symbol} — Feature Importance ({model_name})"),
        xaxis_title="Importance", height=max(300, top_n * 22 + 80),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RESIDUALS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def chart_residuals(actual: pd.Series, pred: pd.Series, model_name: str, symbol: str) -> go.Figure:
    common = actual.index.intersection(pred.index)
    residuals = actual.loc[common] - pred.loc[common]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Residuals over Time", "ACF of Residuals",
                                        "Residual Distribution", "Predicted vs Actual"],
                        vertical_spacing=0.18, horizontal_spacing=0.1)

    # Residuals over time
    fig.add_trace(go.Scatter(x=residuals.index, y=residuals.values,
                              mode="lines", name="Residual",
                              line=dict(color=THEME["blue"], width=1)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=THEME["muted"], row=1, col=1)

    # ACF (manual)
    from statsmodels.tsa.stattools import acf as sm_acf
    try:
        nlags = min(40, len(residuals) // 3)
        acf_vals = sm_acf(residuals.dropna(), nlags=nlags, fft=True)
        n = len(residuals)
        conf = 1.96 / np.sqrt(n)
        fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals,
                              name="ACF", marker_color=THEME["purple"]), row=1, col=2)
        for sign in [1, -1]:
            fig.add_hline(y=sign * conf, line_dash="dash",
                          line_color=THEME["orange"], line_width=1, row=1, col=2)
    except Exception:
        pass

    # Distribution
    fig.add_trace(go.Histogram(x=residuals.values, nbinsx=60,
                                name="Residuals", marker_color=THEME["cyan"],
                                opacity=0.75, histnorm="probability density"), row=2, col=1)

    # Scatter actual vs predicted
    fig.add_trace(go.Scatter(x=actual.loc[common].values, y=pred.loc[common].values,
                              mode="markers", name="Pred vs Actual",
                              marker=dict(color=THEME["green"], size=3, opacity=0.5)), row=2, col=2)
    mn = min(actual.loc[common].min(), pred.loc[common].min())
    mx = max(actual.loc[common].max(), pred.loc[common].max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                              line=dict(color=THEME["red"], dash="dash"), name="Perfect"), row=2, col=2)

    fig.update_layout(
        **base_layout(title=f"{symbol} — Residual Analysis ({model_name})"),
        height=580, showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING SHARPE CHART
# ─────────────────────────────────────────────────────────────────────────────

def chart_rolling_sharpe(
    rolling_sharpe: pd.Series,
    symbol: str,
    window: int = 63,
    bh_returns: Optional[pd.Series] = None,
) -> go.Figure:
    """
    Rolling Sharpe ratio chart with optional buy-and-hold comparison.
    Positive (green) / negative (red) fill makes regime changes immediately visible.
    """
    fig = go.Figure()

    rs = rolling_sharpe.dropna()

    # Positive zone
    fig.add_trace(go.Scatter(
        x=rs.index, y=rs.clip(lower=0).values,
        name=f"Rolling Sharpe ({window}d) > 0",
        fill="tozeroy",
        fillcolor="rgba(63,185,80,0.25)",
        line=dict(color=THEME["green"], width=1.5),
    ))

    # Negative zone
    fig.add_trace(go.Scatter(
        x=rs.index, y=rs.clip(upper=0).values,
        name=f"Rolling Sharpe ({window}d) < 0",
        fill="tozeroy",
        fillcolor="rgba(248,81,73,0.20)",
        line=dict(color=THEME["red"], width=1.5),
    ))

    # Full line on top
    fig.add_trace(go.Scatter(
        x=rs.index, y=rs.values,
        name=f"Rolling Sharpe ({window}d)",
        line=dict(color=THEME["blue"], width=2),
    ))

    # Zero reference
    fig.add_hline(y=0, line_dash="dash", line_color=THEME["muted"], line_width=1)

    # Sharpe = 1 target
    fig.add_hline(y=1, line_dash="dot", line_color=THEME["green"], line_width=1)
    fig.add_annotation(
        x=rs.index[-1], y=1,
        text="Sharpe = 1 (target)",
        showarrow=False,
        font=dict(color=THEME["green"], size=10),
        xanchor="right",
    )

    fig.update_layout(
        **base_layout(
            title=f"{symbol} — {window}-Day Rolling Sharpe Ratio",
            hovermode="x unified",
        ),
        xaxis_title="Date",
        yaxis_title="Rolling Sharpe",
        height=380,
    )
    return fig