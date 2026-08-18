"""
core/risk.py  (IMPROVED VERSION)
================================
Changes vs original:
  • var_gjr_garch()          NEW — asymmetric GARCH, better left-tail coverage
  • regime_conditional_var() NEW — separate vol for bull/bear regimes
  • var_ewma()               FIX-R1 — corrected com= parameter
  • kupiec_pof_test()        FIX-R3 — zero-breach edge case
  • christoffersen_test()    FIX-R2 — degenerate xlogy handling
  • full_var_backtest()      PATCHED — supports new methods
  • ALL_VAR_METHODS          PATCHED — includes gjr_garch, regime_conditional
  • METHOD_LABELS            PATCHED — new entries
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple
import warnings

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


# ─────────────────────────────────────────────────────────────────────────────
# VaR ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def var_historical(returns: pd.Series, confidence: float = 0.95,
                   window: Optional[int] = None) -> pd.Series:
    alpha = 1 - confidence
    if window:
        return -returns.rolling(window).quantile(alpha)
    q = returns.quantile(alpha)
    return pd.Series(-q, index=returns.index)


def var_parametric(returns: pd.Series, confidence: float = 0.95,
                   window: Optional[int] = 252) -> pd.Series:
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    if window:
        mu    = returns.rolling(window).mean()
        sigma = returns.rolling(window).std()
    else:
        mu    = returns.mean()
        sigma = returns.std()
    return -(mu + z * sigma)


def var_ewma(returns: pd.Series, confidence: float = 0.95,
             lam: float = 0.94) -> pd.Series:
    """
    EWMA (RiskMetrics) VaR.
    FIX-R1: corrected com = lam / (1 - lam)  [was (1-lam)/lam — too reactive]
    For λ=0.94: effective window ≈ 32 days, matching RiskMetrics spec.
    """
    alpha_conf = 1 - confidence
    z = stats.norm.ppf(alpha_conf)
    ewm_com  = lam / (1.0 - lam)       # FIX-R1
    ewm_mean = returns.ewm(com=ewm_com).mean()
    ewm_var  = returns.ewm(com=ewm_com).var()
    sigma    = np.sqrt(ewm_var)
    return -(ewm_mean + z * sigma)


def var_cornish_fisher(returns: pd.Series, confidence: float = 0.95,
                       window: Optional[int] = 252) -> pd.Series:
    """
    Cornish-Fisher VaR — adjusts Gaussian z for skewness and excess kurtosis.
    More conservative than plain parametric on fat-tailed financial returns.
    """
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)

    def _cf_z(ret_window: pd.Series) -> float:
        s = float(ret_window.skew())
        k = float(ret_window.kurtosis())
        z_cf = (z
                + (z**2 - 1) / 6 * s
                + (z**3 - 3*z) / 24 * k
                - (2*z**3 - 5*z) / 36 * s**2)
        return z_cf

    if window:
        mu    = returns.rolling(window).mean()
        sigma = returns.rolling(window).std()
        z_cf_series = returns.rolling(window).apply(
            lambda x: _cf_z(pd.Series(x)), raw=False)
        return -(mu + z_cf_series * sigma)
    else:
        z_cf = _cf_z(returns)
        return pd.Series(-(returns.mean() + z_cf * returns.std()),
                         index=returns.index)


def var_garch(returns: pd.Series, confidence: float = 0.95) -> pd.Series:
    """Standard GARCH(1,1)-t VaR."""
    if not HAS_ARCH:
        return var_ewma(returns, confidence)
    try:
        ret_pct = returns * 100
        am = arch_model(ret_pct, vol="Garch", p=1, q=1,
                        dist="StudentsT", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        cond_vol = res.conditional_volatility / 100
        dof      = float(res.params.get("nu", 6))
        alpha    = 1 - confidence
        t_q      = stats.t.ppf(alpha, df=dof)
        mu       = returns.mean()
        var_s    = -(mu + t_q * cond_vol)
        return var_s.reindex(returns.index)
    except Exception:
        return var_ewma(returns, confidence)


def var_gjr_garch(returns: pd.Series, confidence: float = 0.95) -> pd.Series:
    """
    GJR-GARCH(1,1,1)-t VaR — asymmetric volatility model.

    Standard GARCH gives symmetric shock response. In equity markets,
    negative shocks increase volatility MORE than positive shocks
    (leverage effect). GJR-GARCH adds the asymmetry term γ:

        σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I[ε<0] + β·σ²_{t-1}

    γ > 0: bad news has α+γ impact vs α for good news.
    Empirically γ ≈ 0.1-0.2 for S&P 500 (50-100% more vol after down days).

    Impact: After a crash, GJR predicts higher vol → wider VaR bands
    → fewer VaR breaches on the days following large sell-offs.
    """
    if not HAS_ARCH:
        return var_ewma(returns, confidence)
    try:
        ret_pct = returns * 100
        am = arch_model(
            ret_pct, vol="GARCH",
            p=1, o=1, q=1,      # o=1 is the GJR asymmetry (leverage) term
            dist="StudentsT",
            rescale=False
        )
        res = am.fit(disp="off", show_warning=False)
        cond_vol = res.conditional_volatility / 100
        dof      = float(res.params.get("nu", 6))
        t_q      = stats.t.ppf(1 - confidence, df=dof)
        mu       = returns.mean()
        return -(mu + t_q * cond_vol).reindex(returns.index)
    except Exception:
        return var_garch(returns, confidence)


def regime_conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
    bear_threshold: float = -0.10,
) -> pd.Series:
    """
    Regime-Conditional VaR — estimates vol separately in bull/bear regimes.

    Single-regime VaR averages over both regimes, underestimating risk
    in bear markets (where VaR is most needed) and overestimating in bull.

    Method:
      1. Detect regime: bear = 63d rolling return < bear_threshold (-10%)
      2. Estimate vol from same-regime returns within the rolling window
      3. Apply regime-appropriate vol to compute VaR

    Result: tighter VaR in bull (less capital waste),
            wider VaR in bear (better downside protection).
    """
    rolling_63d = returns.rolling(63).sum()
    is_bear     = rolling_63d < bear_threshold
    var_series  = pd.Series(np.nan, index=returns.index)

    for i in range(window, len(returns)):
        win_ret  = returns.iloc[i - window:i]
        bear_win = is_bear.iloc[i - window:i]
        cur_bear = bool(is_bear.iloc[i]) if i < len(is_bear) else False

        bear_ret = win_ret[bear_win]
        bull_ret = win_ret[~bear_win]

        if cur_bear and len(bear_ret) >= 20:
            mu, sigma = bear_ret.mean(), bear_ret.std()
        elif not cur_bear and len(bull_ret) >= 20:
            mu, sigma = bull_ret.mean(), bull_ret.std()
        else:
            mu, sigma = win_ret.mean(), win_ret.std()

        z = stats.norm.ppf(1 - confidence)
        var_series.iloc[i] = -(mu + z * sigma)

    return var_series.dropna()


def conservative_var(var_series: pd.Series, buffer: float = 0.10) -> pd.Series:
    return var_series * (1 + buffer)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95,
                       window: Optional[int] = None) -> float:
    alpha     = 1 - confidence
    threshold = returns.quantile(alpha)
    tail      = returns[returns <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTING FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

def build_var_breaches(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    common = returns.index.intersection(var_series.index)
    r = returns.loc[common]
    v = var_series.loc[common]
    return ((-r) > v).astype(int).dropna()


def kupiec_pof_test(breaches: pd.Series, confidence: float = 0.95) -> Dict:
    """
    Kupiec (1995) POF test.
    FIX-R3: zero-breach case handled via exact binomial probability.
    """
    n        = len(breaches)
    n_breach = int(breaches.sum())
    p        = 1 - confidence

    if n == 0:
        return {"lr_stat": np.nan, "p_value": np.nan,
                "obs_rate": np.nan, "exp_rate": p,
                "n": n, "n_breaches": 0, "reject_h0": False, "test": "Kupiec POF"}

    p_hat = n_breach / n

    if n_breach == 0:
        lr      = -2 * n * np.log(1 - p)
        p_value = float(stats.chi2.sf(lr, df=1))
        return {"lr_stat": float(lr), "p_value": p_value,
                "obs_rate": 0.0, "exp_rate": p,
                "n": n, "n_breaches": 0,
                "reject_h0": p_value < 0.05, "test": "Kupiec POF"}

    try:
        lr = -2 * (
            n_breach * np.log(p / p_hat) +
            (n - n_breach) * np.log((1 - p) / (1 - p_hat))
        )
    except (ZeroDivisionError, ValueError):
        lr = np.nan

    p_value = float(stats.chi2.sf(lr, df=1)) if not np.isnan(lr) else np.nan
    return {
        "lr_stat": float(lr), "p_value": p_value,
        "obs_rate": p_hat, "exp_rate": p,
        "n": n, "n_breaches": n_breach,
        "reject_h0": (p_value < 0.05) if not np.isnan(p_value) else False,
        "test": "Kupiec POF"
    }


def christoffersen_test(breaches: pd.Series, confidence: float = 0.95) -> Dict:
    """
    Christoffersen (1998) CC test.
    FIX-R2: uses xlogy for correct 0·log(0)=0 handling.
    """
    b = breaches.values.astype(int)
    n = len(b)
    if n < 4:
        return {"lr_cc": np.nan, "p_value_cc": np.nan,
                "lr_ind": np.nan, "p_value_ind": np.nan,
                "test": "Christoffersen CC", "reject_h0_cc": False,
                "reject_h0_ind": False, "degenerate": True}

    n00 = int(np.sum((b[:-1]==0) & (b[1:]==0)))
    n01 = int(np.sum((b[:-1]==0) & (b[1:]==1)))
    n10 = int(np.sum((b[:-1]==1) & (b[1:]==0)))
    n11 = int(np.sum((b[:-1]==1) & (b[1:]==1)))

    denom01 = n00 + n01
    denom11 = n10 + n11
    pi01 = n01 / denom01 if denom01 > 0 else 0.0
    pi11 = n11 / denom11 if denom11 > 0 else 0.0
    pi   = (n01 + n11) / max(n00+n01+n10+n11, 1)

    degenerate = (denom11 == 0) or (denom01 == 0)
    if degenerate:
        warnings.warn("Christoffersen test degenerate.", RuntimeWarning, stacklevel=2)
        kupiec = kupiec_pof_test(breaches, confidence)
        return {
            "lr_cc": np.nan, "p_value_cc": np.nan,
            "lr_ind": np.nan, "p_value_ind": np.nan,
            "pi01": pi01, "pi11": pi11,
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "reject_h0_cc": False, "reject_h0_ind": False,
            "test": "Christoffersen CC", "degenerate": True
        }

    from scipy.special import xlogy
    try:
        ll_unres = (xlogy(n00, 1-pi01) + xlogy(n01, pi01) +
                    xlogy(n10, 1-pi11) + xlogy(n11, pi11))
        ll_res   = (xlogy(n00+n10, 1-pi) + xlogy(n01+n11, pi))
        lr_ind   = max(0.0, float(-2.0 * (ll_res - ll_unres)))
    except Exception:
        lr_ind = np.nan

    p_ind  = float(stats.chi2.sf(lr_ind, df=1)) if not np.isnan(lr_ind) else np.nan
    kupiec = kupiec_pof_test(breaches, confidence)
    lr_uc  = kupiec["lr_stat"]
    lr_cc  = ((lr_uc or 0.0) + (lr_ind or 0.0))
    p_cc   = float(stats.chi2.sf(lr_cc, df=2)) if not np.isnan(lr_cc) else np.nan

    return {
        "lr_cc": float(lr_cc), "p_value_cc": p_cc,
        "lr_ind": float(lr_ind) if not np.isnan(lr_ind) else np.nan,
        "p_value_ind": p_ind,
        "pi01": pi01, "pi11": pi11,
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "reject_h0_cc": (p_cc < 0.05) if not np.isnan(p_cc) else False,
        "reject_h0_ind": (p_ind < 0.05) if not np.isnan(p_ind) else False,
        "test": "Christoffersen CC", "degenerate": False
    }


def unconditional_coverage_test(breaches: pd.Series,
                                 confidence: float = 0.95) -> Dict:
    n = len(breaches)
    k = int(breaches.sum())
    p_expected = 1 - confidence
    if n == 0:
        return {"p_value": np.nan, "obs_rate": np.nan, "test": "Binomial UC"}
    result = stats.binomtest(k, n, p=p_expected, alternative="two-sided")
    return {
        "p_value":    float(result.pvalue),
        "obs_rate":   k / n,
        "exp_rate":   p_expected,
        "n":          n,
        "n_breaches": k,
        "reject_h0":  result.pvalue < 0.05,
        "test":       "Binomial Unconditional Coverage"
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL BACKTEST REPORT
# ─────────────────────────────────────────────────────────────────────────────

def full_var_backtest(
    prices: pd.Series,
    confidence: float = 0.95,
    var_window: int = 252,
    var_method: str = "historical",
    conservative_buffer: float = 0.0,
) -> Dict:
    """Full VaR backtest — now supports gjr_garch and regime_conditional."""
    returns = compute_returns(prices)
    if len(returns) < var_window + 10:
        return {"error": f"Need >= {var_window + 10} observations"}

    method_map = {
        "historical":         lambda: var_historical(returns, confidence, window=var_window),
        "parametric":         lambda: var_parametric(returns, confidence, window=var_window),
        "ewma":               lambda: var_ewma(returns, confidence),
        "garch":              lambda: var_garch(returns, confidence),
        "gjr_garch":          lambda: var_gjr_garch(returns, confidence),
        "cornish_fisher":     lambda: var_cornish_fisher(returns, confidence, window=var_window),
        "regime_conditional": lambda: regime_conditional_var(returns, confidence, var_window),
    }

    var_fn     = method_map.get(var_method, method_map["historical"])
    var_series = var_fn()

    if conservative_buffer > 0:
        var_series = conservative_var(var_series, buffer=conservative_buffer)

    var_series = var_series.dropna()
    common     = returns.index.intersection(var_series.index)
    ret_bt     = returns.loc[common]
    var_bt     = var_series.loc[common]

    breaches  = build_var_breaches(ret_bt, var_bt)
    kupiec    = kupiec_pof_test(breaches, confidence)
    christoff = christoffersen_test(breaches, confidence)
    binom_uc  = unconditional_coverage_test(breaches, confidence)
    es        = expected_shortfall(ret_bt, confidence)

    return {
        "returns":         ret_bt,
        "var_series":      var_bt,
        "breaches":        breaches,
        "es":              es,
        "kupiec":          kupiec,
        "christoffersen":  christoff,
        "binomial_uc":     binom_uc,
        "confidence":      confidence,
        "var_method":      var_method,
        "conservative_buffer": conservative_buffer,
        "n_obs":           len(breaches),
        "n_breaches":      int(breaches.sum()),
        "obs_rate":        float(breaches.mean()),
        "exp_rate":        1 - confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL VAR — UPDATED to include new methods
# ─────────────────────────────────────────────────────────────────────────────

ALL_VAR_METHODS = [
    "historical", "parametric", "ewma",
    "garch", "gjr_garch",
    "cornish_fisher", "regime_conditional",
]

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
    "parametric":         "Parametric (Gaussian)",
    "ewma":               "EWMA (RiskMetrics)",
    "garch":              "GARCH(1,1)-t",
    "gjr_garch":          "GJR-GARCH(1,1,1)",
    "cornish_fisher":     "Cornish-Fisher",
    "regime_conditional": "Regime-Conditional",
}


def _score_var_result(result: Dict, confidence: float) -> float:
    if "error" in result:
        return -1.0
    exp_rate = 1 - confidence
    obs_rate = result.get("obs_rate", exp_rate)
    kup_p    = result.get("kupiec", {}).get("p_value", 0.0) or 0.0
    chr_p    = result.get("christoffersen", {}).get("p_value_cc", 0.0) or 0.0
    proximity = max(0.0, 1.0 - abs(obs_rate - exp_rate) / (exp_rate + 1e-9))
    return float(0.40 * min(kup_p, 1.0) + 0.40 * proximity + 0.20 * min(chr_p, 1.0))


def run_all_var_methods_parallel(
    prices: pd.Series,
    confidence: float = 0.95,
    var_window: int = 252,
    conservative_buffer: float = 0.0,
    max_workers: int = 4,
) -> Dict[str, Dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_one(method: str) -> tuple:
        try:
            result = full_var_backtest(
                prices, confidence=confidence,
                var_window=var_window, var_method=method,
                conservative_buffer=conservative_buffer,
            )
        except Exception as e:
            result = {"error": str(e)}
        result["score"]      = _score_var_result(result, confidence)
        result["var_method"] = method
        return method, result

    # Include new methods; skip GARCH variants if arch not installed
    if HAS_ARCH:
        methods = ALL_VAR_METHODS
    else:
        methods = [m for m in ALL_VAR_METHODS
                   if m not in ("garch", "gjr_garch")]

    results = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(methods))) as pool:
        futures = {pool.submit(_run_one, m): m for m in methods}
        for fut in as_completed(futures):
            method, result = fut.result()
            results[method] = result

    return results


def select_best_var_method(all_results: Dict[str, Dict]) -> str:
    best_method, best_score = "historical", -999.0
    for method, result in all_results.items():
        score = result.get("score", -1.0)
        if score > best_score:
            best_score, best_method = score, method
    return best_method


def summarize_wf_results(wf_results: list) -> Dict:
    if not wf_results:
        return {}
    df = pd.DataFrame(wf_results)
    out = {}
    for col in ["rmse", "mae", "mape", "da"]:
        if col in df.columns:
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_std"]  = float(df[col].std())
    return out