"""
Stochastic Projection Engine (SPE) — for Fundamental Event Scanner Pro
=======================================================================
Re-imported from Stochastic Scanner Pro project. Used here for the
Monte Carlo + GARCH integration with fundamental event signals.

Provides:
  - Monte Carlo with t-student innovations
  - GARCH(1,1) volatility forecasting
  - EWMA fallback
  - Multi-horizon projections
  - VaR / CVaR / Confidence Intervals
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


class StochasticProjector:
    MIN_SAMPLES_FOR_T_FIT = 60
    EWMA_DECAY = 0.94
    DEFAULT_HORIZON_DAYS = 21
    N_SIMULATIONS_DEFAULT = 10_000

    def __init__(self, hist, indicators=None, fundamentals=None):
        self.hist = hist
        self.ind = indicators or {}
        self.fund = fundamentals or {}
        self.close = hist["Close"].values
        self.returns = np.diff(np.log(self.close))
        self.returns = self.returns[np.isfinite(self.returns)]

    def _ewma_volatility(self, decay=None):
        if decay is None: decay = self.EWMA_DECAY
        if len(self.returns) < 2: return 0.02
        n = len(self.returns)
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        mean = np.sum(weights * self.returns)
        var = np.sum(weights * (self.returns - mean) ** 2)
        return np.sqrt(var)

    def _fit_garch_11(self):
        r = self.returns
        if len(r) < 100: return None
        r_demeaned = r - np.mean(r)
        r2 = r_demeaned ** 2
        unconditional_var = np.var(r)
        if unconditional_var <= 0: return None

        def neg_log_likelihood(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
                return 1e10
            n = len(r_demeaned)
            sigma2 = np.empty(n)
            sigma2[0] = unconditional_var
            for t in range(1, n):
                sigma2[t] = omega + alpha * r2[t-1] + beta * sigma2[t-1]
                if sigma2[t] <= 0: return 1e10
            if not np.all(np.isfinite(sigma2)): return 1e10
            return 0.5 * np.sum(np.log(2 * np.pi * sigma2) + r2 / sigma2)

        x0 = [unconditional_var * 0.1, 0.1, 0.85]
        bounds = [(1e-10, unconditional_var * 2), (1e-4, 0.5), (1e-4, 0.99)]
        try:
            result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                              bounds=bounds, options={"maxiter": 50})
            if not result.success: return None
            omega, alpha, beta = result.x
            n = len(r_demeaned)
            sigma2 = np.empty(n)
            sigma2[0] = unconditional_var
            for t in range(1, n):
                sigma2[t] = omega + alpha * r2[t-1] + beta * sigma2[t-1]
            return (float(omega), float(alpha), float(beta),
                    float(np.sqrt(max(sigma2[-1], 1e-10))))
        except Exception:
            return None

    def _garch_forecast(self, horizon):
        fit = self._fit_garch_11()
        if fit is None: return None
        omega, alpha, beta, last_sigma = fit
        last_var = last_sigma ** 2
        persistence = alpha + beta
        if persistence >= 1.0: return np.full(horizon, last_sigma)
        long_run_var = omega / (1 - persistence)
        sigmas = np.zeros(horizon)
        var_t = last_var
        for t in range(horizon):
            var_t = long_run_var + persistence * (var_t - long_run_var)
            sigmas[t] = np.sqrt(max(var_t, 1e-10))
        return sigmas

    def _fit_return_distribution(self):
        n = len(self.returns)
        if n < self.MIN_SAMPLES_FOR_T_FIT:
            return ("normal", (np.mean(self.returns),
                               max(0.001, np.std(self.returns))))
        try:
            df, loc, scale = stats.t.fit(self.returns)
            df = np.clip(df, 3.0, 30.0)
            scale = np.clip(scale, 0.0005, 0.15)
            return ("t", (df, loc, scale))
        except Exception:
            return ("normal", (np.mean(self.returns),
                               max(0.001, np.std(self.returns))))

    def monte_carlo(self, horizon=None, n_sims=None):
        horizon = horizon or self.DEFAULT_HORIZON_DAYS
        n_sims = n_sims or self.N_SIMULATIONS_DEFAULT
        if len(self.returns) < 20: return self._fallback(horizon)

        dist_type, params = self._fit_return_distribution()
        garch_sigmas = self._garch_forecast(horizon)
        if garch_sigmas is not None:
            vol_method = "garch(1,1)"
            sigma_path = garch_sigmas
            baseline_vol = float(np.mean(garch_sigmas))
        else:
            vol_method = "ewma"
            ewma_vol = self._ewma_volatility()
            sigma_path = np.full(horizon, ewma_vol)
            baseline_vol = ewma_vol

        if dist_type == "t":
            df, loc, scale = params
            raw_t = stats.t.rvs(df, size=(n_sims, horizon))
            standard_rvs = raw_t / np.sqrt(df / (df - 2)) if df > 2 else \
                           np.random.normal(0, 1, size=(n_sims, horizon))
            mean_return = loc
        else:
            loc, sigma = params
            standard_rvs = np.random.normal(0, 1, size=(n_sims, horizon))
            mean_return = loc

        last_price = self.close[-1]
        increments = mean_return + sigma_path[np.newaxis, :] * standard_rvs
        cum_returns = np.cumsum(increments, axis=1)
        terminal = last_price * np.exp(cum_returns[:, -1])

        var_95 = float(np.percentile(terminal, 5))
        tail_mask = terminal <= var_95
        cvar_95 = float(np.mean(terminal[tail_mask])) if np.any(tail_mask) else var_95

        return {
            "method": dist_type, "vol_method": vol_method,
            "horizon_days": horizon, "n_simulations": n_sims,
            "current_price": round(last_price, 4),
            "expected": round(float(np.mean(terminal)), 4),
            "median": round(float(np.median(terminal)), 4),
            "ci_68_low": round(float(np.percentile(terminal, 16)), 4),
            "ci_68_high": round(float(np.percentile(terminal, 84)), 4),
            "ci_95_low": round(float(np.percentile(terminal, 2.5)), 4),
            "ci_95_high": round(float(np.percentile(terminal, 97.5)), 4),
            "var_95": round(var_95, 4), "cvar_95": round(cvar_95, 4),
            "prob_up": round(float(np.mean(terminal > last_price)) * 100, 2),
            "prob_tp3": round(float(np.mean(terminal > last_price * 1.03)) * 100, 2),
            "prob_tp5": round(float(np.mean(terminal > last_price * 1.05)) * 100, 2),
            "prob_tp10": round(float(np.mean(terminal > last_price * 1.10)) * 100, 2),
            "prob_drop_5": round(float(np.mean(terminal < last_price * 0.95)) * 100, 2),
            "volatility_annualized": round(baseline_vol * np.sqrt(252) * 100, 2),
            "skewness": round(float(stats.skew(terminal)), 3),
            "kurtosis": round(float(stats.kurtosis(terminal)), 3),
            "reliable": len(self.returns) >= self.MIN_SAMPLES_FOR_T_FIT,
        }

    def _fallback(self, horizon):
        last = self.close[-1] if len(self.close) else 100.0
        return {"method": "fallback", "vol_method": "fallback",
                "horizon_days": horizon, "n_simulations": 0,
                "current_price": round(last, 4), "expected": round(last * 1.02, 4),
                "median": round(last * 1.02, 4),
                "ci_68_low": round(last * 0.95, 4), "ci_68_high": round(last * 1.05, 4),
                "ci_95_low": round(last * 0.90, 4), "ci_95_high": round(last * 1.10, 4),
                "var_95": round(last * 0.90, 4), "cvar_95": round(last * 0.85, 4),
                "prob_up": 55.0, "prob_tp3": 40.0, "prob_tp5": 30.0,
                "prob_tp10": 15.0, "prob_drop_5": 30.0,
                "volatility_annualized": 25.0, "skewness": 0.0, "kurtosis": 0.0,
                "reliable": False}
