"""
Fundamental Event Scanner Pro — Monte Carlo Integration
=========================================================
Implements 4 patterns for integrating Stochastic Projection Engine (SPE)
with Quality Score and Event Study results.

PATTERN 1 — INDEPENDENT (side-by-side)
PATTERN 2 — EVENT-CONDITIONAL MC (most rigorous)
PATTERN 3 — QUALITY-ADJUSTED DRIFT (Fama-French inspired)
PATTERN 4 — MASTER FESP SCORE (heuristic ranking)

USAGE:
    from mc_integration import full_integrated_analysis
    result = full_integrated_analysis(price_hist, fundamentals, earnings_hist, sector)
    # result has: spe_independent, spe_event_conditional, fesp_score, etc.

HONEST DISCLAIMERS:
  - All 4 patterns are MODELS, not predictions of the future.
  - Pattern 2 assumes historical reactions repeat — may fail in regime changes.
  - Pattern 3 is theoretically grounded but easily overfit on short samples.
  - Pattern 4 is a ranking heuristic, NOT a probability.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stochastic_projector import StochasticProjector
from quality_score import quality_score, fundamental_zscores
from event_study import full_event_study


# ═══════════════════════════════════════════════════════════════════
# PATTERN 1 — INDEPENDENT (side-by-side)
# ═══════════════════════════════════════════════════════════════════
def pattern_1_independent(price_hist, fundamentals, sector, earnings_hist):
    """
    Run SPE and FEE separately, return both for side-by-side display.
    No coupling between the two engines.
    
    Use case: dashboard wants to show "fundamental score: 72/100" 
    AND "MC prob_up: 62%" as independent signals.
    """
    # Run SPE
    sp = StochasticProjector(price_hist)
    spe = sp.monte_carlo(horizon=21)
    
    # Run Quality Score (with event study for the stability/momentum components)
    es = full_event_study(price_hist, earnings_hist) if earnings_hist is not None else None
    qs_score, qs_details = quality_score(fundamentals, sector, event_study_results=es)
    
    return {
        "pattern": "independent",
        "spe": spe,
        "quality_score": qs_score,
        "quality_breakdown": qs_details,
        "event_study": es,
    }


# ═══════════════════════════════════════════════════════════════════
# PATTERN 2 — EVENT-CONDITIONAL MC
# ═══════════════════════════════════════════════════════════════════
def pattern_2_event_conditional(price_hist, earnings_hist, days_to_next_earnings=None,
                                  horizon=21, n_sims=10000):
    """
    When earnings is near, run MC conditional on BEAT vs MISS scenarios,
    using historical reaction patterns.
    
    Logic:
      1. Run event study to get beat_rate, beat_reaction.avg_t5, miss_reaction.avg_t5
      2. Run TWO Monte Carlo scenarios:
         - "If BEAT": drift = avg historical reaction after beats
         - "If MISS": drift = avg historical reaction after misses
      3. Weight terminal distributions by beat_rate
      4. Aggregate stats from the bimodal distribution
    
    Args:
        price_hist: DataFrame with Close column
        earnings_hist: DataFrame with date, eps_actual, eps_estimate
        days_to_next_earnings: int, if known. If None, computed from last earnings + 90.
        horizon: trading days to project (default 21)
        n_sims: simulations per scenario (default 10000, so 20000 total)
    
    Returns:
        dict with both scenarios + weighted combination
    """
    # Run event study
    es = full_event_study(price_hist, earnings_hist) if earnings_hist is not None else None
    
    if es is None or es.get("n_events", 0) < 4:
        # Not enough history — fall back to standard MC
        sp = StochasticProjector(price_hist)
        return {
            "pattern": "event_conditional",
            "fallback_reason": "Insufficient earnings history (<4 events)",
            "standard_mc": sp.monte_carlo(horizon=horizon, n_sims=n_sims),
        }
    
    # Determine days to next earnings
    if days_to_next_earnings is None:
        if "date" in earnings_hist.columns and len(earnings_hist) > 0:
            last_earnings = pd.to_datetime(earnings_hist["date"].iloc[-1])
            today = pd.Timestamp.now()
            try:
                if last_earnings.tz is not None:
                    last_earnings = last_earnings.tz_localize(None)
            except Exception:
                pass
            days_since = (today - last_earnings).days
            days_to_next_earnings = max(1, 90 - days_since)
        else:
            days_to_next_earnings = 30
    
    # Decide if earnings is "near enough" to apply conditional logic
    if days_to_next_earnings > horizon * 2:
        # Earnings far away — earnings-conditional logic less relevant
        sp = StochasticProjector(price_hist)
        return {
            "pattern": "event_conditional",
            "days_to_next_earnings": days_to_next_earnings,
            "note": f"Earnings {days_to_next_earnings}d away — outside main horizon, using standard MC",
            "standard_mc": sp.monte_carlo(horizon=horizon, n_sims=n_sims),
        }
    
    # Extract historical reaction parameters
    beat_rate = es.get("classification", {}).get("beat_rate", 50) / 100
    
    if "beat_reaction" in es:
        beat_drift_pct = es["beat_reaction"].get("avg_t5", 1.0) / 100
        beat_window_std = es["windows"].get("t5", {}).get("std", 3) / 100
    else:
        beat_drift_pct = 0.01
        beat_window_std = 0.03
    
    if "miss_reaction" in es:
        miss_drift_pct = es["miss_reaction"].get("avg_t5", -2.0) / 100
        miss_window_std = es["windows"].get("t5", {}).get("std", 3) / 100
    else:
        miss_drift_pct = -0.02
        miss_window_std = 0.03
    
    # Run baseline MC to get returns characteristics
    sp = StochasticProjector(price_hist)
    baseline_mc = sp.monte_carlo(horizon=horizon, n_sims=n_sims)
    
    last_price = float(price_hist["Close"].iloc[-1])
    base_vol_daily = baseline_mc["volatility_annualized"] / 100 / np.sqrt(252)
    
    # Scenario A: BEAT — apply beat reaction over event window, normal vol elsewhere
    np.random.seed(42)
    n_event_days = min(5, horizon)  # reaction concentrated in first 5 days
    n_post_days = max(0, horizon - n_event_days)
    
    # Event-window returns: shifted distribution
    beat_event_returns = np.random.normal(beat_drift_pct / n_event_days,
                                           beat_window_std,
                                           size=(n_sims, n_event_days))
    # Post-event returns: normal vol, slight residual drift (PEAD)
    pead_drift = beat_drift_pct * 0.2 / max(n_post_days, 1) if n_post_days > 0 else 0
    if n_post_days > 0:
        beat_post_returns = np.random.normal(pead_drift, base_vol_daily,
                                              size=(n_sims, n_post_days))
        beat_paths = np.concatenate([beat_event_returns, beat_post_returns], axis=1)
    else:
        beat_paths = beat_event_returns
    
    beat_terminal = last_price * np.exp(np.cumsum(beat_paths, axis=1)[:, -1])
    
    # Scenario B: MISS
    np.random.seed(43)
    miss_event_returns = np.random.normal(miss_drift_pct / n_event_days,
                                           miss_window_std,
                                           size=(n_sims, n_event_days))
    miss_pead = miss_drift_pct * 0.3 / max(n_post_days, 1) if n_post_days > 0 else 0
    if n_post_days > 0:
        miss_post_returns = np.random.normal(miss_pead, base_vol_daily,
                                              size=(n_sims, n_post_days))
        miss_paths = np.concatenate([miss_event_returns, miss_post_returns], axis=1)
    else:
        miss_paths = miss_event_returns
    
    miss_terminal = last_price * np.exp(np.cumsum(miss_paths, axis=1)[:, -1])
    
    # Weighted combination
    weighted_terminal = np.concatenate([
        beat_terminal[:int(n_sims * beat_rate)],
        miss_terminal[:int(n_sims * (1 - beat_rate))],
    ])
    np.random.shuffle(weighted_terminal)
    
    def stats_dict(terminal, label):
        return {
            "scenario": label,
            "n_sims": len(terminal),
            "expected": round(float(np.mean(terminal)), 4),
            "median": round(float(np.median(terminal)), 4),
            "ci_68_low": round(float(np.percentile(terminal, 16)), 4),
            "ci_68_high": round(float(np.percentile(terminal, 84)), 4),
            "ci_95_low": round(float(np.percentile(terminal, 2.5)), 4),
            "ci_95_high": round(float(np.percentile(terminal, 97.5)), 4),
            "var_95": round(float(np.percentile(terminal, 5)), 4),
            "prob_up": round(float(np.mean(terminal > last_price)) * 100, 2),
            "prob_tp5": round(float(np.mean(terminal > last_price * 1.05)) * 100, 2),
            "prob_drop_5": round(float(np.mean(terminal < last_price * 0.95)) * 100, 2),
        }
    
    return {
        "pattern": "event_conditional",
        "days_to_next_earnings": days_to_next_earnings,
        "horizon_days": horizon,
        "beat_rate_historical": round(beat_rate * 100, 2),
        "current_price": round(last_price, 4),
        "scenario_beat": stats_dict(beat_terminal, "if_BEAT"),
        "scenario_miss": stats_dict(miss_terminal, "if_MISS"),
        "weighted": stats_dict(weighted_terminal, "weighted_by_beat_rate"),
        "baseline_unconditional": baseline_mc,
        "interpretation": _interpret_event_mc(beat_terminal, miss_terminal, last_price, beat_rate),
    }


def _interpret_event_mc(beat_terminal, miss_terminal, last_price, beat_rate):
    """Generate human-readable interpretation."""
    if last_price <= 0:
        return {"bias": "n/a", "weighted_upside_pct": 0,
                "asymmetry_pct": 0, "risk_reward": "N/A — zero price"}
    expected_beat = float(np.mean(beat_terminal))
    expected_miss = float(np.mean(miss_terminal))
    upside_beat = (expected_beat / last_price - 1) * 100
    downside_miss = (expected_miss / last_price - 1) * 100
    asymmetry = upside_beat - downside_miss
    
    weighted_expected = expected_beat * beat_rate + expected_miss * (1 - beat_rate)
    weighted_upside = (weighted_expected / last_price - 1) * 100
    
    if abs(weighted_upside) < 1:
        bias = "neutral"
    elif weighted_upside > 1:
        bias = "alcista"
    else:
        bias = "bajista"
    
    if asymmetry > 8:
        risk_reward = "Risk/reward FAVORABLE — beats premian más que misses castigan"
    elif asymmetry > 3:
        risk_reward = "Risk/reward moderadamente positivo"
    elif asymmetry > -3:
        risk_reward = "Risk/reward simétrico"
    else:
        risk_reward = "Risk/reward DESFAVORABLE — misses castigan más que beats premian"
    
    return {
        "bias": bias,
        "weighted_upside_pct": round(weighted_upside, 2),
        "asymmetry_pct": round(asymmetry, 2),
        "risk_reward": risk_reward,
    }


# ═══════════════════════════════════════════════════════════════════
# PATTERN 3 — QUALITY-ADJUSTED DRIFT
# ═══════════════════════════════════════════════════════════════════
def pattern_3_quality_adjusted(price_hist, fundamentals, sector, horizon=63):
    """
    Adjust the MC drift based on Quality Score (Fama-French quality factor).
    
    Theoretical basis: high-quality firms historically outperform by ~2-4%/yr.
    
    Logic:
      - Compute Quality Score (0-100)
      - Map to drift adjustment: QS=50 → +0%, QS=100 → +4%/yr, QS=0 → -2%/yr
      - Adjust MC daily drift by adjustment / 252
    
    WARNING: This is easy to overfit. Use longer horizons (3+ months).
    Short-horizon equity returns are dominated by noise, not quality.
    """
    qs, _ = quality_score(fundamentals, sector)
    
    # Map QS to annual drift adjustment
    # QS 50 → 0%, QS 75 → +2%, QS 100 → +4%
    # QS 25 → -1%, QS 0 → -2%
    if qs >= 50:
        annual_adj = (qs - 50) / 50 * 0.04  # max +4% at QS=100
    else:
        annual_adj = (qs - 50) / 50 * 0.02  # max -2% at QS=0
    
    daily_adj = annual_adj / 252
    
    # Run baseline MC and apply adjustment
    sp = StochasticProjector(price_hist)
    baseline = sp.monte_carlo(horizon=horizon)
    
    # Re-simulate with adjusted drift
    n_sims = 10000
    last_price = float(price_hist["Close"].iloc[-1])
    base_vol = baseline["volatility_annualized"] / 100 / np.sqrt(252)
    
    np.random.seed(44)
    paths = np.random.normal(daily_adj, base_vol, size=(n_sims, horizon))
    terminal = last_price * np.exp(np.cumsum(paths, axis=1)[:, -1])
    
    return {
        "pattern": "quality_adjusted",
        "horizon_days": horizon,
        "quality_score": round(qs, 1),
        "annual_drift_adjustment_pct": round(annual_adj * 100, 2),
        "current_price": round(last_price, 4),
        "expected": round(float(np.mean(terminal)), 4),
        "ci_68_low": round(float(np.percentile(terminal, 16)), 4),
        "ci_68_high": round(float(np.percentile(terminal, 84)), 4),
        "var_95": round(float(np.percentile(terminal, 5)), 4),
        "prob_up": round(float(np.mean(terminal > last_price)) * 100, 2),
        "baseline_prob_up": baseline["prob_up"],
        "improvement_vs_baseline_pp": round(
            float(np.mean(terminal > last_price)) * 100 - baseline["prob_up"], 2),
        "warning": "Quality drift adjustment is theoretical. May overfit on short samples.",
    }


# ═══════════════════════════════════════════════════════════════════
# PATTERN 4 — MASTER FESP SCORE
# ═══════════════════════════════════════════════════════════════════
def pattern_4_master_score(price_hist, fundamentals, sector, earnings_hist):
    """
    Heuristic master score combining all signals.
    
    Components (each 0-100):
      - Quality Score (40% weight)
      - Earnings Momentum (20% weight)
      - SPE prob_up scaled (40% weight)
    
    NOT A PROBABILITY. It is a RANKING SCORE for comparing tickers.
    """
    # Run all three engines
    es = full_event_study(price_hist, earnings_hist) if earnings_hist is not None else None
    qs, qs_det = quality_score(fundamentals, sector, event_study_results=es)
    
    sp = StochasticProjector(price_hist)
    spe = sp.monte_carlo(horizon=63)  # 3-month horizon for ranking
    spe_score = spe["prob_up"]  # 0-100
    
    # Earnings momentum: derive from event study
    if es and "beat_reaction" in es:
        em_score = (es["classification"]["beat_rate"] +
                    max(0, min(50, 50 + es["beat_reaction"]["avg_t5"] * 5))) / 1.5
        em_score = max(0, min(100, em_score))
    elif es and "windows" in es and "t21" in es["windows"]:
        # Fallback: use median t21 reaction as momentum proxy
        em_score = max(0, min(100, 50 + es["windows"]["t21"]["mean"] * 3))
    else:
        em_score = 50
    
    # Weighted combination
    fesp = qs * 0.40 + em_score * 0.20 + spe_score * 0.40
    
    # Grade
    if fesp >= 75:
        grade, label = "A", "TOP — combina calidad fundamental + momentum + probabilidad alcista"
    elif fesp >= 65:
        grade, label = "B+", "FUERTE — al menos 2 de 3 componentes son sólidos"
    elif fesp >= 55:
        grade, label = "B", "MEDIO — equilibrado pero sin destacarse"
    elif fesp >= 45:
        grade, label = "C", "DÉBIL — varios componentes en zona neutra o baja"
    else:
        grade, label = "D", "EVITAR — múltiples señales negativas"
    
    return {
        "pattern": "master_score",
        "fesp_score": round(fesp, 1),
        "grade": grade,
        "interpretation": label,
        "components": {
            "quality_score": round(qs, 1),
            "earnings_momentum": round(em_score, 1),
            "spe_prob_up": round(spe_score, 2),
        },
        "weights": {"quality": 0.40, "earnings_momentum": 0.20, "spe": 0.40},
        "warning": "FESP Score es un ranking heurístico, NO una probabilidad. "
                   "Útil para comparar tickers, no para apostar tamaño basado en él.",
    }


# ═══════════════════════════════════════════════════════════════════
# MASTER ENTRY POINT — runs all 4 patterns
# ═══════════════════════════════════════════════════════════════════
def full_integrated_analysis(price_hist, fundamentals, sector="Default",
                                earnings_hist=None, days_to_next_earnings=None):
    """
    Run all 4 integration patterns and return comprehensive analysis.
    
    Returns dict with:
      - p1_independent: side-by-side SPE + QS
      - p2_event_conditional: bimodal MC for upcoming earnings
      - p3_quality_adjusted: drift-adjusted MC for medium horizon
      - p4_master_score: heuristic FESP score
      - summary: key takeaways
    """
    p1 = pattern_1_independent(price_hist, fundamentals, sector, earnings_hist)
    p2 = pattern_2_event_conditional(price_hist, earnings_hist, days_to_next_earnings)
    p3 = pattern_3_quality_adjusted(price_hist, fundamentals, sector, horizon=63)
    p4 = pattern_4_master_score(price_hist, fundamentals, sector, earnings_hist)
    
    return {
        "p1_independent": p1,
        "p2_event_conditional": p2,
        "p3_quality_adjusted": p3,
        "p4_master_score": p4,
        "summary": {
            "fesp_score": p4["fesp_score"],
            "grade": p4["grade"],
            "quality_score": p1["quality_score"],
            "spe_prob_up_21d": p1["spe"]["prob_up"],
            "spe_prob_up_63d": p3["baseline_prob_up"],
            "earnings_in_days": p2.get("days_to_next_earnings"),
            "interpretation": p4["interpretation"],
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("MC INTEGRATION — Self Test (synthetic data)")
    print("=" * 70)
    
    # Generate synthetic price + earnings history
    np.random.seed(42)
    n_days = 252 * 3  # 3 years
    returns = np.random.normal(0.0005, 0.015, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range(end="2026-04-01", periods=n_days, freq="B")
    price_df = pd.DataFrame({"Close": prices, "High": prices * 1.01,
                              "Low": prices * 0.99,
                              "Volume": np.random.randint(1e6, 5e6, n_days)},
                             index=dates)
    
    # Earnings every ~63 trading days
    earnings_dates = [dates[63 * (i + 1) - 1] for i in range(10)]
    earnings_data = []
    for d in earnings_dates:
        is_beat = np.random.random() < 0.6
        actual = np.random.uniform(1.5, 3.0)
        estimate = actual * (np.random.uniform(0.92, 0.99) if is_beat
                              else np.random.uniform(1.01, 1.08))
        earnings_data.append({"date": d, "eps_actual": actual,
                              "eps_estimate": estimate})
    earnings_df = pd.DataFrame(earnings_data)
    
    # Fundamentals
    fundamentals = {
        "pe": 22, "forward_pe": 20, "peg": 1.4, "pb": 4, "ps": 4,
        "roe": 28, "roa": 12, "net_margin": 22, "operating_margin": 28,
        "eps_growth_5y": 14, "eps_growth_q": 12, "revenue_growth": 8,
        "debt_equity": 0.5, "current_ratio": 1.4,
    }
    
    # Run integrated analysis
    result = full_integrated_analysis(price_df, fundamentals, "Technology",
                                        earnings_df, days_to_next_earnings=15)
    
    # Display results
    print("\n── PATTERN 1: INDEPENDENT ──")
    p1 = result["p1_independent"]
    print(f"  Quality Score: {p1['quality_score']:.1f} ({p1['quality_breakdown']['grade']})")
    print(f"  SPE prob_up (21d): {p1['spe']['prob_up']:.1f}%")
    print(f"  SPE expected: ${p1['spe']['expected']:.2f}")
    
    print("\n── PATTERN 2: EVENT-CONDITIONAL MC ──")
    p2 = result["p2_event_conditional"]
    if "scenario_beat" in p2:
        print(f"  Days to next earnings: {p2['days_to_next_earnings']}")
        print(f"  Historical beat rate: {p2['beat_rate_historical']:.1f}%")
        print(f"  IF BEAT  → expected ${p2['scenario_beat']['expected']:.2f}, "
              f"prob_up={p2['scenario_beat']['prob_up']:.0f}%")
        print(f"  IF MISS  → expected ${p2['scenario_miss']['expected']:.2f}, "
              f"prob_up={p2['scenario_miss']['prob_up']:.0f}%")
        print(f"  WEIGHTED → expected ${p2['weighted']['expected']:.2f}, "
              f"prob_up={p2['weighted']['prob_up']:.0f}%")
        print(f"  Bias: {p2['interpretation']['bias']}")
        print(f"  R/R:  {p2['interpretation']['risk_reward']}")
    
    print("\n── PATTERN 3: QUALITY-ADJUSTED DRIFT ──")
    p3 = result["p3_quality_adjusted"]
    print(f"  QS: {p3['quality_score']:.1f} → drift adj: {p3['annual_drift_adjustment_pct']:+.2f}%/yr")
    print(f"  Adjusted prob_up (63d): {p3['prob_up']:.1f}%")
    print(f"  vs baseline: {p3['improvement_vs_baseline_pp']:+.1f}pp")
    
    print("\n── PATTERN 4: MASTER FESP SCORE ──")
    p4 = result["p4_master_score"]
    print(f"  FESP Score: {p4['fesp_score']:.1f} ({p4['grade']})")
    print(f"  Components:")
    for k, v in p4["components"].items():
        print(f"    {k}: {v:.1f}")
    print(f"  → {p4['interpretation']}")
    
    print("\n── SUMMARY ──")
    s = result["summary"]
    print(f"  Final ranking score: {s['fesp_score']:.1f} ({s['grade']})")
    print(f"  Quality: {s['quality_score']:.1f}/100")
    print(f"  SPE 21d prob_up: {s['spe_prob_up_21d']:.0f}%")
    print(f"  SPE 63d prob_up: {s['spe_prob_up_63d']:.0f}%")
    print(f"\n  {s['interpretation']}")
