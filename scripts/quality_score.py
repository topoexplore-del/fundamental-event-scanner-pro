"""
Fundamental Event Scanner Pro — Quality Score Engine
=====================================================
Multifactor scoring based on fundamentals + earnings quality.

Score breakdown (0-100):
  Valuation (25 pts)         — P/E, PEG, P/B vs sector benchmarks
  Profitability (25 pts)     — ROE, ROA, net margin
  Growth (25 pts)            — EPS growth, revenue growth
  Stability (15 pts)         — Earnings consistency, debt level
  Earnings momentum (10 pts) — Recent surprise patterns

DESIGN:
  - All sub-scores are bounded [0, max_pts]
  - Missing data → that component scores 50% of max (neutral, not punitive)
  - Sector-aware benchmarks (Tech tolerates higher P/E than Banking)
  - Returns dict with full breakdown for transparency

NOT GUARANTEED:
  Past quality does not predict future returns. Even high-quality firms
  underperform during regime shifts. Use this as ONE input among many.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ─── Sector-specific benchmarks ───
# Loose ranges tuned to major-market historical norms.
SECTOR_PE_BANDS = {
    "Technology":            (15, 25, 45),    # bargain, fair, expensive
    "Communication Services":(12, 20, 35),
    "Consumer Cyclical":     (12, 18, 28),
    "Consumer Defensive":    (15, 22, 30),
    "Energy":                (8,  14, 22),
    "Financial Services":    (8,  13, 18),
    "Healthcare":            (15, 22, 35),
    "Industrials":           (12, 18, 28),
    "Real Estate":           (15, 25, 45),
    "Basic Materials":       (8,  15, 25),
    "Utilities":             (14, 18, 25),
    "Default":               (12, 18, 30),
}

SECTOR_ROE_TARGETS = {
    "Technology":            (8, 18, 30),     # weak / good / excellent
    "Financial Services":    (5, 12, 18),
    "Consumer Defensive":    (10, 18, 28),
    "Healthcare":            (8, 15, 25),
    "Energy":                (5, 12, 22),
    "Utilities":             (5, 10, 14),
    "Real Estate":           (4,  8, 14),
    "Default":               (6, 14, 22),
}

SECTOR_NET_MARGIN_TARGETS = {
    "Technology":            (10, 18, 30),
    "Financial Services":    (15, 22, 30),
    "Consumer Cyclical":     (3,   7, 12),
    "Consumer Defensive":    (5,   9, 15),
    "Energy":                (5,  10, 18),
    "Healthcare":            (8,  15, 25),
    "Utilities":             (8,  12, 18),
    "Default":               (5,  10, 18),
}


def _bands(value, low, mid, high, max_score):
    """Score a value against (low, mid, high) bands. low=worst, high=best.
    Returns score in [0, max_score].
    Adjust direction with reverse_for_lower_is_better flag externally."""
    if value is None or pd.isna(value):
        return max_score * 0.5  # neutral if missing
    if value <= low:
        return 0.0
    if value >= high:
        return float(max_score)
    if value <= mid:
        # Linear between low (0) and mid (0.5*max)
        return max_score * 0.5 * (value - low) / (mid - low)
    # Linear between mid (0.5*max) and high (max)
    return max_score * (0.5 + 0.5 * (value - mid) / (high - mid))


def _bands_lower_better(value, expensive, fair, bargain, max_score):
    """For metrics where LOWER is better (e.g., P/E).
    expensive=worst, bargain=best."""
    if value is None or pd.isna(value) or value < 0:
        return max_score * 0.3  # negative P/E is bad signal (loss-making)
    if value <= bargain:
        return float(max_score)
    if value >= expensive:
        return 0.0
    if value <= fair:
        return max_score * (0.5 + 0.5 * (fair - value) / (fair - bargain))
    return max_score * 0.5 * (expensive - value) / (expensive - fair)


# ──────────────────────────────────────────────────────────────────
# VALUATION SUB-SCORE (25 pts)
# ──────────────────────────────────────────────────────────────────
def score_valuation(fund, sector="Default"):
    """Returns (score, details_dict)."""
    pe_bands = SECTOR_PE_BANDS.get(sector, SECTOR_PE_BANDS["Default"])
    bargain, fair, expensive = pe_bands

    pe = fund.get("pe")
    forward_pe = fund.get("forward_pe")
    peg = fund.get("peg")
    pb = fund.get("pb")
    ps = fund.get("ps")

    # P/E component (12 pts)
    pe_score = _bands_lower_better(pe, expensive, fair, bargain, 12)

    # Forward P/E component (5 pts) — if available, weight toward future expectation
    fpe_score = _bands_lower_better(forward_pe, expensive, fair, bargain, 5) \
        if forward_pe and forward_pe > 0 else 2.5

    # PEG component (4 pts) — PEG < 1 is great, 1-2 fair, >2 bad
    if peg is None or pd.isna(peg) or peg <= 0:
        peg_score = 2.0  # neutral
    elif peg < 1:
        peg_score = 4.0
    elif peg < 1.5:
        peg_score = 3.0
    elif peg < 2:
        peg_score = 2.0
    elif peg < 3:
        peg_score = 1.0
    else:
        peg_score = 0.0

    # P/B component (4 pts) — typical range 1-5 for healthy firms
    if pb is None or pd.isna(pb) or pb <= 0:
        pb_score = 2.0
    elif pb < 1:
        pb_score = 3.0  # potentially undervalued OR distressed
    elif pb < 3:
        pb_score = 4.0
    elif pb < 6:
        pb_score = 2.5
    else:
        pb_score = 1.0

    total = pe_score + fpe_score + peg_score + pb_score
    return total, {
        "pe_score": round(pe_score, 1),
        "forward_pe_score": round(fpe_score, 1),
        "peg_score": round(peg_score, 1),
        "pb_score": round(pb_score, 1),
        "subtotal": round(total, 1),
        "max": 25,
        "interpretation": _valuation_label(pe, bargain, fair, expensive),
    }


def _valuation_label(pe, bargain, fair, expensive):
    if pe is None or pd.isna(pe) or pe <= 0:
        return "N/A o negativo (pérdidas)"
    if pe <= bargain:
        return "BARATO vs sector"
    if pe <= fair:
        return "Valuación justa"
    if pe <= expensive:
        return "Caro vs sector"
    return "MUY CARO — riesgo de mean reversion"


# ──────────────────────────────────────────────────────────────────
# PROFITABILITY SUB-SCORE (25 pts)
# ──────────────────────────────────────────────────────────────────
def score_profitability(fund, sector="Default"):
    roe_bands = SECTOR_ROE_TARGETS.get(sector, SECTOR_ROE_TARGETS["Default"])
    margin_bands = SECTOR_NET_MARGIN_TARGETS.get(sector, SECTOR_NET_MARGIN_TARGETS["Default"])

    roe = fund.get("roe")
    roa = fund.get("roa")
    net_margin = fund.get("net_margin")
    operating_margin = fund.get("operating_margin")

    # ROE (10 pts) — sector-adjusted
    roe_score = _bands(roe, roe_bands[0], roe_bands[1], roe_bands[2], 10)

    # ROA (5 pts) — universal: <2% weak, 2-8% good, >8% excellent
    roa_score = _bands(roa, 2, 6, 12, 5)

    # Net margin (6 pts) — sector-adjusted
    nm_score = _bands(net_margin, margin_bands[0], margin_bands[1], margin_bands[2], 6)

    # Operating margin (4 pts) — generic
    om_score = _bands(operating_margin, 5, 12, 25, 4)

    total = roe_score + roa_score + nm_score + om_score
    return total, {
        "roe_score": round(roe_score, 1),
        "roa_score": round(roa_score, 1),
        "net_margin_score": round(nm_score, 1),
        "operating_margin_score": round(om_score, 1),
        "subtotal": round(total, 1),
        "max": 25,
        "interpretation": _profit_label(roe, roe_bands),
    }


def _profit_label(roe, bands):
    if roe is None or pd.isna(roe):
        return "N/A"
    if roe < bands[0]:
        return "RENTABILIDAD DÉBIL"
    if roe < bands[1]:
        return "Rentabilidad aceptable"
    if roe < bands[2]:
        return "Rentabilidad sólida"
    return "RENTABILIDAD EXCELENTE"


# ──────────────────────────────────────────────────────────────────
# GROWTH SUB-SCORE (25 pts)
# ──────────────────────────────────────────────────────────────────
def score_growth(fund):
    eps_growth_5y = fund.get("eps_growth_5y")  # annualized
    eps_growth_q = fund.get("eps_growth_q")    # quarterly YoY
    revenue_growth = fund.get("revenue_growth")  # quarterly YoY

    # EPS growth 5y (10 pts) — universal: <0 bad, 0-10 ok, 10-20 good, >20 excellent
    eps5y_score = _bands(eps_growth_5y, -5, 8, 25, 10)

    # EPS growth quarterly (8 pts) — short-term signal
    epsq_score = _bands(eps_growth_q, -10, 5, 30, 8)

    # Revenue growth (7 pts) — top line is harder to fake than EPS
    rev_score = _bands(revenue_growth, -5, 5, 20, 7)

    total = eps5y_score + epsq_score + rev_score
    return total, {
        "eps_growth_5y_score": round(eps5y_score, 1),
        "eps_growth_quarterly_score": round(epsq_score, 1),
        "revenue_growth_score": round(rev_score, 1),
        "subtotal": round(total, 1),
        "max": 25,
        "interpretation": _growth_label(eps_growth_5y, revenue_growth),
    }


def _growth_label(eps_5y, rev_growth):
    if eps_5y is None or pd.isna(eps_5y):
        return "N/A"
    if eps_5y < 0:
        return "EPS EN DECLIVE"
    if eps_5y < 5:
        return "Crecimiento bajo"
    if eps_5y < 15:
        return "Crecimiento moderado"
    if eps_5y < 30:
        return "Crecimiento sólido"
    return "HIPER-CRECIMIENTO (¿sostenible?)"


# ──────────────────────────────────────────────────────────────────
# STABILITY SUB-SCORE (15 pts)
# ──────────────────────────────────────────────────────────────────
def score_stability(fund, earnings_consistency=None):
    """earnings_consistency: dict from event_study with 'std_t1', 'beat_rate'"""
    debt_equity = fund.get("debt_equity")
    current_ratio = fund.get("current_ratio")
    quick_ratio = fund.get("quick_ratio")

    # Debt/Equity (6 pts) — lower is better, but 0 may also be suboptimal
    if debt_equity is None or pd.isna(debt_equity):
        de_score = 3.0
    elif debt_equity < 0.3:
        de_score = 6.0
    elif debt_equity < 0.6:
        de_score = 5.0
    elif debt_equity < 1.0:
        de_score = 3.5
    elif debt_equity < 2.0:
        de_score = 2.0
    else:
        de_score = 0.5

    # Current ratio (4 pts) — > 1 is healthy, > 1.5 strong
    cr_score = _bands(current_ratio, 0.8, 1.3, 2.0, 4)

    # Earnings consistency (5 pts) — comes from event_study
    if earnings_consistency is None:
        ec_score = 2.5
    else:
        beat_rate = earnings_consistency.get("beat_rate", 0.5)
        std_t1 = earnings_consistency.get("std_t1", 0.05)
        # Higher beat rate + lower std = more consistent
        ec_score = (beat_rate * 2.5) + max(0, 2.5 - std_t1 * 25)
        ec_score = min(5, max(0, ec_score))

    total = de_score + cr_score + ec_score
    return total, {
        "debt_equity_score": round(de_score, 1),
        "current_ratio_score": round(cr_score, 1),
        "earnings_consistency_score": round(ec_score, 1),
        "subtotal": round(total, 1),
        "max": 15,
    }


# ──────────────────────────────────────────────────────────────────
# EARNINGS MOMENTUM SUB-SCORE (10 pts)
# ──────────────────────────────────────────────────────────────────
def score_earnings_momentum(event_study_results):
    """Recent earnings beats and post-earnings drift."""
    if event_study_results is None:
        return 5.0, {"subtotal": 5.0, "max": 10, "note": "Sin datos de event study"}

    beat_rate = event_study_results.get("beat_rate", 0.5)
    avg_t5_after_beat = event_study_results.get("avg_t5_after_beat", 0)
    avg_t5_after_miss = event_study_results.get("avg_t5_after_miss", 0)
    last_4q_beats = event_study_results.get("last_4q_beats", 2)

    # Recent beat rate (5 pts) — last 4 quarters
    beat_4q_score = (last_4q_beats / 4) * 5

    # Drift after beats (3 pts) — does the stock follow through?
    if avg_t5_after_beat > 3:
        drift_score = 3
    elif avg_t5_after_beat > 1:
        drift_score = 2
    elif avg_t5_after_beat > 0:
        drift_score = 1
    else:
        drift_score = 0

    # Asymmetry penalty (2 pts) — punishes if misses cause big drops
    if avg_t5_after_miss < -5:
        asym_score = 0
    elif avg_t5_after_miss < -2:
        asym_score = 1
    else:
        asym_score = 2

    total = beat_4q_score + drift_score + asym_score
    return min(10, total), {
        "recent_beats_score": round(beat_4q_score, 1),
        "drift_score": round(drift_score, 1),
        "asymmetry_score": round(asym_score, 1),
        "subtotal": round(min(10, total), 1),
        "max": 10,
        "last_4q_beats": last_4q_beats,
    }


# ──────────────────────────────────────────────────────────────────
# MASTER QUALITY SCORE
# ──────────────────────────────────────────────────────────────────
def quality_score(fund, sector="Default", event_study_results=None):
    """
    Master function. Returns (total_score, full_breakdown_dict).
    Total score is 0-100.
    """
    val_score, val_det = score_valuation(fund, sector)
    prof_score, prof_det = score_profitability(fund, sector)
    growth_score, growth_det = score_growth(fund)
    stab_score, stab_det = score_stability(fund, event_study_results)
    em_score, em_det = score_earnings_momentum(event_study_results)

    total = val_score + prof_score + growth_score + stab_score + em_score
    total = max(0, min(100, total))

    if total >= 75:
        grade = "A"
        interpretation = "Calidad excepcional — fundamentales sólidos"
    elif total >= 65:
        grade = "B+"
        interpretation = "Buena calidad — fortalezas claras"
    elif total >= 55:
        grade = "B"
        interpretation = "Calidad media — sin grandes destacados"
    elif total >= 45:
        grade = "C"
        interpretation = "Calidad débil — varias áreas de preocupación"
    else:
        grade = "D"
        interpretation = "Calidad pobre — evitar o due diligence profunda"

    return total, {
        "total_score": round(total, 1),
        "grade": grade,
        "interpretation": interpretation,
        "valuation": val_det,
        "profitability": prof_det,
        "growth": growth_det,
        "stability": stab_det,
        "earnings_momentum": em_det,
        "sector": sector,
    }


# ──────────────────────────────────────────────────────────────────
# Z-SCORE for individual fundamentals (vs typical norms)
# ──────────────────────────────────────────────────────────────────
def fundamental_zscores(fund, sector="Default"):
    """
    Returns dict of z-scores for each metric vs typical sector ranges.
    z=0 is sector median, z>2 is "extremely high", z<-2 is "extremely low".
    """
    pe_bands = SECTOR_PE_BANDS.get(sector, SECTOR_PE_BANDS["Default"])
    roe_bands = SECTOR_ROE_TARGETS.get(sector, SECTOR_ROE_TARGETS["Default"])

    z = {}

    pe = fund.get("pe")
    if pe and pe > 0:
        # σ ≈ (high - low) / 4 (rough estimate of sector dispersion)
        sigma_pe = (pe_bands[2] - pe_bands[0]) / 4
        z["pe"] = round((pe - pe_bands[1]) / sigma_pe, 2)

    roe = fund.get("roe")
    if roe is not None:
        sigma_roe = (roe_bands[2] - roe_bands[0]) / 4
        z["roe"] = round((roe - roe_bands[1]) / sigma_roe, 2)

    eps_g = fund.get("eps_growth_5y")
    if eps_g is not None:
        z["eps_growth_5y"] = round((eps_g - 10) / 8, 2)  # vs 10% typical

    return z


# ──────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("QUALITY SCORE — Self Test")
    print("=" * 60)

    # Sample: a high-quality tech stock
    apple_like = {
        "pe": 28, "forward_pe": 26, "peg": 1.8, "pb": 35, "ps": 7,
        "roe": 145, "roa": 28, "net_margin": 26, "operating_margin": 30,
        "eps_growth_5y": 18, "eps_growth_q": 15, "revenue_growth": 7,
        "debt_equity": 1.6, "current_ratio": 1.0,
    }
    score, breakdown = quality_score(apple_like, "Technology",
                                      event_study_results={"beat_rate": 0.85,
                                                            "avg_t5_after_beat": 2.5,
                                                            "avg_t5_after_miss": -3.5,
                                                            "last_4q_beats": 4})
    print(f"\nAAPL-like (Tech, premium): score = {score:.1f} → {breakdown['grade']}")
    print(f"  Valuation: {breakdown['valuation']['subtotal']}/{breakdown['valuation']['max']} "
          f"({breakdown['valuation']['interpretation']})")
    print(f"  Profitability: {breakdown['profitability']['subtotal']}/{breakdown['profitability']['max']} "
          f"({breakdown['profitability']['interpretation']})")
    print(f"  Growth: {breakdown['growth']['subtotal']}/{breakdown['growth']['max']} "
          f"({breakdown['growth']['interpretation']})")
    print(f"  Stability: {breakdown['stability']['subtotal']}/{breakdown['stability']['max']}")
    print(f"  Earnings momentum: {breakdown['earnings_momentum']['subtotal']}/{breakdown['earnings_momentum']['max']}")
    print(f"\nInterpretation: {breakdown['interpretation']}")
