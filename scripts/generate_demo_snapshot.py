"""
Generate a demo snapshot.json with synthetic data for FESP.
Used to demonstrate the dashboard without requiring API access.
"""
import json
import os
import sys
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_score import quality_score, fundamental_zscores
from event_study import full_event_study
from mc_integration import full_integrated_analysis


def make_synthetic_ticker(ticker, sector, pe, roe, roa, eps_g_5y,
                           initial_price, drift, vol, beat_rate=0.65,
                           seed=42, has_news=False, has_insider=False,
                           days_to_earnings=None):
    np.random.seed(seed)

    # Generate price history
    n_days = 252 * 3
    omega, alpha_p, beta_p = 5e-6, 0.08, 0.90
    sigma2 = np.zeros(n_days)
    returns = np.zeros(n_days)
    sigma2[0] = omega / (1 - alpha_p - beta_p)
    shocks = np.random.standard_t(df=6, size=n_days) / np.sqrt(6 / 4)
    for t in range(1, n_days):
        sigma2[t] = omega + alpha_p * returns[t-1]**2 + beta_p * sigma2[t-1]
        returns[t] = drift + np.sqrt(sigma2[t]) * shocks[t] * (vol / 0.015)
    prices = initial_price * np.exp(np.cumsum(returns))
    # Generate exactly n_days business days, working backwards from today
    end_date = pd.Timestamp.now().normalize()
    dates = pd.bdate_range(end=end_date, periods=n_days)
    # Trim or pad to match
    if len(dates) != n_days:
        dates = pd.bdate_range(end=end_date, periods=n_days + 5)[-n_days:]
    hist = pd.DataFrame({"Close": prices, "High": prices * 1.01,
                         "Low": prices * 0.99,
                         "Volume": np.random.randint(1e6, 1e7, n_days)},
                        index=dates)

    # Generate earnings history
    earnings_data = []
    for i in range(12):  # 3 years quarterly
        is_beat = np.random.random() < beat_rate
        actual = np.random.uniform(0.8, 3.0)
        if is_beat:
            estimate = actual * np.random.uniform(0.92, 0.99)
        else:
            estimate = actual * np.random.uniform(1.01, 1.10)
        earnings_data.append({
            "date": datetime.now() - timedelta(days=90 * (12 - i)),
            "eps_actual": actual,
            "eps_estimate": estimate,
        })
    earnings_df = pd.DataFrame(earnings_data)

    fundamentals = {
        "pe": pe, "forward_pe": pe * 0.92,
        "peg": pe / max(eps_g_5y, 1), "pb": pe * 0.18, "ps": pe * 0.18,
        "roe": roe, "roa": roa, "net_margin": roe * 0.7,
        "operating_margin": roe * 0.85,
        "eps_growth_q": eps_g_5y * 0.9, "eps_growth_5y": eps_g_5y,
        "revenue_growth": eps_g_5y * 0.6,
        "debt_equity": np.random.uniform(0.3, 1.5),
        "current_ratio": np.random.uniform(1.0, 2.5),
        "dividend_yield": np.random.uniform(0, 3),
        "beta": np.random.uniform(0.7, 1.5),
        "ebitda": initial_price * 1e7,
        "ev_revenue": pe * 0.15, "ev_ebitda": pe * 0.7,
        "analyst_target": prices[-1] * np.random.uniform(0.95, 1.15),
        "ma_50d": prices[-50:].mean() if len(prices) >= 50 else prices[-1],
        "ma_200d": prices[-200:].mean() if len(prices) >= 200 else prices[-1],
        "52w_high": float(prices.max()),
        "52w_low": float(prices.min()),
    }

    analysis = full_integrated_analysis(hist, fundamentals, sector, earnings_df,
                                         days_to_next_earnings=days_to_earnings)

    es = analysis["p1_independent"]["event_study"]
    es_summary = {
        "n_events": es.get("n_events", 0),
        "beat_rate": es.get("classification", {}).get("beat_rate"),
        "avg_t1_after_beat": es.get("beat_reaction", {}).get("avg_t1"),
        "avg_t5_after_beat": es.get("beat_reaction", {}).get("avg_t5"),
        "avg_t21_after_beat": es.get("beat_reaction", {}).get("avg_t21"),
        "avg_t1_after_miss": es.get("miss_reaction", {}).get("avg_t1"),
        "avg_t5_after_miss": es.get("miss_reaction", {}).get("avg_t5"),
        "avg_t21_after_miss": es.get("miss_reaction", {}).get("avg_t21"),
        "asymmetry_t5": es.get("asymmetry", {}).get("t5_diff"),
        "pead_drift": es.get("pead_signal", {}).get("drift_t1_to_t21"),
        "last_4q_beats": es.get("last_4q_beats"),
    }

    p2 = analysis["p2_event_conditional"]
    p2_summary = {"days_to_next_earnings": p2.get("days_to_next_earnings")}
    if "scenario_beat" in p2:
        p2_summary.update({
            "beat_expected": p2["scenario_beat"]["expected"],
            "beat_prob_up": p2["scenario_beat"]["prob_up"],
            "miss_expected": p2["scenario_miss"]["expected"],
            "miss_prob_up": p2["scenario_miss"]["prob_up"],
            "weighted_expected": p2["weighted"]["expected"],
            "weighted_prob_up": p2["weighted"]["prob_up"],
            "bias": p2["interpretation"]["bias"],
        })

    spe = analysis["p1_independent"]["spe"]
    p4 = analysis["p4_master_score"]

    sources = ["yahoo_price", "yahoo_fundamentals"]
    news_sentiment = None
    insider_signal = None
    upcoming = None
    if has_news:
        sources += ["alpha_vantage_overview", "alpha_vantage_news"]
        s_score = np.random.uniform(-0.2, 0.4)
        news_sentiment = {
            "n_articles": np.random.randint(10, 50),
            "weighted_sentiment": round(s_score, 3),
            "label": "BULLISH" if s_score > 0.35 else
                     "SOMEWHAT_BULLISH" if s_score > 0.15 else
                     "NEUTRAL" if s_score > -0.15 else
                     "SOMEWHAT_BEARISH" if s_score > -0.35 else "BEARISH",
        }
    if has_insider:
        sources += ["alpha_vantage_insider"]
        n_buys = np.random.randint(0, 8)
        n_sells = np.random.randint(0, 12)
        total_buys = n_buys * np.random.uniform(50000, 500000)
        total_sells = n_sells * np.random.uniform(50000, 500000)
        insider_signal = {
            "n_buys_90d": n_buys, "n_sells_90d": n_sells,
            "total_buy_value": round(total_buys, 0),
            "total_sell_value": round(total_sells, 0),
            "net_flow": round(total_buys - total_sells, 0),
            "signal": "INSIDER_BUYING" if total_buys > total_sells * 1.5 else
                      "INSIDER_SELLING" if total_sells > total_buys * 1.5 else "NEUTRAL",
        }
    if days_to_earnings is not None and days_to_earnings <= 60:
        upcoming = {
            "next_earnings_date": (datetime.now() + timedelta(days=days_to_earnings)).strftime("%Y-%m-%d"),
            "days_to_next": days_to_earnings,
            "estimate": float(earnings_df["eps_actual"].tail(4).mean()) * 1.02,
        }

    return {
        "ticker": ticker, "name": ticker + " Inc.",
        "sector": sector, "currency": "USD",
        "close": round(float(prices[-1]), 4),
        "market_cap": int(initial_price * 1e9),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources_used": sources,
        "fundamentals": {k: round(v, 4) if isinstance(v, (int, float)) else v
                         for k, v in fundamentals.items()},
        "zscores": fundamental_zscores(fundamentals, sector),
        "quality_score": p4["components"]["quality_score"],
        "quality_grade": analysis["p1_independent"]["quality_breakdown"]["grade"],
        "quality_breakdown": {
            "valuation": analysis["p1_independent"]["quality_breakdown"]["valuation"]["subtotal"],
            "valuation_label": analysis["p1_independent"]["quality_breakdown"]["valuation"]["interpretation"],
            "profitability": analysis["p1_independent"]["quality_breakdown"]["profitability"]["subtotal"],
            "profitability_label": analysis["p1_independent"]["quality_breakdown"]["profitability"]["interpretation"],
            "growth": analysis["p1_independent"]["quality_breakdown"]["growth"]["subtotal"],
            "growth_label": analysis["p1_independent"]["quality_breakdown"]["growth"]["interpretation"],
            "stability": analysis["p1_independent"]["quality_breakdown"]["stability"]["subtotal"],
            "earnings_momentum": analysis["p1_independent"]["quality_breakdown"]["earnings_momentum"]["subtotal"],
        },
        "event_study": es_summary,
        "spe_21d": {
            "expected": spe["expected"], "prob_up": spe["prob_up"],
            "prob_tp5": spe["prob_tp5"],
            "ci_68_low": spe["ci_68_low"], "ci_68_high": spe["ci_68_high"],
            "var_95": spe["var_95"],
            "vol_annualized": spe["volatility_annualized"],
            "vol_method": spe["vol_method"], "reliable": spe["reliable"],
        },
        "event_conditional_mc": p2_summary,
        "fesp_score": p4["fesp_score"],
        "fesp_grade": p4["grade"],
        "fesp_interpretation": p4["interpretation"],
        "news_sentiment": news_sentiment,
        "insider_signal": insider_signal,
        "upcoming_earnings": upcoming,
        "process_ms": np.random.randint(800, 2500),
    }


def main():
    print("Generating FESP demo snapshot...\n")
    groups = {
        "US Mega Cap Tech": [
            ("AAPL", "Technology", 28, 145, 28, 18, 180, 0.0006, 0.018, 0.85, 1, True, True, 12),
            ("MSFT", "Technology", 32, 38, 18, 14, 420, 0.0007, 0.016, 0.80, 2, True, True, 5),
            ("GOOGL", "Communication Services", 22, 28, 15, 16, 165, 0.0005, 0.020, 0.78, 3, True, False, 18),
            ("NVDA", "Technology", 65, 95, 55, 45, 820, 0.0010, 0.030, 0.92, 4, True, True, 25),
            ("META", "Communication Services", 24, 32, 22, 21, 510, 0.0008, 0.022, 0.80, 5, True, False, 8),
            ("TSLA", "Consumer Cyclical", 65, 18, 8, 12, 240, 0.0003, 0.035, 0.62, 6, False, False, 32),
        ],
        "US Large Cap": [
            ("BRK-B", "Financial Services", 11, 14, 5, 8, 460, 0.0004, 0.012, 0.85, 10, False, False, 45),
            ("JPM", "Financial Services", 12, 17, 1.5, 9, 230, 0.0004, 0.018, 0.78, 11, True, False, 22),
            ("WMT", "Consumer Defensive", 32, 22, 8, 7, 165, 0.0003, 0.015, 0.72, 12, False, False, 50),
            ("UNH", "Healthcare", 18, 25, 10, 13, 530, 0.0005, 0.018, 0.68, 13, False, True, 14),
            ("V", "Financial Services", 30, 50, 22, 13, 295, 0.0005, 0.014, 0.82, 14, True, False, 38),
        ],
        "Europe": [
            ("ASML", "Technology", 35, 50, 20, 25, 720, 0.0007, 0.025, 0.90, 20, True, False, 28),
            ("SAP", "Technology", 28, 18, 9, 10, 195, 0.0004, 0.018, 0.65, 21, False, False, 42),
            ("NVO", "Healthcare", 26, 75, 28, 18, 88, 0.0008, 0.022, 0.70, 22, True, False, 16),
        ],
        "Asia ADRs": [
            ("BABA", "Consumer Cyclical", 14, 8, 3, -8, 92, 0.0001, 0.025, 0.55, 30, False, False, 35),
            ("TSM", "Technology", 22, 28, 18, 15, 180, 0.0006, 0.022, 0.78, 31, True, False, 22),
        ],
        "Latam ADRs": [
            ("VALE", "Basic Materials", 6, 22, 12, -2, 11, 0.0001, 0.025, 0.50, 40, False, False, 55),
            ("ITUB", "Financial Services", 8, 18, 1.5, 8, 7, 0.0003, 0.020, 0.62, 41, False, False, 30),
            ("EC", "Energy", 6, 18, 8, 5, 12, 0.0002, 0.022, 0.48, 42, False, False, 40),
            ("CIB", "Financial Services", 7, 14, 1.5, 12, 38, 0.0004, 0.018, 0.65, 43, False, False, 25),
        ],
    }

    results = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0-demo",
        "groups": {},
        "n_tickers": 0, "n_failed": 0,
        "n_grade_a": 0, "n_grade_b_plus": 0,
        "n_deep_enriched": 0,
        "note": "DEMO snapshot with synthetic data (no API calls made).",
    }

    for group_name, tickers_data in groups.items():
        print(f"═══ {group_name} ═══")
        group_results = []
        for params in tickers_data:
            ticker = params[0]
            print(f"  {ticker}...", end=" ", flush=True)
            try:
                r = make_synthetic_ticker(*params)
                group_results.append(r)
                results["n_tickers"] += 1
                if r["fesp_grade"] == "A":
                    results["n_grade_a"] += 1
                elif r["fesp_grade"] == "B+":
                    results["n_grade_b_plus"] += 1
                if r["news_sentiment"] or r["insider_signal"]:
                    results["n_deep_enriched"] += 1
                print(f"FESP={r['fesp_score']:.0f} ({r['fesp_grade']}) "
                      f"QS={r['quality_score']:.0f}")
            except Exception as e:
                print(f"FAILED: {e}")
                results["n_failed"] += 1
        results["groups"][group_name] = group_results

    out = os.path.join(os.path.dirname(__file__), "..", "data", "snapshot.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nWrote demo snapshot: {out}")
    print(f"Tickers: {results['n_tickers']} | "
          f"Grade A: {results['n_grade_a']} | "
          f"Grade B+: {results['n_grade_b_plus']}")
    print(f"Size: {os.path.getsize(out)/1024:.1f} KB")


if __name__ == "__main__":
    main()
