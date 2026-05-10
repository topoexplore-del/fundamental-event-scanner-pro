"""
Fundamental Event Scanner Pro — Data Builder (Hybrid API)
==========================================================
Multi-source data pipeline using:
  - Yahoo Finance (primary, unlimited)
  - Alpha Vantage (deep enrichment, 25/day free tier)
  - Marketstack (international fallback, 100/month free tier)

Strategy:
  - All tickers get Yahoo data (price + basic fundamentals + earnings)
  - Top N tickers (FESP_DEEP_ENRICHMENT_TOP_N) get AV enrichment:
    * High-quality fundamentals (50+ fields normalized to GAAP/IFRS)
    * Earnings with surprises
    * News sentiment (Alpha Intelligence)
    * Insider transactions
    * Upcoming earnings calendar
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_score import quality_score, fundamental_zscores
from event_study import full_event_study
from mc_integration import full_integrated_analysis
from api_clients import (fetch_complete_ticker, quota_status,
                          ALPHA_VANTAGE_KEY, MARKETSTACK_KEY)


TICKER_GROUPS = {
    "US Mega Cap Tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "ORCL", "AVGO", "CRM", "ADBE", "NFLX",
    ],
    "US Large Cap": [
        "BRK-B", "V", "MA", "JPM", "JNJ", "WMT", "PG", "HD",
        "UNH", "BAC", "KO", "PFE", "CVX", "ABBV", "MRK",
        "PEP", "COST", "TMO", "LLY", "DIS", "MCD", "NKE",
    ],
    "US Finance": ["GS", "MS", "C", "WFC", "AXP", "BLK", "SCHW", "PYPL"],
    "Europe": ["ASML", "SAP", "NVO", "SHEL", "BP", "AZN", "RIO", "UL", "HSBC"],
    "Asia ADRs": ["BABA", "TSM", "TCEHY", "SONY", "TM", "BIDU", "JD", "SE"],
    "Latam ADRs": ["VALE", "PBR", "ITUB", "BBD", "MELI", "EC", "CIB"],
    "ETFs Index": ["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "EEM", "EFA",
                   "XLK", "XLF", "XLE", "XLV", "GXG"],
}


def process_ticker(ticker, deep=False, deep_mode="full", quiet=False):
    """Run full pipeline for one ticker using hybrid API approach."""
    t0 = time.time()
    data = fetch_complete_ticker(ticker, deep=deep, deep_mode=deep_mode)
    if data is None:
        return None

    hist = data["price_history"]
    fundamentals = data["fundamentals"]
    earnings_hist = data["earnings_history"]
    sector = data["sector"]

    # Compute eps_growth_5y from earnings history if missing
    if (fundamentals.get("eps_growth_5y") is None and
            earnings_hist is not None and len(earnings_hist) >= 8):
        try:
            recent_eps = earnings_hist["eps_actual"].tail(4).mean()
            old_eps = earnings_hist["eps_actual"].head(4).mean()
            if old_eps > 0:
                years_span = (pd.to_datetime(earnings_hist["date"].iloc[-1]) -
                              pd.to_datetime(earnings_hist["date"].iloc[0])).days / 365.25
                if years_span > 0.5:
                    cagr = (recent_eps / old_eps) ** (1 / years_span) - 1
                    fundamentals["eps_growth_5y"] = round(cagr * 100, 2)
        except Exception:
            pass

    days_to_earn = None
    if "upcoming_earnings" in data:
        days_to_earn = data["upcoming_earnings"].get("days_to_next")

    try:
        analysis = full_integrated_analysis(
            hist, fundamentals, sector, earnings_hist,
            days_to_next_earnings=days_to_earn,
        )
    except Exception as e:
        if not quiet:
            print(f"  [{ticker}] analysis error: {e}")
        return None

    es = analysis["p1_independent"]["event_study"]
    es_summary = {}
    if es and isinstance(es, dict):
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
            "asymmetry_label": es.get("asymmetry", {}).get("interpretation"),
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
            "risk_reward": p2["interpretation"]["risk_reward"],
        })

    spe_21d = analysis["p1_independent"]["spe"]
    p4 = analysis["p4_master_score"]

    return {
        "ticker": ticker,
        "name": fundamentals.get("name", "")[:50],
        "sector": sector,
        "industry": fundamentals.get("industry"),
        "country": fundamentals.get("country"),
        "currency": fundamentals.get("currency", "USD"),
        "exchange": fundamentals.get("exchange"),
        "close": round(float(hist["Close"].iloc[-1]), 4),
        "market_cap": fundamentals.get("market_cap"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources_used": data["sources_used"],

        "fundamentals": {
            "pe": fundamentals.get("pe"),
            "forward_pe": fundamentals.get("forward_pe"),
            "peg": fundamentals.get("peg"),
            "pb": fundamentals.get("pb"),
            "ps": fundamentals.get("ps"),
            "ev_revenue": fundamentals.get("ev_revenue"),
            "ev_ebitda": fundamentals.get("ev_ebitda"),
            "roe": fundamentals.get("roe"),
            "roa": fundamentals.get("roa"),
            "net_margin": fundamentals.get("net_margin"),
            "operating_margin": fundamentals.get("operating_margin"),
            "eps_growth_q": fundamentals.get("eps_growth_q"),
            "eps_growth_5y": fundamentals.get("eps_growth_5y"),
            "revenue_growth": fundamentals.get("revenue_growth"),
            "debt_equity": fundamentals.get("debt_equity"),
            "current_ratio": fundamentals.get("current_ratio"),
            "dividend_yield": fundamentals.get("dividend_yield"),
            "payout_ratio": fundamentals.get("payout_ratio"),
            "beta": fundamentals.get("beta"),
            "eps": fundamentals.get("eps"),
            "ebitda": fundamentals.get("ebitda"),
            "52w_high": fundamentals.get("52w_high"),
            "52w_low": fundamentals.get("52w_low"),
            "ma_50d": fundamentals.get("ma_50d"),
            "ma_200d": fundamentals.get("ma_200d"),
            "analyst_target": fundamentals.get("analyst_target"),
        },

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
            "expected": spe_21d["expected"],
            "prob_up": spe_21d["prob_up"],
            "prob_tp5": spe_21d["prob_tp5"],
            "ci_68_low": spe_21d["ci_68_low"],
            "ci_68_high": spe_21d["ci_68_high"],
            "var_95": spe_21d["var_95"],
            "vol_annualized": spe_21d["volatility_annualized"],
            "vol_method": spe_21d["vol_method"],
            "reliable": spe_21d["reliable"],
        },
        "event_conditional_mc": p2_summary,
        "fesp_score": p4["fesp_score"],
        "fesp_grade": p4["grade"],
        "fesp_interpretation": p4["interpretation"],

        # Alpha Intelligence enrichments (only present if deep=True)
        "news_sentiment": data.get("news_sentiment"),
        "insider_signal": data.get("insider_signal"),
        "upcoming_earnings": data.get("upcoming_earnings"),

        "process_ms": int((time.time() - t0) * 1000),
    }


def load_watchlist():
    """Load user's watchlist for deep enrichment from watchlist.txt"""
    watchlist_path = Path(__file__).parent.parent / "watchlist.txt"
    if not watchlist_path.exists():
        return []

    tickers = []
    with open(watchlist_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Take only the first word in case there are inline comments
            ticker = line.split("#")[0].split()[0].strip().upper()
            if ticker:
                tickers.append(ticker)
    return tickers


def select_deep_enrichment_tickers(all_tickers, top_n):
    """
    Choose tickers for AV enrichment using watchlist.txt + budget awareness.
    
    Strategy:
      1. Load user's watchlist from watchlist.txt
      2. Cross-reference with universe
      3. Cap at top_n based on remaining AV quota
      4. Each ticker consumes ~5 AV requests (OVERVIEW + EARNINGS +
         NEWS + INSIDER + CALENDAR)
    """
    if top_n <= 0:
        return set()

    watchlist = load_watchlist()
    if not watchlist:
        # Fallback to default priority list if no watchlist
        watchlist = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            "BRK-B", "JPM", "V", "WMT", "JNJ", "UNH", "MA",
            "SPY", "QQQ", "VOO",
        ]

    # Filter to tickers in our universe
    universe_set = set(all_tickers)
    selected = []
    for t in watchlist:
        if t in universe_set and len(selected) < top_n:
            selected.append(t)

    return set(selected)


def calculate_budget_aware_top_n(requested_top_n, deep_mode="full"):
    """
    Determine how many deep tickers we can actually enrich given
    current AV quota remaining.
    
    Cost per ticker by mode:
      - 'full':              5 AV requests
      - 'lite':              2 AV requests
      - 'fundamentals_only': 1 AV request
    """
    q = quota_status()
    if not q["alpha_vantage"]["key_configured"]:
        return 0

    av_remaining = q["alpha_vantage"]["remaining"]
    cost_per_ticker = {"full": 5, "lite": 2, "fundamentals_only": 1}.get(deep_mode, 5)
    max_possible = av_remaining // cost_per_ticker

    if requested_top_n is None:
        return min(max_possible, 25)  # reasonable upper limit
    return min(requested_top_n, max_possible)


def build_snapshot(tickers=None, groups=None, out_dir="data", verbose=True,
                    deep_top_n=None, deep_mode="full"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if groups is None:
        groups = TICKER_GROUPS
    if tickers is not None:
        groups = {"Custom": list(tickers)}

    if deep_top_n is None:
        deep_top_n = int(os.environ.get("FESP_DEEP_ENRICHMENT_TOP_N", "5"))

    # Apply budget awareness — never exceed what AV quota allows
    actual_top_n = calculate_budget_aware_top_n(deep_top_n, deep_mode)

    all_tickers = []
    for ts in groups.values():
        all_tickers.extend(ts)
    deep_set = select_deep_enrichment_tickers(all_tickers, actual_top_n)

    total = len(all_tickers)
    print(f"Building FESP snapshot")
    print(f"  Tickers           : {total} across {len(groups)} groups")
    print(f"  Deep mode         : {deep_mode}")
    print(f"  Deep requested    : {deep_top_n}")
    print(f"  Deep actual       : {actual_top_n} (after AV quota check)")
    print(f"  Watchlist tickers : {len(deep_set)} ({', '.join(sorted(deep_set)) if deep_set else 'none'})")
    if ALPHA_VANTAGE_KEY:
        print(f"  Alpha Vantage     : configured")
    if MARKETSTACK_KEY:
        print(f"  Marketstack       : configured")
    print()

    q = quota_status()
    if q["alpha_vantage"]["key_configured"]:
        print(f"  AV quota today    : {q['alpha_vantage']['today']}/{q['alpha_vantage']['limit_free_tier']} used")
    if q["marketstack"]["key_configured"]:
        print(f"  MS quota month    : {q['marketstack']['month']}/{q['marketstack']['limit_free_tier']} used")
    print()

    results = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0",
        "groups": {},
        "n_tickers": 0,
        "n_failed": 0,
        "n_grade_a": 0,
        "n_grade_b_plus": 0,
        "n_deep_enriched": 0,
    }

    idx = 0
    for group_name, tickers_in_group in groups.items():
        if verbose:
            print(f"═══ {group_name} ═══")
        group_results = []
        for ticker in tickers_in_group:
            idx += 1
            is_deep = ticker in deep_set
            tag = "[DEEP]" if is_deep else "      "
            if verbose:
                print(f"  {tag} [{idx}/{total}] {ticker}...", end=" ", flush=True)
            r = process_ticker(ticker, deep=is_deep, deep_mode=deep_mode,
                                quiet=not verbose)
            if r is None:
                results["n_failed"] += 1
                if verbose:
                    print("FAILED")
                continue
            group_results.append(r)
            results["n_tickers"] += 1
            if is_deep:
                results["n_deep_enriched"] += 1
            if r["fesp_grade"] == "A":
                results["n_grade_a"] += 1
            elif r["fesp_grade"] == "B+":
                results["n_grade_b_plus"] += 1
            if verbose:
                src = ",".join(s.split("_")[0][:3] for s in r["sources_used"][:3])
                print(f"FESP={r['fesp_score']:.0f} ({r['fesp_grade']}) "
                      f"QS={r['quality_score']:.0f} src={src} {r['process_ms']}ms")

        results["groups"][group_name] = group_results

    output_file = out_path / "snapshot.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'═'*60}\nSUMMARY")
    print(f"  Tickers processed : {results['n_tickers']} / {total}")
    print(f"  Deep-enriched     : {results['n_deep_enriched']}")
    print(f"  Grade A           : {results['n_grade_a']}")
    print(f"  Grade B+          : {results['n_grade_b_plus']}")
    print(f"  Failed            : {results['n_failed']}")

    q = quota_status()
    if q["alpha_vantage"]["key_configured"]:
        print(f"  AV quota today    : {q['alpha_vantage']['today']}/{q['alpha_vantage']['limit_free_tier']} used")
    if q["marketstack"]["key_configured"]:
        print(f"  MS quota month    : {q['marketstack']['month']}/{q['marketstack']['limit_free_tier']} used")

    print(f"\n  Output            : {output_file}")
    print(f"  Size              : {os.path.getsize(output_file)/1024:.1f} KB")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="FESP — Build snapshot from Yahoo + Alpha Vantage + Marketstack")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated tickers")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deep-top-n", type=int, default=None,
                        help="Number of top tickers to enrich with Alpha Vantage. "
                             "If not set, uses budget-aware auto.")
    parser.add_argument("--deep-mode", default="full",
                        choices=["full", "lite", "fundamentals_only"],
                        help="AV depth: 'full'=5 endpoints (5 reqs/ticker), "
                             "'lite'=2 endpoints (2 reqs/ticker), "
                             "'fundamentals_only'=1 endpoint (1 req/ticker). "
                             "Use 'lite' to fit more tickers in free tier.")
    args = parser.parse_args()

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]

    build_snapshot(tickers=tickers, out_dir=args.out_dir,
                    verbose=not args.quiet, deep_top_n=args.deep_top_n,
                    deep_mode=args.deep_mode)


if __name__ == "__main__":
    main()
