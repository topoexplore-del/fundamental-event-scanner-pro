"""
Fundamental Event Scanner Pro — Event Study Module
====================================================
Analyzes historical price reactions around earnings announcements.

Methodology:
  - For each earnings event, compute returns at T+1, T+5, T+21, T+63
  - Classify event as BEAT (EPS actual > estimate) or MISS
  - Aggregate statistics:
      - Mean and median return per window
      - Win rate (% positive)
      - Standard deviation of reactions
      - Beat vs miss differential
      - Recent quarter pattern (last 4 quarters)
  - Detect post-earnings announcement drift (PEAD)

OUTPUT used by:
  - quality_score.py (stability + earnings momentum components)
  - fundamental_event_engine.py (full integration)
  - Bloomberg-style dashboard (per-ticker earnings stats)

NOT GUARANTEED:
  Historical patterns DO NOT predict future earnings reactions perfectly.
  A ticker that beat 8/10 times historically can miss next quarter.
  Use as ONE input among many.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# Windows after earnings announcement (trading days)
EVENT_WINDOWS = [1, 5, 21, 63]


def _find_next_trading_day(price_df, target_date):
    """Find the next trading day on or after target_date."""
    target = pd.to_datetime(target_date)
    if hasattr(target, "tz_localize"):
        try:
            target = target.tz_localize(None)
        except Exception:
            pass

    # Make price index naive for comparison
    price_idx = price_df.index
    if hasattr(price_idx, "tz") and price_idx.tz is not None:
        price_idx = price_idx.tz_localize(None)

    mask = price_idx >= target
    matches = np.where(mask)[0]
    if len(matches) == 0:
        return None
    return matches[0]


def compute_event_returns(price_df, earnings_dates, windows=None):
    """
    For each earnings date, compute returns at multiple windows.
    
    Args:
        price_df: DataFrame with 'Close' column, indexed by date
        earnings_dates: list of dates
        windows: trading-day windows (default [1, 5, 21, 63])
    
    Returns:
        DataFrame with columns:
            date, t0_idx, t0_close,
            t1_return, t5_return, t21_return, t63_return
    """
    if windows is None:
        windows = EVENT_WINDOWS

    results = []
    for date in earnings_dates:
        idx = _find_next_trading_day(price_df, date)
        if idx is None or idx >= len(price_df) - max(windows):
            continue

        t0_close = float(price_df["Close"].iloc[idx])
        # Use day BEFORE earnings as reference (if available)
        # to capture the announcement-day reaction
        if idx > 0:
            t_minus_1_close = float(price_df["Close"].iloc[idx - 1])
        else:
            t_minus_1_close = t0_close

        row = {"date": price_df.index[idx], "t0_idx": idx, "t0_close": t0_close,
               "tm1_close": t_minus_1_close}

        for w in windows:
            future_idx = idx + w
            if future_idx >= len(price_df):
                row[f"t{w}_return"] = np.nan
                continue
            future_close = float(price_df["Close"].iloc[future_idx])
            if t_minus_1_close <= 0:
                row[f"t{w}_return"] = np.nan
                continue
            ret = (future_close / t_minus_1_close - 1) * 100
            row[f"t{w}_return"] = round(ret, 3)

        results.append(row)

    return pd.DataFrame(results)


def classify_beat_miss(earnings_df, surprise_threshold=0.0):
    """
    Classify each earnings event as BEAT, MISS, or INLINE.
    
    Args:
        earnings_df: DataFrame with 'eps_actual', 'eps_estimate' columns
        surprise_threshold: minimum % surprise to count as beat/miss (default 0)
    
    Returns:
        DataFrame with added 'classification' column
    """
    df = earnings_df.copy()
    if "eps_actual" not in df.columns or "eps_estimate" not in df.columns:
        df["classification"] = "UNKNOWN"
        df["surprise_pct"] = np.nan
        return df

    surprises = []
    classifications = []
    for _, row in df.iterrows():
        actual = row.get("eps_actual")
        estimate = row.get("eps_estimate")
        if pd.isna(actual) or pd.isna(estimate) or estimate == 0:
            surprises.append(np.nan)
            classifications.append("UNKNOWN")
            continue

        if estimate > 0:
            surprise = ((actual - estimate) / abs(estimate)) * 100
        else:
            # Negative estimate (loss expected). Smaller loss = beat.
            surprise = ((actual - estimate) / abs(estimate)) * 100

        surprises.append(round(surprise, 2))
        if surprise > surprise_threshold:
            classifications.append("BEAT")
        elif surprise < -surprise_threshold:
            classifications.append("MISS")
        else:
            classifications.append("INLINE")

    df["surprise_pct"] = surprises
    df["classification"] = classifications
    return df


def event_study_stats(event_returns_df, classification_df=None):
    """
    Compute aggregate statistics across all earnings events.
    
    Returns:
        dict with comprehensive statistics
    """
    if event_returns_df is None or len(event_returns_df) == 0:
        return _empty_stats()

    stats = {
        "n_events": len(event_returns_df),
        "windows": {},
    }

    # Per-window aggregates
    for w in EVENT_WINDOWS:
        col = f"t{w}_return"
        if col not in event_returns_df.columns:
            continue
        returns = event_returns_df[col].dropna()
        if len(returns) == 0:
            continue
        stats["windows"][f"t{w}"] = {
            "n": int(len(returns)),
            "mean": round(float(returns.mean()), 3),
            "median": round(float(returns.median()), 3),
            "std": round(float(returns.std()), 3),
            "win_rate": round(float((returns > 0).mean()) * 100, 2),
            "max_gain": round(float(returns.max()), 2),
            "max_loss": round(float(returns.min()), 2),
        }

    # Beat vs Miss differential
    if classification_df is not None and "classification" in classification_df.columns:
        merged = event_returns_df.copy()
        # Align by index — assumes same order
        if len(classification_df) == len(event_returns_df):
            merged["classification"] = classification_df["classification"].values
            merged["surprise_pct"] = classification_df["surprise_pct"].values

            beats = merged[merged["classification"] == "BEAT"]
            misses = merged[merged["classification"] == "MISS"]
            inline = merged[merged["classification"] == "INLINE"]

            stats["classification"] = {
                "beats": len(beats),
                "misses": len(misses),
                "inline": len(inline),
                "unknown": (merged["classification"] == "UNKNOWN").sum(),
                "beat_rate": round(len(beats) / max(len(merged), 1) * 100, 2),
            }

            # Beat-specific reactions
            if len(beats) > 0 and "t5_return" in beats.columns:
                beat_t5 = beats["t5_return"].dropna()
                if len(beat_t5) > 0:
                    stats["beat_reaction"] = {
                        "n": len(beat_t5),
                        "avg_t1": round(float(beats["t1_return"].dropna().mean()), 3) if "t1_return" in beats else None,
                        "avg_t5": round(float(beat_t5.mean()), 3),
                        "avg_t21": round(float(beats["t21_return"].dropna().mean()), 3) if "t21_return" in beats else None,
                        "win_rate_t5": round(float((beat_t5 > 0).mean()) * 100, 2),
                    }

            # Miss-specific reactions
            if len(misses) > 0 and "t5_return" in misses.columns:
                miss_t5 = misses["t5_return"].dropna()
                if len(miss_t5) > 0:
                    stats["miss_reaction"] = {
                        "n": len(miss_t5),
                        "avg_t1": round(float(misses["t1_return"].dropna().mean()), 3) if "t1_return" in misses else None,
                        "avg_t5": round(float(miss_t5.mean()), 3),
                        "avg_t21": round(float(misses["t21_return"].dropna().mean()), 3) if "t21_return" in misses else None,
                        "win_rate_t5": round(float((miss_t5 > 0).mean()) * 100, 2),
                    }

            # Asymmetry — gap between beat and miss reactions
            if "beat_reaction" in stats and "miss_reaction" in stats:
                stats["asymmetry"] = {
                    "t5_diff": round(stats["beat_reaction"]["avg_t5"] - stats["miss_reaction"]["avg_t5"], 3),
                    "interpretation": _asymmetry_label(
                        stats["beat_reaction"]["avg_t5"],
                        stats["miss_reaction"]["avg_t5"]
                    ),
                }

    # Recent pattern (last 4 events)
    if len(event_returns_df) >= 4:
        recent = event_returns_df.tail(4)
        recent_t5 = recent["t5_return"].dropna() if "t5_return" in recent.columns else pd.Series([])
        if len(recent_t5) > 0:
            stats["recent_4q"] = {
                "avg_t5": round(float(recent_t5.mean()), 3),
                "win_rate": round(float((recent_t5 > 0).mean()) * 100, 2),
            }

        # Last 4q beats (used by quality_score)
        if classification_df is not None and len(classification_df) >= 4:
            last_4 = classification_df.tail(4)
            stats["last_4q_beats"] = int((last_4["classification"] == "BEAT").sum())

    # PEAD signal — does drift continue after T+1?
    if "t1" in stats["windows"] and "t21" in stats["windows"]:
        t1_mean = stats["windows"]["t1"]["mean"]
        t21_mean = stats["windows"]["t21"]["mean"]
        drift = t21_mean - t1_mean
        if abs(drift) >= 1:
            stats["pead_signal"] = {
                "drift_t1_to_t21": round(drift, 3),
                "interpretation": "Drift positivo (PEAD)" if drift > 1
                                  else "Drift negativo (mean reversion)" if drift < -1
                                  else "Sin drift claro",
            }

    return stats


def _asymmetry_label(beat_avg, miss_avg):
    """Label the asymmetry of reactions."""
    diff = beat_avg - miss_avg
    if diff > 8:
        return "Asimetría fuerte hacia beats — premia más que castiga"
    elif diff > 3:
        return "Asimetría positiva moderada"
    elif diff > -3:
        return "Reacciones simétricas"
    else:
        return "Asimetría negativa — castiga más que premia"


def _empty_stats():
    return {
        "n_events": 0,
        "windows": {},
        "note": "Sin datos suficientes para event study",
    }


def full_event_study(price_df, earnings_df):
    """
    Complete event study pipeline.
    
    Args:
        price_df: DataFrame with 'Close', indexed by date
        earnings_df: DataFrame with columns ['date', 'eps_actual', 'eps_estimate']
    
    Returns:
        dict with full study results
    """
    if earnings_df is None or len(earnings_df) == 0:
        return _empty_stats()

    # Compute returns around each earnings date
    event_returns = compute_event_returns(price_df,
                                            earnings_df["date"].tolist())

    # Classify beat/miss
    classified = classify_beat_miss(earnings_df)

    # Aggregate statistics
    stats = event_study_stats(event_returns, classified)

    # Add for quality_score compatibility
    if "windows" in stats and "t1" in stats["windows"]:
        stats["std_t1"] = stats["windows"]["t1"].get("std", 0.05) / 100
    if "classification" in stats:
        stats["beat_rate"] = stats["classification"].get("beat_rate", 50) / 100
    if "beat_reaction" in stats:
        stats["avg_t5_after_beat"] = stats["beat_reaction"].get("avg_t5", 0)
    if "miss_reaction" in stats:
        stats["avg_t5_after_miss"] = stats["miss_reaction"].get("avg_t5", 0)

    return stats


# ──────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EVENT STUDY — Self Test (synthetic data)")
    print("=" * 60)

    # Generate synthetic 5y daily price + 20 quarterly earnings
    np.random.seed(42)
    n_days = 252 * 5
    returns = np.random.normal(0.0005, 0.015, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range(end="2026-04-01", periods=n_days, freq="B")
    price_df = pd.DataFrame({"Close": prices}, index=dates)

    # Earnings every ~63 trading days, with 70% beat rate
    earnings_dates = [dates[63 * (i + 1) - 1] for i in range(18)]
    earnings_data = []
    for i, d in enumerate(earnings_dates):
        is_beat = np.random.random() < 0.7
        actual = np.random.uniform(1.5, 3.0)
        estimate = actual * (np.random.uniform(0.95, 1.0) if is_beat
                              else np.random.uniform(1.0, 1.05))
        earnings_data.append({"date": d, "eps_actual": actual,
                              "eps_estimate": estimate})

    earnings_df = pd.DataFrame(earnings_data)
    stats = full_event_study(price_df, earnings_df)

    print(f"\nN events analyzed: {stats['n_events']}")
    print(f"\nReturn windows:")
    for w_name, w_stats in stats["windows"].items():
        print(f"  {w_name}: avg={w_stats['mean']:+.2f}%  "
              f"win_rate={w_stats['win_rate']:.0f}%  "
              f"std={w_stats['std']:.2f}")

    if "classification" in stats:
        c = stats["classification"]
        print(f"\nClassification: beats={c['beats']} misses={c['misses']} "
              f"inline={c['inline']}  beat_rate={c['beat_rate']}%")

    if "beat_reaction" in stats:
        br = stats["beat_reaction"]
        print(f"After BEAT:  avg T+1={br['avg_t1']:+.2f}%  T+5={br['avg_t5']:+.2f}%  "
              f"T+21={br['avg_t21']:+.2f}%  (n={br['n']})")
    if "miss_reaction" in stats:
        mr = stats["miss_reaction"]
        print(f"After MISS:  avg T+1={mr['avg_t1']:+.2f}%  T+5={mr['avg_t5']:+.2f}%  "
              f"T+21={mr['avg_t21']:+.2f}%  (n={mr['n']})")

    if "asymmetry" in stats:
        print(f"\nAsymmetry: T+5 diff={stats['asymmetry']['t5_diff']:+.2f}pp")
        print(f"           {stats['asymmetry']['interpretation']}")
    if "pead_signal" in stats:
        print(f"\nPEAD signal: drift={stats['pead_signal']['drift_t1_to_t21']:+.2f}pp")
        print(f"             {stats['pead_signal']['interpretation']}")
