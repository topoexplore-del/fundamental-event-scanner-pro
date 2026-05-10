"""
Fundamental Event Scanner Pro — Multi-Source Data Clients
==========================================================
Unified wrappers for Yahoo Finance, Alpha Vantage, and Marketstack.

DESIGN PRINCIPLES:
  1. API keys NEVER hardcoded — loaded from environment variables
  2. Rate limiting respected per API (AV=5/min, MS=5/sec)
  3. Built-in caching — never re-request same data in 24h window
  4. Graceful fallback — if one API fails, try next
  5. Respect free-tier budgets — track daily/monthly usage

DATA SOURCE STRATEGY (free tier optimized):
  - Yahoo:       primary source (unlimited, basic fundamentals + prices)
  - Alpha Vantage: enrich TOP tickers with deep fundamentals + earnings surprise
  - Marketstack: fallback for international tickers where Yahoo data is poor

ENVIRONMENT VARIABLES REQUIRED (in .env file, NEVER in code):
  ALPHA_VANTAGE_API_KEY=your_key_here
  MARKETSTACK_API_KEY=your_key_here

USAGE:
    from api_clients import get_data
    data = get_data("AAPL", deep=True)  # uses all available sources optimally
"""

import os
import sys
import time
import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# Environment loading — NEVER hardcode keys
# ═══════════════════════════════════════════════════════════════════
def _load_env():
    """Load .env file if exists. Keys come from environment ONLY."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env()

ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
MARKETSTACK_KEY = os.environ.get("MARKETSTACK_API_KEY", "")

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Rate Limiter
# ═══════════════════════════════════════════════════════════════════
class RateLimiter:
    """Sliding-window rate limiter."""
    def __init__(self, max_requests, window_seconds):
        self.max = max_requests
        self.window = window_seconds
        self.times = deque()

    def wait_if_needed(self):
        now = time.time()
        # Drop old timestamps outside window
        while self.times and self.times[0] < now - self.window:
            self.times.popleft()
        if len(self.times) >= self.max:
            sleep_time = self.times[0] + self.window - now + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.times.append(time.time())


# Per-API limiters (free tier limits)
AV_PER_MIN = RateLimiter(5, 60)  # Alpha Vantage: 5/min
MS_PER_SEC = RateLimiter(5, 1)   # Marketstack: 5/sec

# Daily/monthly counters (persisted to disk)
USAGE_FILE = CACHE_DIR / "_api_usage.json"


def _load_usage():
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_usage(usage):
    try:
        USAGE_FILE.write_text(json.dumps(usage, indent=2))
    except Exception:
        pass


def _check_quota(api_name, max_per_day=None, max_per_month=None):
    """Returns True if we can make a request, False if quota exhausted."""
    usage = _load_usage()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    month = datetime.now(timezone.utc).strftime("%Y-%m")

    api_usage = usage.get(api_name, {})
    daily = api_usage.get("daily", {}).get(today, 0)
    monthly = api_usage.get("monthly", {}).get(month, 0)

    if max_per_day and daily >= max_per_day:
        return False, f"Daily quota exhausted ({daily}/{max_per_day})"
    if max_per_month and monthly >= max_per_month:
        return False, f"Monthly quota exhausted ({monthly}/{max_per_month})"

    return True, "OK"


def _record_request(api_name):
    """Increment counters."""
    usage = _load_usage()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    month = datetime.now(timezone.utc).strftime("%Y-%m")

    if api_name not in usage:
        usage[api_name] = {"daily": {}, "monthly": {}}
    usage[api_name].setdefault("daily", {})
    usage[api_name].setdefault("monthly", {})
    usage[api_name]["daily"][today] = usage[api_name]["daily"].get(today, 0) + 1
    usage[api_name]["monthly"][month] = usage[api_name]["monthly"].get(month, 0) + 1
    _save_usage(usage)


# ═══════════════════════════════════════════════════════════════════
# Caching
# ═══════════════════════════════════════════════════════════════════
def _cache_key(*args):
    s = "_".join(str(a) for a in args)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def _cache_get(key, max_age_hours=24):
    f = CACHE_DIR / f"{key}.json"
    if not f.exists():
        return None
    age_hours = (time.time() - f.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def _cache_set(key, data):
    f = CACHE_DIR / f"{key}.json"
    try:
        f.write_text(json.dumps(data, default=str))
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# Yahoo Finance Client (primary source, unlimited)
# ═══════════════════════════════════════════════════════════════════
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


def yahoo_fetch_price(ticker, period="5y"):
    """Fetch historical OHLCV from Yahoo Finance."""
    if not HAS_YF:
        return None, None
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist is None or len(hist) < 50:
            return None, None
        return tk, hist
    except Exception:
        return None, None


def yahoo_fetch_fundamentals(tk):
    """Extract fundamentals from yfinance Ticker.info."""
    try:
        info = tk.info
        if not info or len(info) < 5:
            return {}, "Unknown"
    except Exception:
        return {}, "Unknown"

    sector = info.get("sector", "Unknown")
    fund = {
        "name": info.get("longName") or info.get("shortName") or "",
        "market_cap": info.get("marketCap"),
        "currency": info.get("currency", "USD"),
        "pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg": info.get("pegRatio") or info.get("trailingPegRatio"),
        "pb": info.get("priceToBook"),
        "ps": info.get("priceToSalesTrailing12Months"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "net_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        "gross_margin": info.get("grossMargins"),
        "eps_growth_q": info.get("earningsQuarterlyGrowth"),
        "revenue_growth": info.get("revenueGrowth"),
        "debt_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        "free_cashflow": info.get("freeCashflow"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "_source": "yahoo",
    }

    # Convert decimals to percentages
    for k in ("roe", "roa", "net_margin", "operating_margin", "gross_margin",
              "eps_growth_q", "revenue_growth", "dividend_yield"):
        if fund.get(k) is not None:
            fund[k] = round(fund[k] * 100, 2)
    if fund.get("debt_equity") and fund["debt_equity"] > 5:
        fund["debt_equity"] = fund["debt_equity"] / 100

    return fund, sector


def yahoo_fetch_earnings(tk, years=5):
    """Fetch earnings history from yfinance."""
    try:
        ed = tk.get_earnings_dates(limit=years * 4 + 5)
        if ed is None or len(ed) == 0:
            return None
        ed = ed.copy()
        ed.reset_index(inplace=True)
        col_map = {
            "Earnings Date": "date",
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_actual",
            "Surprise(%)": "surprise_pct",
        }
        for old, new in col_map.items():
            if old in ed.columns:
                ed.rename(columns={old: new}, inplace=True)
        if "eps_actual" in ed.columns:
            ed = ed[ed["eps_actual"].notna()].copy()
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=years * 365)
        if "date" in ed.columns:
            try:
                if hasattr(ed["date"].iloc[0], "tz") and ed["date"].iloc[0].tz is not None:
                    ed["date"] = ed["date"].dt.tz_localize(None)
            except Exception:
                pass
            ed = ed[ed["date"] >= cutoff].copy()
        if len(ed) == 0:
            return None
        ed = ed.sort_values("date").reset_index(drop=True)
        cols = ["date", "eps_actual"]
        if "eps_estimate" in ed.columns:
            cols.append("eps_estimate")
        else:
            ed["eps_estimate"] = np.nan
            cols.append("eps_estimate")
        return ed[cols]
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# Alpha Vantage Client
# ═══════════════════════════════════════════════════════════════════
import urllib.request
import urllib.parse

def _av_request(params, api_name="alpha_vantage"):
    """Make Alpha Vantage API request with rate limiting + quota check."""
    if not ALPHA_VANTAGE_KEY:
        return None, "No API key configured"

    ok, msg = _check_quota("alpha_vantage", max_per_day=25)  # free tier
    if not ok:
        return None, msg

    AV_PER_MIN.wait_if_needed()
    params["apikey"] = ALPHA_VANTAGE_KEY
    url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FESP/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        _record_request("alpha_vantage")

        # Handle AV-specific errors
        if "Note" in data:
            return None, f"AV rate-limit: {data['Note'][:80]}"
        if "Information" in data and "rate" in data.get("Information", "").lower():
            return None, f"AV quota: {data['Information'][:80]}"
        if "Error Message" in data:
            return None, f"AV error: {data['Error Message']}"

        return data, "OK"
    except Exception as e:
        return None, f"AV request failed: {e}"


def alpha_vantage_overview(ticker):
    """
    Fetch comprehensive fundamentals via Alpha Vantage OVERVIEW endpoint.
    Includes 50+ fundamentals normalized to GAAP/IFRS.
    Returns the AV's much-richer data dict, or None.
    """
    cache_k = _cache_key("av_overview", ticker)
    cached = _cache_get(cache_k, max_age_hours=24 * 7)  # 1 week cache for fundamentals
    if cached:
        return cached

    data, msg = _av_request({"function": "OVERVIEW", "symbol": ticker})
    if data is None or len(data) < 5:
        return None

    # Normalize keys to FESP format
    def _f(k, factor=1.0):
        v = data.get(k)
        if v is None or v in ("None", "-", "N/A", ""):
            return None
        try:
            return float(v) * factor
        except (TypeError, ValueError):
            return None

    fund = {
        "name": data.get("Name", ""),
        "currency": data.get("Currency", "USD"),
        "exchange": data.get("Exchange", ""),
        "sector_av": data.get("Sector", ""),
        "industry": data.get("Industry", ""),
        "country": data.get("Country", ""),
        "market_cap": _f("MarketCapitalization"),
        "ebitda": _f("EBITDA"),

        # Valuation
        "pe": _f("PERatio"),
        "forward_pe": _f("ForwardPE"),
        "peg": _f("PEGRatio"),
        "pb": _f("PriceToBookRatio"),
        "ps": _f("PriceToSalesRatioTTM"),
        "ev_revenue": _f("EVToRevenue"),
        "ev_ebitda": _f("EVToEBITDA"),

        # Profitability (already in % from AV)
        "roe": _f("ReturnOnEquityTTM", 100),  # AV gives decimal, we want %
        "roa": _f("ReturnOnAssetsTTM", 100),
        "net_margin": _f("ProfitMargin", 100),
        "operating_margin": _f("OperatingMarginTTM", 100),

        # Growth (already in %)
        "eps_growth_q": _f("QuarterlyEarningsGrowthYOY", 100),
        "revenue_growth": _f("QuarterlyRevenueGrowthYOY", 100),

        # Dividends
        "dividend_yield": _f("DividendYield", 100),
        "dividend_per_share": _f("DividendPerShare"),
        "payout_ratio": _f("PayoutRatio"),

        # Per-share
        "eps": _f("EPS"),
        "book_value": _f("BookValue"),
        "revenue_per_share": _f("RevenuePerShareTTM"),

        # Other
        "beta": _f("Beta"),
        "52w_high": _f("52WeekHigh"),
        "52w_low": _f("52WeekLow"),
        "ma_50d": _f("50DayMovingAverage"),
        "ma_200d": _f("200DayMovingAverage"),
        "shares_outstanding": _f("SharesOutstanding"),
        "analyst_target": _f("AnalystTargetPrice"),

        "_source": "alpha_vantage",
    }
    _cache_set(cache_k, fund)
    return fund


def alpha_vantage_earnings(ticker, years=5):
    """
    Fetch earnings history with surprises via Alpha Vantage EARNINGS endpoint.
    Returns DataFrame with columns: date, eps_actual, eps_estimate, surprise, surprise_pct
    """
    cache_k = _cache_key("av_earnings", ticker, years)
    cached = _cache_get(cache_k, max_age_hours=24)
    if cached:
        try:
            df = pd.DataFrame(cached)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass

    data, msg = _av_request({"function": "EARNINGS", "symbol": ticker})
    if data is None or "quarterlyEarnings" not in data:
        return None

    quarterly = data["quarterlyEarnings"]
    if not quarterly:
        return None

    rows = []
    cutoff = datetime.now() - timedelta(days=years * 365)
    for q in quarterly:
        try:
            date = pd.to_datetime(q.get("reportedDate"))
            if date < cutoff:
                continue
            rows.append({
                "date": date,
                "eps_actual": float(q.get("reportedEPS")) if q.get("reportedEPS") not in (None, "None", "-") else None,
                "eps_estimate": float(q.get("estimatedEPS")) if q.get("estimatedEPS") not in (None, "None", "-") else None,
                "surprise": float(q.get("surprise")) if q.get("surprise") not in (None, "None", "-") else None,
                "surprise_pct": float(q.get("surprisePercentage")) if q.get("surprisePercentage") not in (None, "None", "-") else None,
            })
        except (TypeError, ValueError):
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    _cache_set(cache_k, df.to_dict(orient="records"))
    return df


def alpha_vantage_news_sentiment(ticker, limit=50):
    """
    Fetch news sentiment via Alpha Vantage NEWS_SENTIMENT (Alpha Intelligence).
    Returns aggregated sentiment metrics + recent news topics.
    """
    cache_k = _cache_key("av_news", ticker, limit)
    cached = _cache_get(cache_k, max_age_hours=6)  # news is more volatile
    if cached:
        return cached

    data, msg = _av_request({
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": limit,
    })
    if data is None or "feed" not in data:
        return None

    feed = data.get("feed", [])
    if not feed:
        return None

    sentiments = []
    relevances = []
    for article in feed:
        for ts in article.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker:
                try:
                    sentiments.append(float(ts.get("ticker_sentiment_score", 0)))
                    relevances.append(float(ts.get("relevance_score", 0)))
                except (TypeError, ValueError):
                    pass

    if not sentiments:
        return None

    sentiments = np.array(sentiments)
    relevances = np.array(relevances)
    weighted_sentiment = float(np.average(sentiments, weights=relevances)) \
        if relevances.sum() > 0 else float(sentiments.mean())

    # Classify
    if weighted_sentiment >= 0.35:
        label = "BULLISH"
    elif weighted_sentiment >= 0.15:
        label = "SOMEWHAT_BULLISH"
    elif weighted_sentiment >= -0.15:
        label = "NEUTRAL"
    elif weighted_sentiment >= -0.35:
        label = "SOMEWHAT_BEARISH"
    else:
        label = "BEARISH"

    result = {
        "n_articles": len(sentiments),
        "weighted_sentiment": round(weighted_sentiment, 4),
        "mean_sentiment": round(float(sentiments.mean()), 4),
        "max_sentiment": round(float(sentiments.max()), 4),
        "min_sentiment": round(float(sentiments.min()), 4),
        "label": label,
        "_source": "alpha_vantage_news",
    }
    _cache_set(cache_k, result)
    return result


def alpha_vantage_insider_transactions(ticker):
    """
    Fetch insider transactions (Alpha Intelligence).
    Returns aggregate stats for last 90 days.
    """
    cache_k = _cache_key("av_insider", ticker)
    cached = _cache_get(cache_k, max_age_hours=24)
    if cached:
        return cached

    data, msg = _av_request({"function": "INSIDER_TRANSACTIONS", "symbol": ticker})
    if data is None or "data" not in data:
        return None

    transactions = data.get("data", [])
    if not transactions:
        return None

    # Aggregate last 90 days
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
    total_buys = 0
    total_sells = 0
    n_buys = 0
    n_sells = 0
    for t in transactions:
        try:
            t_date = pd.to_datetime(t.get("transaction_date"))
            if t_date < cutoff:
                continue
            shares = float(t.get("shares", 0))
            price = float(t.get("share_price", 0))
            value = shares * price
            if t.get("acquisition_or_disposal") == "A":
                total_buys += value
                n_buys += 1
            else:
                total_sells += value
                n_sells += 1
        except (TypeError, ValueError):
            continue

    result = {
        "n_buys_90d": n_buys,
        "n_sells_90d": n_sells,
        "total_buy_value": round(total_buys, 0),
        "total_sell_value": round(total_sells, 0),
        "net_flow": round(total_buys - total_sells, 0),
        "buy_sell_ratio": round(total_buys / total_sells, 2) if total_sells > 0 else None,
        "signal": "INSIDER_BUYING" if (total_buys > total_sells * 1.5) else
                  "INSIDER_SELLING" if (total_sells > total_buys * 1.5) else "NEUTRAL",
        "_source": "alpha_vantage_insider",
    }
    _cache_set(cache_k, result)
    return result


def alpha_vantage_earnings_calendar(ticker, horizon="3month"):
    """
    Fetch upcoming earnings dates via Alpha Vantage EARNINGS_CALENDAR.
    horizon: '3month', '6month', or '12month'
    """
    cache_k = _cache_key("av_calendar", ticker, horizon)
    cached = _cache_get(cache_k, max_age_hours=24)
    if cached:
        return cached

    # CSV endpoint requires direct request
    if not ALPHA_VANTAGE_KEY:
        return None
    ok, msg = _check_quota("alpha_vantage", max_per_day=25)
    if not ok:
        return None

    AV_PER_MIN.wait_if_needed()
    url = (f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR"
           f"&symbol={ticker}&horizon={horizon}&apikey={ALPHA_VANTAGE_KEY}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FESP/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            csv_text = r.read().decode()
        _record_request("alpha_vantage")

        if not csv_text or "symbol" not in csv_text.lower():
            return None

        from io import StringIO
        df = pd.read_csv(StringIO(csv_text))
        if df is None or len(df) == 0:
            return None

        # Parse next earnings
        df["reportDate"] = pd.to_datetime(df["reportDate"])
        next_earn = df.iloc[0] if len(df) > 0 else None
        if next_earn is None:
            return None

        days_to = (next_earn["reportDate"] - pd.Timestamp.now()).days
        result = {
            "next_earnings_date": str(next_earn["reportDate"].date()),
            "days_to_next": int(days_to),
            "fiscal_date_ending": str(next_earn.get("fiscalDateEnding", "")),
            "estimate": float(next_earn["estimate"]) if pd.notna(next_earn.get("estimate")) else None,
            "currency": next_earn.get("currency", "USD"),
            "_source": "alpha_vantage_calendar",
        }
        _cache_set(cache_k, result)
        return result
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# Marketstack Client
# ═══════════════════════════════════════════════════════════════════
def _ms_request(endpoint, params):
    """Make Marketstack API request with rate limiting + quota check."""
    if not MARKETSTACK_KEY:
        return None, "No API key configured"

    ok, msg = _check_quota("marketstack", max_per_month=100)  # free tier
    if not ok:
        return None, msg

    MS_PER_SEC.wait_if_needed()
    params["access_key"] = MARKETSTACK_KEY
    url = f"https://api.marketstack.com/v2/{endpoint}?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FESP/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        _record_request("marketstack")

        if "error" in data:
            return None, f"MS error: {data['error'].get('message', 'unknown')}"
        return data, "OK"
    except Exception as e:
        return None, f"MS request failed: {e}"


def marketstack_eod(ticker, days=365):
    """
    Fetch end-of-day prices from Marketstack.
    Use for international tickers where Yahoo data is poor.
    """
    cache_k = _cache_key("ms_eod", ticker, days)
    cached = _cache_get(cache_k, max_age_hours=24)
    if cached:
        try:
            df = pd.DataFrame(cached)
            df.index = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass

    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    data, msg = _ms_request("eod", {
        "symbols": ticker,
        "date_from": date_from,
        "limit": 1000,
    })
    if data is None or "data" not in data:
        return None

    rows = data["data"]
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Standardize column names to match yfinance
    rename_map = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
        "adj_open": "AdjOpen", "adj_high": "AdjHigh", "adj_low": "AdjLow",
        "adj_close": "AdjClose", "adj_volume": "AdjVolume",
    }
    df = df.rename(columns=rename_map)
    # Use adjusted close as primary
    if "AdjClose" in df.columns:
        df["Close"] = df["AdjClose"]

    _cache_set(cache_k, df.to_dict(orient="records"))
    df.index = df["date"]
    return df


def marketstack_ticker_info(ticker):
    """Fetch ticker metadata via Marketstack tickerinfo endpoint."""
    cache_k = _cache_key("ms_info", ticker)
    cached = _cache_get(cache_k, max_age_hours=24 * 7)
    if cached:
        return cached

    data, msg = _ms_request("tickerinfo", {"ticker": ticker})
    if data is None or "data" not in data:
        return None

    info = data["data"]
    result = {
        "name": info.get("name"),
        "ticker": info.get("ticker"),
        "exchange": info.get("stock_exchange", {}).get("acronym"),
        "country": info.get("stock_exchange", {}).get("country"),
        "currency": info.get("stock_exchange", {}).get("currency_code"),
        "isin": info.get("isin"),
        "cik": info.get("cik"),
        "_source": "marketstack",
    }
    _cache_set(cache_k, result)
    return result


# ═══════════════════════════════════════════════════════════════════
# Hybrid Smart Fetcher — uses best available source per data type
# ═══════════════════════════════════════════════════════════════════
def _merge_fundamentals(yahoo_fund, av_fund):
    """
    Merge Yahoo + Alpha Vantage fundamentals.
    Strategy: prefer AV when available (better quality), fall back to Yahoo.
    """
    if not av_fund:
        return yahoo_fund
    if not yahoo_fund:
        yahoo_fund = {}

    merged = dict(yahoo_fund)
    # AV takes precedence for these high-value fields
    av_priority_fields = [
        "pe", "forward_pe", "peg", "pb", "ps", "roe", "roa", "net_margin",
        "operating_margin", "eps_growth_q", "revenue_growth", "beta",
        "52w_high", "52w_low", "shares_outstanding", "analyst_target",
        "ev_revenue", "ev_ebitda", "payout_ratio", "eps", "book_value",
        "ebitda", "ma_50d", "ma_200d",
    ]
    for k in av_priority_fields:
        if av_fund.get(k) is not None:
            merged[k] = av_fund[k]

    # AV-specific fields that Yahoo doesn't have
    for k in ("ev_revenue", "ev_ebitda", "ebitda", "ma_50d", "ma_200d",
              "analyst_target", "payout_ratio", "industry", "country"):
        if av_fund.get(k) is not None:
            merged[k] = av_fund[k]

    merged["_sources"] = [yahoo_fund.get("_source", "yahoo"),
                          av_fund.get("_source", "alpha_vantage")]
    return merged


def fetch_complete_ticker(ticker, deep=False, deep_mode="full"):
    """
    Master fetcher — uses optimal data sources for each data type.

    Args:
        ticker: stock symbol
        deep: if True, use Alpha Vantage for enrichment
        deep_mode: 'full' (5 endpoints) | 'lite' (2 endpoints) | 'fundamentals_only' (1 endpoint)
            - full: OVERVIEW + EARNINGS + NEWS + INSIDER + CALENDAR (5 reqs)
            - lite: OVERVIEW + EARNINGS (2 reqs) — best fundamentals for budget
            - fundamentals_only: OVERVIEW only (1 req) — bare minimum AV enrichment

    Returns dict with available data.
    """
    result = {
        "ticker": ticker,
        "sources_used": [],
        "fetch_started_at": datetime.now(timezone.utc).isoformat(),
    }

    # Step 1: Yahoo for price + base fundamentals (free, unlimited)
    tk, hist = yahoo_fetch_price(ticker)
    if hist is None:
        # Fall back to Marketstack
        if MARKETSTACK_KEY:
            ms_hist = marketstack_eod(ticker, days=365 * 2)
            if ms_hist is not None:
                result["price_history"] = ms_hist
                result["sources_used"].append("marketstack")
        if "price_history" not in result:
            return None
    else:
        result["price_history"] = hist
        result["sources_used"].append("yahoo_price")

    yahoo_fund, sector = yahoo_fetch_fundamentals(tk) if tk else ({}, "Unknown")
    yahoo_earnings = yahoo_fetch_earnings(tk, years=5) if tk else None
    if yahoo_fund:
        result["sources_used"].append("yahoo_fundamentals")

    # Step 2: Alpha Vantage enrichment (only if deep mode + key configured)
    av_fund = None
    av_earnings = None
    news = None
    insider = None
    calendar = None

    if deep and ALPHA_VANTAGE_KEY:
        # Always fetch OVERVIEW (best fundamentals)
        av_fund = alpha_vantage_overview(ticker)
        if av_fund:
            result["sources_used"].append("alpha_vantage_overview")

        if deep_mode in ("full", "lite"):
            av_earnings = alpha_vantage_earnings(ticker, years=5)
            if av_earnings is not None:
                result["sources_used"].append("alpha_vantage_earnings")

        if deep_mode == "full":
            news = alpha_vantage_news_sentiment(ticker)
            if news:
                result["sources_used"].append("alpha_vantage_news")

            insider = alpha_vantage_insider_transactions(ticker)
            if insider:
                result["sources_used"].append("alpha_vantage_insider")

            calendar = alpha_vantage_earnings_calendar(ticker)
            if calendar:
                result["sources_used"].append("alpha_vantage_calendar")

    # Merge fundamentals (AV preferred, Yahoo fallback)
    result["fundamentals"] = _merge_fundamentals(yahoo_fund, av_fund)

    # Use AV earnings if available (better data with surprises)
    if av_earnings is not None and len(av_earnings) > 0:
        result["earnings_history"] = av_earnings
    else:
        result["earnings_history"] = yahoo_earnings

    # Sector from AV if Yahoo didn't have it
    if sector == "Unknown" and av_fund and av_fund.get("sector_av"):
        sector = av_fund["sector_av"]
    result["sector"] = sector

    # Optional enrichments
    if news:
        result["news_sentiment"] = news
    if insider:
        result["insider_signal"] = insider
    if calendar:
        result["upcoming_earnings"] = calendar

    return result


# ═══════════════════════════════════════════════════════════════════
# Quota status helper
# ═══════════════════════════════════════════════════════════════════
def quota_status():
    """Returns current quota usage."""
    usage = _load_usage()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    av_today = usage.get("alpha_vantage", {}).get("daily", {}).get(today, 0)
    ms_month = usage.get("marketstack", {}).get("monthly", {}).get(month, 0)
    return {
        "alpha_vantage": {
            "today": av_today,
            "limit_free_tier": 25,
            "remaining": max(0, 25 - av_today),
            "key_configured": bool(ALPHA_VANTAGE_KEY),
        },
        "marketstack": {
            "month": ms_month,
            "limit_free_tier": 100,
            "remaining": max(0, 100 - ms_month),
            "key_configured": bool(MARKETSTACK_KEY),
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("API CLIENTS — Self Test")
    print("=" * 60)

    print("\nKeys configured:")
    print(f"  Alpha Vantage: {'YES' if ALPHA_VANTAGE_KEY else 'NO (set ALPHA_VANTAGE_API_KEY)'}")
    print(f"  Marketstack:   {'YES' if MARKETSTACK_KEY else 'NO (set MARKETSTACK_API_KEY)'}")

    print("\nQuota status:")
    q = quota_status()
    for api, info in q.items():
        if info["key_configured"]:
            limit_field = "today" if api == "alpha_vantage" else "month"
            print(f"  {api}: {info[limit_field]}/{info['limit_free_tier']} used "
                  f"({info['remaining']} remaining)")

    print("\nTrying Yahoo fetch for AAPL...")
    tk, hist = yahoo_fetch_price("AAPL")
    if hist is not None:
        print(f"  ✓ Got {len(hist)} days of price data, last close: ${hist['Close'].iloc[-1]:.2f}")
        fund, sector = yahoo_fetch_fundamentals(tk)
        print(f"  ✓ Sector: {sector}")
        print(f"  ✓ P/E: {fund.get('pe')}, ROE: {fund.get('roe')}%")
    else:
        print("  ✗ Yahoo fetch failed (sandbox limitation or network)")

    if ALPHA_VANTAGE_KEY:
        print("\nTrying Alpha Vantage OVERVIEW for AAPL...")
        av = alpha_vantage_overview("AAPL")
        if av:
            print(f"  ✓ AV data: P/E={av.get('pe')}, ROE={av.get('roe')}%")
            print(f"  ✓ Industry: {av.get('industry')}")
            print(f"  ✓ Analyst Target: ${av.get('analyst_target')}")
        else:
            print("  ✗ AV failed (key invalid, quota exhausted, or network)")

    if MARKETSTACK_KEY:
        print("\nTrying Marketstack EOD for AAPL (will use 1 of 100 monthly requests)...")
        # Don't actually run unless explicitly testing — saves quota
        print("  (skipped to save quota; uncomment to test)")
