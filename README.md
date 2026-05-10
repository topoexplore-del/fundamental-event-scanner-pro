# Fundamental Event Scanner Pro

Multi-source fundamental analysis system combining **Yahoo Finance + Alpha Vantage + Marketstack** with stochastic projections, event study, and Monte Carlo integration.

## What it does

For each ticker:

1. **Fetches data** from up to 3 APIs (Yahoo unlimited, AV deep enrichment, Marketstack fallback)
2. **Computes Quality Score** (0-100) across 5 dimensions:
   - Valuation (P/E, PEG, P/B vs sector)
   - Profitability (ROE, ROA, margins)
   - Growth (EPS, revenue)
   - Stability (debt, consistency)
   - Earnings momentum (recent surprises)
3. **Runs Event Study** on 5 years of quarterly earnings:
   - T+1, T+5, T+21, T+63 reaction windows
   - Beat vs miss differential
   - PEAD (Post-Earnings Announcement Drift) detection
4. **Stochastic Projection Engine (SPE)** — Monte Carlo + GARCH(1,1) + t-student
5. **4 MC Integration Patterns:**
   - Pattern 1: Independent (side-by-side)
   - Pattern 2: Event-conditional MC (bimodal for upcoming earnings)
   - Pattern 3: Quality-adjusted drift
   - Pattern 4: Master FESP Score (heuristic ranking)
6. **Alpha Intelligence enrichment** (deep mode):
   - News sentiment (weighted across articles)
   - Insider transactions (90-day buy/sell flow)
   - Upcoming earnings calendar
7. **Bloomberg-style dense table** displays everything

## Honesty Disclaimers

- **Quality Score does NOT predict the future.** It scores past quality, which has weak short-term correlation with returns.
- **Yahoo Finance fundamentals have errors and gaps.** Alpha Vantage is more reliable but rate-limited (25/day free).
- **Marketstack does NOT provide fundamentals** — it's a price-only API. Used here for international ticker fallback.
- **Event study assumes future reactions resemble past.** Regime changes invalidate this.
- **The MC integration patterns are MODELS, not predictions.** Each has documented limitations.

## API Key Setup (CRITICAL — security)

**Never hardcode API keys in code.**

```bash
# Copy the template
cp .env.example .env

# Edit .env with your keys (NEVER commit this file)
nano .env
```

Your `.env` file should look like:
```
ALPHA_VANTAGE_API_KEY=your_real_key_here
MARKETSTACK_API_KEY=your_real_key_here
```

The `.gitignore` ensures `.env` never leaves your computer. **If you ever expose these keys (in chat, screenshots, GitHub commits), regenerate them immediately:**
- Alpha Vantage: https://www.alphavantage.co/support/#api-key
- Marketstack: https://marketstack.com/dashboard

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys (optional — Yahoo works without keys)
cp .env.example .env
# Edit .env with your keys

# 3. Generate snapshot
python scripts/build_data.py --out-dir data

# 4. Or generate demo (no APIs needed)
python scripts/generate_demo_snapshot.py

# 5. Preview locally
python -m http.server 8000
# Open http://localhost:8000
```

## How free-tier API quotas are managed

The system is designed to work even with the most restrictive free tiers:

| API | Free tier | Strategy |
|---|---|---|
| Yahoo Finance | Unlimited (unofficial) | Primary source for ALL tickers |
| Alpha Vantage | **25 req/day** | Used selectively for "deep" tickers |
| Marketstack | **100 req/month** | Fallback only when Yahoo fails |

### Watchlist-based deep enrichment

You define which tickers get the AV-deep treatment in **`watchlist.txt`**. Edit that file to choose your priorities. The system reads it on every run and applies budget-aware selection.

### Deep modes (optimize for free tier)

```bash
# Mode "full" (5 AV requests per ticker): OVERVIEW + EARNINGS + NEWS + INSIDER + CALENDAR
python scripts/build_data.py --deep-mode full --deep-top-n 5
# → 5 tickers × 5 reqs = 25/25 daily quota

# Mode "lite" (2 AV requests per ticker): OVERVIEW + EARNINGS
python scripts/build_data.py --deep-mode lite --deep-top-n 12
# → 12 tickers × 2 reqs = 24/25 daily quota

# Mode "fundamentals_only" (1 AV request per ticker): OVERVIEW only
python scripts/build_data.py --deep-mode fundamentals_only --deep-top-n 25
# → 25 tickers × 1 req = 25/25 daily quota

# Auto-budget (let the system decide based on remaining quota)
python scripts/build_data.py --deep-mode lite
# → reads watchlist.txt, fits as many as quota allows

# No AV at all — Yahoo only
python scripts/build_data.py --deep-top-n 0
```

### Recommended config for plan free

For most people on free tier, use **`lite` mode with 12 tickers**:
- Captures the highest-value AV data (deep fundamentals + earnings with surprises)
- Skips news sentiment and insider data (require full mode)
- Uses 24 of 25 daily AV requests, leaving 1 in reserve

The default GitHub Action workflow uses this config.

## Files

```
fundamental-event-scanner-pro/
├── index.html                          # Bloomberg-style dashboard
├── README.md
├── requirements.txt
├── .env.example                        # Template (safe to commit)
├── .env                                # Your real keys (gitignored)
├── .gitignore                          # CRITICAL — protects .env from leaking
├── data/
│   └── snapshot.json                   # Generated output
└── scripts/
    ├── stochastic_projector.py         # SPE engine (GARCH + MC + t-student)
    ├── quality_score.py                # Multifactor 0-100 score
    ├── event_study.py                  # Earnings reaction analysis
    ├── mc_integration.py               # 4 integration patterns
    ├── api_clients.py                  # Yahoo + AV + Marketstack with rate limiting
    ├── build_data.py                   # Main pipeline
    └── generate_demo_snapshot.py       # Synthetic demo (no APIs)
```

## Reading the Dashboard

The Bloomberg-style table shows 37 columns per ticker:

**Identification** (sticky left): Ticker, Sector, Price, Market Cap

**Scores**: FESP (master 0-100, A/B+/B/C/D), QS (Quality 0-100)

**Valuation**: P/E, Forward P/E, PEG, P/B
- Green = cheap relative to sector
- Red = expensive (mean reversion risk)

**Profitability**: ROE, ROA, Net Margin
- Green = strong; bold green = excellent

**Growth**: EPS Q (quarterly YoY), EPS 5Y (annualized), Revenue Q

**Stability**: Debt/Equity, Current Ratio, Dividend Yield, Beta

**Event Study**: Beat Rate, T+5 after Beat, T+5 after Miss, PEAD drift, Last 4Q beats

**SPE 21d Projection**: Prob UP, Target, Upside %, VaR 95%, Volatility

**Event-conditional MC**: Days to earnings, Beat/Miss/Weighted expected prices

**Alpha Intelligence** (only for deep-enriched tickers):
- SENT: News sentiment label (BULLISH / SOMEWHAT_BULLISH / NEUTRAL / SOMEWHAT_BEARISH / BEARISH)
- INSIDER: BUY / SELL / NEU based on 90-day insider transaction flow
- NEXT ER: Days until next earnings + date

## The 4 MC Integration Patterns explained

**Pattern 1 — Independent**: Just runs SPE and Quality Score side-by-side. No coupling.

**Pattern 2 — Event-conditional MC** (most rigorous): When earnings is < 42 days away:
- Computes historical beat_rate from event study
- Runs TWO Monte Carlos:
  - "If BEAT" — drift = avg_t5_after_beat (historical post-beat reaction)
  - "If MISS" — drift = avg_t5_after_miss
- Weights terminal distributions by beat_rate
- Result is **bimodal**, capturing the binary nature of earnings events

**Pattern 3 — Quality-adjusted drift**: Adjusts MC drift based on Quality Score.
- QS=100 → +4% annual drift adjustment (quality factor ~Fama-French)
- QS=50 → 0% (neutral)
- QS=0 → -2% (penalize weak quality)
- Use 63-day horizon — short horizons are noise-dominated.

**Pattern 4 — Master FESP Score**: Heuristic ranking 0-100:
- 40% Quality Score
- 20% Earnings momentum
- 40% SPE prob_up
- A grade ≥75, B+ ≥65, B ≥55, C ≥45, D <45

## Limitations

1. **Yahoo data quality varies wildly** by ticker, especially internationally
2. **Free-tier rate limits** make full daily enrichment impossible
3. **Earnings estimates** (for surprise calculation) are sparse for non-US tickers
4. **Event study assumes regime stability** — fails during crises
5. **No analyst targets** without paid AV plan
6. **No real-time data** — all snapshots are end-of-day

## License

No warranty. Not financial advice. Past quality does not predict future returns. Use as ONE input among many in your investment process.
