# Cross-Asset Market Monitor and Signal Engine

A single market-data spine feeding three outputs: systematic cross-asset
trade signals with backtested performance, a fixed-income yield-curve
relative-value screen, and an auto-generated daily HTML market briefing.

**Status: Phases 1–5 complete** (data spine + signals + curve/RV + backtester +
daily briefing). Only the GitHub Actions automation (Phase 6) remains.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv`). Python 3.11
is pinned via `.python-version` and installed automatically by `uv`.

```bash
uv sync                       # installs pinned dependencies into .venv
cp .env.example .env          # then fill in the two keys below
uv run pytest                 # run the test suite
```

### API keys

| Variable | Where to get it | Notes |
|---|---|---|
| `FRED_API_KEY` | https://fred.stlouisfed.org/docs/api/api_key.html (free, instant) | Must be a 32-char lowercase alphanumeric string |
| `ALPHAVANTAGE_API_KEY` | https://www.alphavantage.co/support/#api-key (free, instant) | Free tier is capped at **25 requests/day total** |
| `ANTHROPIC_API_KEY` | optional, only for the future LLM narrative pass | not used yet |

Put real values in `.env` (gitignored). Never commit `.env`. In CI, the same
variable names are populated from GitHub Actions secrets (see Phase 6).

## Running the Phase 1 data-pull check

```bash
uv run python scripts/pull_data_summary.py
```

Pulls the full Treasury curve, both credit-spread series, the equity
basket, and the FX basket, and prints latest level + day-over-day change
for every configured series. Fails loudly (non-zero exit) if any series is
missing, empty, or has a gap larger than `max_gap_calendar_days` in
`config/params.yaml` -- it does not substitute placeholder data.

## Architecture

```
cross_asset_engine/
  data/            # one loader per vendor + a unified MarketData accessor
  config/          # instruments.yaml (universe), params.yaml (tunables)
  settings.py      # loads the two YAML config files
  signals/         # (Phase 2) momentum, carry, regime
  curve/           # (Phase 3) bootstrap, DV01, RV screens
  backtest/        # (Phase 4) engine, metrics
  briefing/        # (Phase 5) HTML + JSON output
tests/
scripts/
  pull_data_summary.py   # Phase 1 acceptance script
  run_daily.py           # (Phase 5+) full pipeline entrypoint
```

Nothing outside `cross_asset_engine/data/market_data.py` is allowed to
import a vendor loader directly or hardcode a series ID / ticker -- signals,
the curve module, and the briefing only ever call `MarketData`. This is
what makes swapping a vendor (or adding a new asset class) a change to one
file instead of a scavenger hunt.

Every vendor loader returns the same tidy schema: columns
`date, asset_id, value, source`. `MarketData` adds an `asset_class` column
(`rates` / `credit` / `equities` / `fx`) and runs a gap check before
returning data to callers.

### Data sources (confirmed live 2026-07-08)

- **Rates**: FRED constant-maturity Treasury par yields --
  `DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30`.
- **Credit**: FRED ICE BofA OAS -- `BAMLC0A0CM` (investment grade),
  `BAMLH0A0HYM2` (high yield). **Caveat:** FRED now caps ICE BofA index
  series to a rolling ~3-year history (a 2026 policy change) -- full-depth
  history for these two series is not available for free. Credit-carry
  backtests will be correspondingly short-sample; this is called out again
  wherever it matters (Phase 4).
- **Equities**: FRED US index levels -- `SP500`, `NASDAQCOM`, `DJIA`.
  Sourced from FRED, *not* Alpha Vantage: as of a 2026 change, Alpha
  Vantage's `TIME_SERIES_DAILY` restricts `outputsize=full` to premium, so
  the free tier caps equity history at ~100 trading days -- too short for
  126/252-day momentum or multi-year backtests. FRED serves 10+ years of
  daily index levels for free (`NASDAQCOM` back to 1990; `SP500`/`DJIA`
  capped at ~10yr by S&P/Dow licensing). These are *price* indices
  (dividend-excluding), noted where it matters for momentum. No clean free
  daily source exists for developed-ex-US / EM indices, so the equity
  universe is US-only.
- **FX**: Alpha Vantage `FX_DAILY` on four major pairs (EURUSD, USDJPY,
  GBPUSD, AUDUSD). Unlike the equity endpoint, `FX_DAILY` still serves
  `outputsize=full` (5000 points, back to ~2007) on the free tier.

The full endpoint verification (base URLs, param names, response shapes,
current rate limits) was done by hitting both vendors' live APIs directly
rather than trusting cached knowledge of their docs -- see git history on
`cross_asset_engine/data/fred.py` and `alphavantage.py` for the reasoning
in the module docstrings.

### Caching

Every raw vendor response is cached to disk under `cache/<source>/<key>/<as_of_date>.json`
before parsing. This means: (a) re-running the pipeline the same day never
re-hits a rate-limited API for data it already has, and (b) a bug found in
the parsing code can be fixed and replayed against the exact bytes that were
originally fetched, which is what "reproducible" means here.

### Failure behavior

Per the project's core constraint, missing or broken data is a loud failure,
never a silent substitution:

- A vendor HTTP error, malformed JSON, or an explicit rate-limit/error
  payload raises `DataFetchError`.
- A series with a larger-than-expected gap between observations (beyond a
  configurable holiday/weekend allowance) raises `DataGapError`.
- Non-numeric values in a response raise `DataFetchError` rather than being
  coerced or dropped silently.

## Signals (Phase 2)

Every signal emits a standardized `SignalResult` (asset, date, score,
direction, and the inputs it was built from) and self-registers in
`SIGNAL_REGISTRY`, so the backtester and briefing iterate over them
generically. Run `uv run python scripts/signals_summary.py` for a live read.

- **Momentum** (`signals/momentum.py`) — time-series (sign of an asset's own
  trailing return) and cross-sectional (z-score across the group) at 63/126/252-day
  lookbacks. Equities and FX run on price/rate levels directly. Rates run on a
  bond **total-return proxy** built from yields (carry − modified-duration × Δyield),
  so a *falling* yield correctly reads as a *rising* bond price / positive
  long-duration momentum. Modified duration is the analytic par-bond value here;
  Phase 3 swaps in the exact bootstrapped DV01.
- **Carry** (`signals/carry.py`) — FX = short-rate differential (r_base − r_quote);
  credit = the OAS itself; rates = yield + a rolldown **hook** (Phase 3 fills
  rolldown from the curve); equity = dividend yield − financing, a documented
  hook that returns nothing rather than fabricate the (unsourced) dividend yield.
  `apply_regime_scaling` multiplies carry size by the regime multiplier.
- **Regime** (`signals/regime.py`) — ported from the Perihelion volatility
  project. Two methods: (A) z-score of EWMA realized vol vs a trailing year,
  bucketed LOW/NORMAL/HIGH/CRISIS; (B) a self-contained, deterministic 2-state
  Gaussian HMM (Baum-Welch EM) whose smoothed P(high-vol) maps to the same
  buckets. `current_regime()` gives the latest read; `carry_scale()` turns it
  into the exposure multiplier — carry is scaled down as stress rises, since
  carry bleeds precisely in risk-off regimes.

## Yield curve & RV screens (Phase 3)

`uv run python scripts/curve_summary.py` bootstraps the live curve and runs
the screens.

- **Bootstrap** (`curve/bootstrap.py`) — interpolates the CMT par curve onto a
  semiannual grid (linear-on-yield default, cubic optional) and bootstraps
  discount factors short-to-long from the par condition, then derives zero and
  implied-forward curves. Correctness is pinned by a hand-computed test and the
  flat-par→flat-zero identity.
- **DV01** (`curve/dv01.py`) — model-free 1bp-bump valuation: `dv01_par_bond`
  and `dv01_from_cashflows` (bump the zero curve and reprice). This is the
  sizing primitive that makes the RV trades pure shape views.
- **RV screens** (`curve/rv_screens.py`) — slope (2s10s, 5s30s), butterfly
  (2s5s10s), and carry-and-rolldown, each z-scored vs a rolling window and
  sized DV01-neutral. Carry-and-rolldown ranks each point by total return per
  unit DV01 over a configurable horizon, and its rolldown feeds back into the
  Phase 2 rates-carry signal (the hook left open there is now closed).

All curve definitions (slope pairs, butterfly legs, horizon, interpolation,
z-score window) live in `config/params.yaml`.

## Backtester (Phase 4)

`uv run python scripts/backtest_report.py` backtests equity momentum, FX carry,
and a 2s10s curve trade, and writes `reports/equity_curves.png` +
`reports/backtest_snapshot.json`.

- **Engine** (`backtest/engine.py`) — pairs `position(t)` (past-only) with the
  *forward* return realized after t, sampled non-overlapping every `horizon`
  days. No-lookahead is structural: a position can only earn returns that
  happen after the data used to form it.
- **Metrics** (`backtest/metrics.py`) — hit rate, annualized return/vol, Sharpe,
  max drawdown, turnover. Transaction cost (bps) is applied to traded notional
  and **net-of-cost metrics are reported alongside gross**.
- **Anti-overfitting** — the headline uses a single lookback; the lookback
  *sweep* reports the spread of Sharpe across 63/126/252d rather than
  cherry-picking the best cell.
- **No-lookahead guard** (`tests/test_no_lookahead.py`) — pins the production
  momentum builder to an honest backward-only reference (fails if anyone makes
  it peek), and shows a deliberately leaked signal scoring an impossible ~100%
  hit rate as the canary.

Results are deliberately reported honestly (modest/mixed), not tuned to look
good. One caveat surfaced by the data: the equity-momentum sleeve spans
NASDAQ's full history (1990+) while the S&P/Dow series only start ~2016, so the
pre-2016 equity backtest is NASDAQ-weighted — a history-availability artifact,
not a modeling choice.

## Daily briefing (Phase 5)

`uv run python scripts/run_daily.py` runs the whole pipeline and writes a
**self-contained** `docs/index.html` (inline CSS, base64-embedded charts — no
external requests) plus a dated `docs/briefing_<date>.json` snapshot. Every
number in the HTML is reproduced in the JSON, so the briefing is fully
auditable. The page has a levels-and-moves table, the day's signals with their
backtested hit-rate/Sharpe, the curve snapshot with RV-screen highlights, and a
written narrative.

- **Narrative** (`briefing/narrative.py`) — rules-based templating by default:
  turns the computed numbers into desk-style prose (risk tone, per-asset movers,
  cross-asset synthesis with divergence flags, signals as trade ideas calibrated
  to their Sharpe, and the flagged curve RV trade). It states what the pattern is
  *consistent with*, never a cause — there is no news feed.
- **Optional LLM narrative** (`briefing/llm_narrative.py`, `--llm` flag) — an
  Anthropic pass over the *same* JSON. The system prompt forbids inventing
  figures or asserting causes, and the numbers are passed explicitly in the
  request; the model has no tools and fetches nothing. Requires `ANTHROPIC_API_KEY`;
  it fails loudly rather than silently falling back if the key is missing.

```bash
uv run python scripts/run_daily.py          # rules-based narrative (default)
uv run python scripts/run_daily.py --llm    # optional Anthropic narrative pass
```

## Testing

`uv run pytest` covers: tidy-schema validation and gap detection, dated-cache
hit/miss behavior, both vendor loaders against mocked HTTP responses
(including the FRED holiday-marker convention and Alpha Vantage's
rate-limit response shape), and the unified `MarketData` accessor end to
end against a small synthetic instrument universe.
