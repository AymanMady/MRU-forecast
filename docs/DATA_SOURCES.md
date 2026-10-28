# Data sources and target definition

This file is the contract of the project. Everything downstream — the models, the
intervals, the dashboard, the API — is only as meaningful as what is written here.

## 1. What is being forecast

| Target | Definition | Observed or derived | Rate type |
|---|---|---|---|
| `USD_MRU` | number of MRU required to buy **1 USD** | observed quote | indicative mid-market |
| `EUR_MRU` | number of MRU required to buy **1 EUR** | **derived cross** | indicative mid-market (derived) |

A rising number means the Ouguiya is **weaker**.

### EUR/MRU is a derived cross, not a quote

The primary provider publishes no usable history for EUR/MRU, so it is constructed:

```
EUR/MRU  =  (USD/MRU)  ×  (EUR/USD)
```

This is a legitimate and standard construction, but it is **not an observed price**.
It inherits the measurement noise of *both* legs, which is visible in the results:
the point models beat a random walk on USD/MRU at short horizons and do not manage
it on EUR/MRU at any horizon. The pair is labelled `is_derived_cross: true`
everywhere it is served, and nothing in the project presents it as a quote.

## 2. Providers

### Primary — Yahoo Finance chart endpoint

```
https://query1.finance.yahoo.com/v8/finance/chart/{symbol}
```

| Series | Symbol | History available |
|---|---|---|
| `USD_MRU` | `MRU=X` | from 2018-11-01 |
| `EUR_USD` | `EURUSD=X` | from 2003 |
| `BRENT_OIL` | `BZ=F` | from 2007 |
| `GOLD` | `GC=F` | from 2000 |

Chosen because it is the only source found that publishes a **daily MRU series with
multi-year history and requires no API key**. No account, no key, no rate-limit
negotiation — a reader can clone the repository and reproduce the dataset.

**Request the window explicitly.** With `range=max` the endpoint silently
downsamples to weekly bars: the same symbol returns 413 points instead of 2048.
Ingestion always passes `period1` / `period2` as epoch bounds.

### Cross-check — fawazahmed0/currency-api on jsDelivr

```
https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date}/v1/currencies/{base}.min.json
```

An independent source, sampled on a handful of dates to verify that the primary
provider's **level** is right. It is not used to build history: dated releases start
in March 2024 and each date is a separate HTTP request.

### Not used — Banque Centrale de Mauritanie

The BCM publishes the **official** rate, which is the authoritative reference for
Mauritania. It is not used here because there is no stable machine-readable
historical series to ingest reproducibly. This is a real limitation, stated plainly:

> **The rates in this project are indicative mid-market provider quotes, not the
> official BCM rate, and the two can differ.**

## 3. The daily value is the median of OHLC — here is why

The `close` field of `MRU=X` is not trustworthy. Measured over the full history:

| Field | std of daily % change | Example failure |
|---|---|---|
| `close` | **43.2%** | 2022-05-31: close `2.00` against an open of `36.55` |
| `open` | 20.0% | |
| median of O/H/L/C | **2.5%** | unaffected by any single corrupted field |

A corrupted *single field* cannot move the median of four. The daily representative
value is therefore:

```python
value = median(open, high, low, close)
```

`close` is still archived, and `tests/test_data_sources.py` pins both the failure
and the fix so a future refactor cannot quietly undo it.

## 4. What cleaning removes, and what it deliberately keeps

Bad ticks are removed; **real policy steps are kept**. The distinction is structural:

- a **bad tick** deviates from the previous *and* the next observation, in the same
  direction — the series jumps away and comes straight back;
- a **step devaluation** deviates from the previous observation but not the next —
  the new level persists.

`detect_isolated_spikes` encodes exactly that asymmetry, and runs **iteratively**:
a corrupted *run* hides its members from a single pass because each bad point's
neighbour is itself bad. December 2022 in the live feed is precisely that case.

Effect on USD/MRU:

| | raw | cleaned |
|---|---|---|
| days moving more than 5% | 11 | **0** |
| largest daily move | 81.5% | 3.6% |
| minimum value | 2.00 | 33.90 |

Gaps are filled **forward only**, bounded to 10 business days. Backward filling would
place a future observation in the past; it is used nowhere in this project.

## 5. Known limitations

1. **Provider disagreement sets a floor on accuracy.** Over the sampled dates the two
   providers differ on USD/MRU by −0.2% to −3.0%, and the sign is systematic rather
   than random. A model reporting a MAE of ~0.14 MRU (≈0.35%) is operating *well
   inside* the disagreement between two honest sources — a fact that belongs next to
   every accuracy number in this project.
2. **Not the official rate.** See above.
3. **EUR/MRU is derived.** See above.
4. **Forward-filling induces autocorrelation.** A filled flat day followed by a
   catch-up move creates negative lag-1 autocorrelation on its own. Some of the
   observed ACF(1) ≈ −0.30 is therefore measurement rather than economics, which is
   why model value is judged on a backtest rather than on that statistic.
5. **The history starts in November 2018** — about 2,050 business days. Long enough
   for the backtest used here, short enough that a single regime change is a large
   fraction of the sample.
