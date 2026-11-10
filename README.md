# Lodestar

Type a ticker, get a full fundamental read and a buy / hold / sell call.

**Live (hosted snapshot):** https://yashsarda31.github.io/lodestar/

```
run.bat                      # or: python app.py   ->  http://127.0.0.1:8820
python engine.py RELIANCE TCS AAPL     # scores in the terminal, no browser
```

Bare symbols resolve NSE first (`RELIANCE` -> `RELIANCE.NS`), then as typed (`AAPL`), then BSE.
`TCS.NS`, `infy.ns`, `NSE:TCS`, `500325.BO` and company names ("tata motors") also work. Old symbols
and everyday names Yahoo's search no longer knows (`Zomato` -> `ETERNAL.NS`, `HDFC`, `SBI`, `HUL`)
are mapped in `ALIASES` in `engine.py`; the hosted search index carries them too.

## What it reads

| Pillar | Weight | Measured on |
|---|---|---|
| Sales | 15 | 3-yr CAGR, latest-year growth, latest-quarter YoY, years of growth |
| Margins | 15 | operating and net margin levels, trend vs prior years, TTM vs last FY |
| EPS | 20 | 3-yr CAGR, latest year, latest quarter, cash conversion, earnings surprises |
| Return on equity | 15 | ROE, 3-yr average, trend, ROCE (ROA for lenders), DuPont split |
| Forward estimates | 20 | consensus EPS growth for the next two years, revisions, target upside, rating |
| Valuation | 15 | PEG, forward P/E, P/E vs own 5-yr median, EV/EBITDA, FCF yield (P/B vs ROE for lenders) |

Each metric maps to 0-100 through a fixed hurdle table (the `*_MAP` constants at the top of
`engine.py`). Composite >= 65 is BUY, < 45 is SELL, otherwise HOLD. A BUY is capped at HOLD when
the valuation pillar is under 30, the forward pillar is under 35, or the company is loss-making
with no forecast profit.

## What it is not

- The hurdles are judgment. Nothing here has been backtested, so a score of 70 has no measured
  edge over a score of 50.
- Margins are scored on absolute levels with no peer comparison.
- Data is Yahoo Finance as published. It is sometimes wrong (Indian banks, recent demergers).
  Known traps handled: unadjusted pre-split EPS, one-off / discontinued profit, stale quarterly
  statements, consensus on a different basis from reported numbers.

## Hosted version (GitHub Pages)

Pages can't run Python, so the hosted site is a pre-computed snapshot: `build_static.py` runs the
same engine over every symbol in `universe.txt` (NIFTY 200 + 20 large US names) and writes
`docs/data/*.json`; the front-end reads those instead of calling the API. A GitHub Action
(`.github/workflows/pages.yml`) refreshes the snapshot every weekday at 19:30 IST and redeploys.
If a symbol fails on a given day, yesterday's file is kept. To analyse a ticker outside the list,
run the app locally.

```
python build_static.py                 # rebuild docs/ locally
python build_static.py RELIANCE AAPL   # refresh just these
python build_static.py --render-only   # no Yahoo calls: push a front-end or alias change into docs/
```

## Files

- `engine.py` - data pull, cleaning, scoring, narrative. Edit hurdles and weights here.
- `app.py` - FastAPI server: `/`, `/api/analyze?ticker=`, `/api/search?q=`.
- `static/index.html` - the whole front-end; charts are hand-rolled SVG, no JS dependencies.
- `build_static.py`, `universe.txt`, `docs/` - the hosted snapshot (see above).
