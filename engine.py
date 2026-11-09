"""
engine.py - Lodestar's analysis engine.

Pulls a company's statements, consensus estimates and price history from Yahoo Finance
(via yfinance), cleans the known data traps, and scores six pillars:

    Sales · Margins · EPS · Returns (ROE) · Forward estimates · Valuation

Each pillar is a weighted set of metrics; each metric is mapped to 0-100 through an explicit
piecewise-linear hurdle table (see the *_MAP constants). The weighted pillar scores make a
composite, and the composite makes the call:  BUY >= 65 · HOLD 45-65 · SELL < 45, with three
guard-rails (stretched valuation, falling estimates, losses) that cap a BUY at HOLD.

The hurdles are judgment, not a backtest. The output says so.

CLI smoke test:   python engine.py RELIANCE TCS AAPL
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Lenders have no gross/operating margin, no meaningful FCF and structural leverage.
LENDER_WORDS = ("Bank", "Insurance", "Credit Services", "Mortgage", "Financial Conglomerates")

WEIGHTS = {"sales": 15, "margins": 15, "eps": 20, "returns": 15, "forward": 20, "valuation": 15}
BUY_AT, SELL_BELOW = 65.0, 45.0


class AnalysisError(Exception):
    pass


# ───────────────────────────── small helpers ─────────────────────────────
def num(x):
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def lerp(x, pts):
    """Piecewise-linear map of x through [(x0, score0), (x1, score1), ...], clamped at both ends."""
    if x is None:
        return None
    if x <= pts[0][0]:
        return float(pts[0][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return float(pts[-1][1])


def growth_map(s):
    """Growth hurdles in %, scaled by s (1.0 for INR reporters, lower for low-inflation currencies)."""
    return [(-15, 0), (0, 22), (6 * s, 40), (10 * s, 55), (15 * s, 72), (20 * s, 85), (30 * s, 100)]


OPM_MAP = [(-5, 0), (0, 10), (5, 30), (10, 48), (15, 62), (20, 76), (30, 92), (40, 100)]
NPM_MAP = [(-5, 0), (0, 10), (3, 30), (7, 50), (12, 68), (20, 88), (30, 100)]
NPM_LENDER_MAP = [(0, 5), (8, 35), (15, 60), (22, 80), (30, 100)]
MARGIN_TREND_MAP = [(-6, 0), (-3, 20), (-1, 40), (0, 52), (1, 64), (3, 82), (6, 100)]
ROE_MAP = [(0, 0), (8, 28), (12, 48), (15, 62), (20, 80), (28, 100)]
ROE_TREND_MAP = [(-8, 0), (-4, 22), (-1, 42), (0, 52), (2, 68), (5, 88), (8, 100)]
ROCE_MAP = [(0, 0), (8, 28), (12, 46), (16, 62), (22, 80), (30, 100)]
ROA_MAP = [(0, 0), (0.5, 25), (1.0, 50), (1.5, 70), (2.0, 88), (3.0, 100)]
CASH_CONV_MAP = [(0.2, 0), (0.5, 25), (0.8, 52), (1.0, 72), (1.3, 100)]
SURPRISE_MAP = [(-15, 0), (-5, 25), (0, 50), (3, 68), (8, 88), (15, 100)]
CONSISTENCY_MAP = [(0, 5), (0.34, 30), (0.67, 62), (1.0, 95)]
REVISION_MAP = [(-100, 0), (-50, 20), (0, 50), (50, 80), (100, 100)]
UPSIDE_MAP = [(-25, 0), (-10, 18), (0, 38), (10, 56), (20, 74), (35, 92), (50, 100)]
RATING_MAP = [(1, 100), (1.5, 90), (2, 75), (2.5, 58), (3, 40), (4, 12), (5, 0)]
PEG_MAP = [(0.4, 100), (0.8, 88), (1.2, 70), (1.6, 54), (2.0, 40), (3.0, 18), (4.5, 0)]
PE_MAP = [(6, 100), (12, 84), (18, 68), (24, 52), (32, 36), (45, 18), (70, 0)]
PE_REL_MAP = [(0.5, 100), (0.75, 82), (1.0, 55), (1.25, 32), (1.6, 10), (2.0, 0)]
EV_EBITDA_MAP = [(5, 100), (9, 80), (14, 60), (20, 40), (30, 18), (45, 0)]
FCF_YIELD_MAP = [(-2, 0), (0, 15), (2, 42), (4, 65), (6, 85), (9, 100)]
PB_REL_MAP = [(0.5, 100), (0.8, 80), (1.0, 60), (1.4, 35), (2.0, 10), (3.0, 0)]


def grade(score):
    if score is None:
        return "No data"
    return "Excellent" if score >= 80 else "Strong" if score >= 65 else "Fair" if score >= 50 else "Weak" if score >= 35 else "Poor"


def pct(v, signed=False, dp=1):
    if v is None:
        return "n/a"
    return f"{v:+.{dp}f}%" if signed else f"{v:.{dp}f}%"


def pts_txt(v):
    return f"{abs(v):.1f} pts"


def fy_label(d, est=False):
    return f"FY{d.year % 100:02d}" + ("E" if est else "")


def q_label(d):
    return d.strftime("%b '%y")


def sval(df, names, col):
    if df is None or getattr(df, "empty", True) or col is None or col not in df.columns:
        return None
    for n in names:
        if n in df.index:
            v = num(df.at[n, col])
            if v is not None:
                return v
    return None


def match_col(df, col, tol=45):
    if df is None or getattr(df, "empty", True):
        return None
    best = None
    for c in df.columns:
        d = abs((c - col).days)
        if d <= tol and (best is None or d < best[0]):
            best = (d, c)
    return best[1] if best else None


def est_row(df, key):
    try:
        if df is None or df.empty or key not in df.index:
            return None
        return df.loc[key]
    except Exception:
        return None


def metric(label, value, unit, score, weight, text=None, good=None, bad=None):
    """One scored line. `good` / `bad` are the sentences used in the case for / against."""
    if value is None and score is None:
        text = None
    return {"label": label, "value": value, "unit": unit, "score": None if score is None else round(score, 1),
            "weight": weight, "text": text, "good": good or text, "bad": bad or text}


def pillar(key, name, metrics, blurb):
    total = sum(m["weight"] for m in metrics) or 1
    avail = [m for m in metrics if m["score"] is not None]
    w = sum(m["weight"] for m in avail)
    score = sum(m["score"] * m["weight"] for m in avail) / w if avail and w / total >= 0.4 else None
    return {"key": key, "name": name, "weight": WEIGHTS[key], "score": None if score is None else round(score, 1),
            "grade": grade(score), "coverage": round(w / total, 2), "metrics": metrics, "blurb": blurb}


# ───────────────────────────── data pull ─────────────────────────────
def resolve_symbol(query: str):
    """'RELIANCE' -> RELIANCE.NS (NSE first, then as typed, then BSE); 'NSE:TCS'; 'AAPL'; free text via search."""
    q = (query or "").strip()
    if not q:
        raise AnalysisError("Type a ticker to analyse.")
    up = q.upper()
    cands = []
    if ":" in up:
        ex, s = up.split(":", 1)
        s = s.strip().replace("_", "&") if ex.strip() == "NSE" and "_" in s else s.strip()
        cands = [s + {"NSE": ".NS", "BSE": ".BO"}.get(ex.strip(), "")]
    elif " " in up:
        cands = []
    elif "." in up:
        cands = [up]
    else:
        cands = [up + ".NS", up, up + ".BO"]
    for c in cands:
        info = _info(c)
        if info:
            return c, info
    for hit in search(q, limit=4):
        info = _info(hit["symbol"])
        if info:
            return hit["symbol"], info
    raise AnalysisError(f"Couldn't find a listed company for “{q}”. Try the exchange symbol, e.g. RELIANCE, TCS.NS or AAPL.")


def _info(sym):
    try:
        info = yf.Ticker(sym).info or {}
    except Exception:
        return None
    price = num(info.get("currentPrice")) or num(info.get("regularMarketPrice"))
    if not price or not (info.get("longName") or info.get("shortName")):
        return None
    return info


def search(q, limit=7):
    out, seen = [], set()
    try:
        quotes = yf.Search(q, max_results=12, news_count=0).quotes
    except Exception:
        return out
    for x in quotes:
        if x.get("quoteType") != "EQUITY" or not x.get("symbol"):
            continue
        sym = x["symbol"]
        base = sym.split(".")[0]
        if sym.endswith(".BO") and base + ".NS" in seen:
            continue
        if "-" in base and sym.endswith((".NS", ".BO")):  # NSE series suffixes (-BL, -BE ...)
            continue
        seen.add(sym)
        out.append({"symbol": sym, "name": x.get("longname") or x.get("shortname") or sym,
                    "exchange": x.get("exchDisp") or x.get("exchange") or ""})
    out.sort(key=lambda r: 0 if r["symbol"].endswith(".NS") else 1)  # stable: NSE listings first
    return out[:limit]


def pull(sym, info):
    """Everything Yahoo has on the company. The cookie/crumb already exists (resolve_symbol fetched .info),
    so the remaining requests run side by side, each on its own Ticker object."""
    names = ("income_stmt", "quarterly_income_stmt", "balance_sheet", "cashflow", "earnings_estimate",
             "revenue_estimate", "eps_trend", "eps_revisions", "earnings_history", "analyst_price_targets", "hist")

    def one(name):
        for attempt in range(2):
            try:
                t = yf.Ticker(sym)
                if name == "hist":
                    return t.history(period="6y", interval="1wk", auto_adjust=False)
                return getattr(t, name)
            except Exception:
                time.sleep(0.4)
        return None

    with ThreadPoolExecutor(max_workers=6) as ex:
        raw = dict(zip(names, ex.map(one, names)))
    raw["info"] = info
    return raw


# ───────────────────────────── statement tables ─────────────────────────────
def underlying(df, c, eps, ni, f=1.0):
    """Profit and EPS before one-offs and discontinued operations, from Yahoo's 'Normalized Income'.
    Returns (ni_u, eps_u, is_adjusted); falls back to the reported figures when the gap is under 10%
    or the share count can't be trusted."""
    norm = sval(df, ["Normalized Income"], c)
    if norm is None or ni is None or abs(norm - ni) <= 0.10 * max(abs(ni), abs(norm), 1.0):
        return ni, eps, False
    sh = sval(df, ["Diluted Average Shares", "Basic Average Shares"], c)
    if not sh or eps is None:
        return norm, eps, eps is None
    sh *= f
    variants = [sval(df, [k], c) for k in ("Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest",
                                           "Diluted NI Availto Com Stockholders", "Net Income")]
    ok = any(v is not None and abs(eps) > 1e-9 and v / eps > 0 and abs(v / eps / sh - 1) <= 0.25 for v in variants)
    if not ok:
        return norm, eps, False
    return norm, norm / sh, True


def build_annual(raw, notes):
    inc, bal, cf = raw["income_stmt"], raw["balance_sheet"], raw["cashflow"]
    if inc is None or inc.empty:
        return []
    cols = sorted(c for c in inc.columns if sval(inc, ["Total Revenue", "Operating Revenue"], c))[-5:]
    hist = raw.get("hist")
    splits = []
    if hist is not None and not hist.empty and "Stock Splits" in hist.columns:
        s = hist["Stock Splits"]
        splits = [(i.tz_localize(None) if i.tzinfo else i, float(v)) for i, v in s[s > 0].items()]
    shares_now = num(raw["info"].get("sharesOutstanding"))

    rows = []
    for c in cols:
        bc, cc = match_col(bal, c), match_col(cf, c)
        r = {
            "end": c, "fy": fy_label(c),
            "revenue": sval(inc, ["Total Revenue", "Operating Revenue"], c),
            "gross": sval(inc, ["Gross Profit"], c),
            "op": sval(inc, ["Operating Income"], c),
            "ebit": sval(inc, ["EBIT", "Operating Income"], c),
            "ebitda": sval(inc, ["EBITDA", "Normalized EBITDA"], c),
            "ni": sval(inc, ["Net Income Common Stockholders", "Net Income"], c),
            "eps": sval(inc, ["Diluted EPS", "Basic EPS"], c),
            "interest": sval(inc, ["Interest Expense", "Interest Expense Non Operating"], c),
            "equity": sval(bal, ["Stockholders Equity", "Common Stock Equity"], bc),
            "debt": sval(bal, ["Total Debt"], bc),
            "assets": sval(bal, ["Total Assets"], bc),
            "ocf": sval(cf, ["Operating Cash Flow"], cc),
            "fcf": sval(cf, ["Free Cash Flow"], cc),
        }
        applied = 1.0
        if r["eps"] is None and r["ni"] is not None:
            sh = sval(inc, ["Diluted Average Shares", "Basic Average Shares"], c)
            if sh:
                r["eps"] = r["ni"] / sh
        # Yahoo sometimes leaves a pre-split / pre-bonus EPS in one year only. If the share count implied
        # by NI / EPS only matches today's count after applying later splits, the EPS is unadjusted.
        if r["eps"] and r["ni"] and r["eps"] > 0 and r["ni"] > 0 and shares_now:
            f = 1.0
            for d, ratio in splits:
                if d > c:
                    f *= ratio
            if abs(f - 1) > 0.01:
                implied = r["ni"] / r["eps"]
                a, b = abs(math.log(implied / shares_now)), abs(math.log(implied * f / shares_now))
                if b < a - 0.2:
                    r["eps"] /= f
                    applied = f
                    notes.append(f"{r['fy']} EPS on Yahoo was not adjusted for a later split/bonus; restated by {f:g}×.")
        r["epsReported"], r["niReported"] = r["eps"], r["ni"]
        r["ni"], r["eps"], r["oneoff"] = underlying(inc, c, r["eps"], r["ni"], applied)
        if r["oneoff"] and r["epsReported"] is not None:
            notes.append(f"{r['fy']} profit includes one-off or discontinued items: reported EPS {r['epsReported']:.2f}, underlying {r['eps']:.2f}. Underlying is used throughout.")
        rows.append(r)

    for i, r in enumerate(rows):
        p = rows[i - 1] if i else None
        rev = r["revenue"]
        r["grossMargin"] = r["gross"] / rev * 100 if r["gross"] is not None else None
        r["opMargin"] = r["op"] / rev * 100 if r["op"] is not None else None
        r["ebitdaMargin"] = r["ebitda"] / rev * 100 if r["ebitda"] is not None else None
        r["netMargin"] = r["ni"] / rev * 100 if r["ni"] is not None else None
        r["revGrowth"] = (rev / p["revenue"] - 1) * 100 if p and p["revenue"] else None
        r["epsGrowth"] = (r["eps"] / p["eps"] - 1) * 100 if p and p["eps"] and p["eps"] > 0 and r["eps"] is not None else None
        avg = lambda k: ((r[k] + p[k]) / 2 if p and p.get(k) else r[k]) if r.get(k) else None
        eq, assets = avg("equity"), avg("assets")
        cap = None
        if r["equity"] and r["equity"] > 0:
            cap_now = r["equity"] + (r["debt"] or 0)
            cap_prev = (p["equity"] + (p["debt"] or 0)) if p and p.get("equity") and p["equity"] > 0 else None
            cap = (cap_now + cap_prev) / 2 if cap_prev else cap_now
        r["roe"] = r["ni"] / eq * 100 if r["ni"] is not None and eq and eq > 0 else None
        r["roce"] = r["ebit"] / cap * 100 if r["ebit"] is not None and cap else None
        r["roa"] = r["ni"] / assets * 100 if r["ni"] is not None and assets else None
        r["de"] = r["debt"] / r["equity"] if r["debt"] is not None and r["equity"] and r["equity"] > 0 else None
        r["turnover"] = rev / assets if assets else None
        r["leverage"] = assets / eq if assets and eq and eq > 0 else None
        opx = r["op"] if r["op"] is not None else r["ebit"]
        r["cover"] = opx / r["interest"] if opx is not None and r["interest"] and r["interest"] > 0 else None
        r["negEquity"] = r["equity"] is not None and r["equity"] <= 0
    return rows


def build_quarterly(raw):
    q = raw["quarterly_income_stmt"]
    if q is None or q.empty:
        return []
    cols = sorted(c for c in q.columns if sval(q, ["Total Revenue", "Operating Revenue"], c))[-8:]
    rows = []
    for c in cols:
        rev = sval(q, ["Total Revenue", "Operating Revenue"], c)
        op, ni = sval(q, ["Operating Income"], c), sval(q, ["Net Income Common Stockholders", "Net Income"], c)
        ni, eps, _ = underlying(q, c, sval(q, ["Diluted EPS", "Basic EPS"], c), ni)
        rows.append({"end": c, "label": q_label(c), "revenue": rev, "op": op, "ni": ni,
                     "eps": eps,
                     "opMargin": op / rev * 100 if op is not None else None,
                     "netMargin": ni / rev * 100 if ni is not None else None})
    for r in rows:
        ya = next((x for x in rows if 350 <= (r["end"] - x["end"]).days <= 380), None)
        r["revYoY"] = (r["revenue"] / ya["revenue"] - 1) * 100 if ya and ya["revenue"] else None
        r["epsYoY"] = (r["eps"] / ya["eps"] - 1) * 100 if ya and ya["eps"] and ya["eps"] > 0 and r["eps"] is not None else None
    return rows


def cagr(first, last, years):
    if first is None or last is None or years <= 0 or first <= 0 or last <= 0:
        return None
    return ((last / first) ** (1 / years) - 1) * 100


def pe_band(hist, annual, lag_days=75):
    """Median / 10th / 90th percentile of the trailing P/E over the price history, using each
    fiscal year's EPS from `lag_days` after year-end (roughly when it was published)."""
    if hist is None or hist.empty:
        return None
    events = [(r["end"] + pd.Timedelta(days=lag_days), r["eps"]) for r in annual if r["eps"] is not None]
    if len(events) < 2:
        return None
    vals = []
    for d, close in hist["Close"].dropna().items():
        d = d.tz_localize(None) if d.tzinfo else d
        eps = None
        for ed, e in events:
            if ed <= d:
                eps = e
        if eps and eps > 0:
            pe = float(close) / eps
            if 0 < pe < 400:
                vals.append(pe)
    if len(vals) < 60:
        return None
    vals.sort()
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
    return {"median": statistics.median(vals), "low": q(0.10), "high": q(0.90), "weeks": len(vals)}


# ───────────────────────────── the analysis ─────────────────────────────
def analyze(query: str) -> dict:
    sym, info = resolve_symbol(query)
    if (info.get("quoteType") or "EQUITY") != "EQUITY":
        raise AnalysisError(f"{sym} is a {str(info.get('quoteType')).lower()}, not a company — there are no statements to analyse.")
    raw = pull(sym, info)
    notes: list[str] = []
    flags: list[dict] = []
    today = pd.Timestamp(dt.date.today())

    price = num(info.get("currentPrice")) or num(info.get("regularMarketPrice"))
    cur = info.get("currency") or ""
    fcur = info.get("financialCurrency") or cur
    same_cur = cur == fcur
    sector, industry = info.get("sector") or "", info.get("industry") or ""
    lender = sector == "Financial Services" and any(w in industry for w in LENDER_WORDS)
    s = 1.0 if fcur == "INR" else 0.65  # growth-hurdle scale
    G = growth_map(s)
    name = info.get("longName") or info.get("shortName") or sym
    short = name.replace(" Limited", "").replace(" Ltd.", "").replace(" Ltd", "").replace(", Inc.", "").replace(" Inc.", "").replace(" Corporation", "")

    annual = build_annual(raw, notes)
    if len(annual) < 2:
        raise AnalysisError(f"Yahoo Finance has no usable annual statements for {sym}, so there is nothing to score.")
    quarterly = build_quarterly(raw)
    last, prev = annual[-1], annual[-2]
    n_years = len(annual) - 1
    span = min(n_years, 3)
    base = annual[-1 - span]

    # ── freshest quarter / TTM ──
    fresh_q = bool(quarterly) and (today - quarterly[-1]["end"]).days <= 200
    lq = quarterly[-1] if fresh_q else None
    if quarterly and not fresh_q:
        notes.append(f"Yahoo's quarterly statements stop at {q_label(quarterly[-1]['end'])}; latest-quarter figures use Yahoo's summary fields instead.")
    q_rev_yoy = lq["revYoY"] if lq else None
    q_eps_yoy = lq["epsYoY"] if lq else None
    q_name = lq["label"] if lq else "Latest-quarter"
    if q_rev_yoy is None and not lender:
        v = num(info.get("revenueGrowth"))
        q_rev_yoy = v * 100 if v is not None and abs(v) <= 1.5 else None
    if q_eps_yoy is None:
        v = num(info.get("earningsGrowth"))
        if v is None:
            v = num(info.get("earningsQuarterlyGrowth"))
        q_eps_yoy = v * 100 if v is not None and abs(v) <= 3 else None

    ttm = {"revenue": num(info.get("totalRevenue")), "eps": num(info.get("trailingEps")), "ni": num(info.get("netIncomeToCommon")),
           "opMargin": None, "netMargin": None, "fromQuarters": False}
    if fresh_q and len(quarterly) >= 4:
        l4 = quarterly[-4:]
        gaps = [(b["end"] - a["end"]).days for a, b in zip(l4, l4[1:])]
        if all(80 <= g <= 100 for g in gaps):
            rev = sum(x["revenue"] for x in l4)
            ttm["revenue"], ttm["fromQuarters"] = rev, True
            if all(x["ni"] is not None for x in l4):
                ttm["ni"] = sum(x["ni"] for x in l4)
            if all(x["op"] is not None for x in l4):
                ttm["opMargin"] = sum(x["op"] for x in l4) / rev * 100
            if all(x["eps"] is not None for x in l4):
                ttm["eps"] = sum(x["eps"] for x in l4)
    if ttm["revenue"] and ttm["ni"] is not None:
        ttm["netMargin"] = ttm["ni"] / ttm["revenue"] * 100

    # ═════════════ 1 · SALES ═════════════
    rev_cagr = cagr(base["revenue"], last["revenue"], span)
    yoys = [r["revGrowth"] for r in annual if r["revGrowth"] is not None]
    grew = sum(1 for g in yoys if g > 0)
    consistency = grew / len(yoys) if yoys else None
    sales = pillar("sales", "Sales", [
        metric(f"{span}-year sales CAGR", rev_cagr, "pct", lerp(rev_cagr, G), 35,
               f"Sales compounded at {pct(rev_cagr)} a year over {span} years"),
        metric(f"{last['fy']} sales growth", last["revGrowth"], "spct", lerp(last["revGrowth"], G), 20,
               f"{last['fy']} sales {'grew' if (last['revGrowth'] or 0) >= 0 else 'fell'} {pct(abs(last['revGrowth'])) if last['revGrowth'] is not None else 'n/a'}"),
        metric(f"{q_name} sales, YoY", q_rev_yoy, "spct", lerp(q_rev_yoy, G), 25,
               f"Latest-quarter sales {'up' if (q_rev_yoy or 0) >= 0 else 'down'} {pct(abs(q_rev_yoy)) if q_rev_yoy is not None else 'n/a'} year on year"),
        metric("Years of growth", consistency, "ratio", lerp(consistency, CONSISTENCY_MAP), 20,
               f"Sales grew in {grew} of the last {len(yoys)} years",
               good=f"Sales grew in {grew} of the last {len(yoys)} years", bad=f"Sales grew in only {grew} of the last {len(yoys)} years"),
    ], "How fast the top line compounds, and whether it is still doing so in the latest quarter.")
    sales["metrics"][3]["display"] = f"{grew} of {len(yoys)}" if yoys else None

    # ═════════════ 2 · MARGINS ═════════════
    def prior_avg(key, n=3):
        vals = [r[key] for r in annual[:-1][-n:] if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    opm, npm, gm = last["opMargin"], last["netMargin"], last["grossMargin"]
    opm_tr = opm - prior_avg("opMargin") if opm is not None and prior_avg("opMargin") is not None else None
    npm_tr = npm - prior_avg("netMargin") if npm is not None and prior_avg("netMargin") is not None else None
    gm_tr = gm - prior_avg("grossMargin") if gm is not None and prior_avg("grossMargin") is not None else None
    ttm_opm_d = ttm["opMargin"] - opm if ttm["opMargin"] is not None and opm is not None else None

    def trend_txt(what, d, ref="its prior-years average"):
        if d is None:
            return None
        return f"{what} is {pts_txt(d)} {'above' if d >= 0 else 'below'} {ref}"

    if lender:
        margins = pillar("margins", "Margins", [
            metric(f"{last['fy']} net margin", npm, "pct", lerp(npm, NPM_LENDER_MAP), 50, f"Net margin of {pct(npm)}"),
            metric("Net margin vs prior years", npm_tr, "pp", lerp(npm_tr, MARGIN_TREND_MAP), 50, trend_txt("Net margin", npm_tr)),
        ], "For a lender only the net margin is meaningful — gross and operating margins don't exist in the statements.")
    else:
        margins = pillar("margins", "Margins", [
            metric(f"{last['fy']} operating margin", opm, "pct", lerp(opm, OPM_MAP), 30, f"Operating margin of {pct(opm)}"),
            metric(f"{last['fy']} net margin", npm, "pct", lerp(npm, NPM_MAP), 20, f"Net margin of {pct(npm)}"),
            metric("Operating margin vs prior years", opm_tr, "pp", lerp(opm_tr, MARGIN_TREND_MAP), 30, trend_txt("Operating margin", opm_tr)),
            metric("TTM operating margin vs last FY", ttm_opm_d, "pp", lerp(ttm_opm_d, MARGIN_TREND_MAP), 10,
                   trend_txt("Trailing-12-month operating margin", ttm_opm_d, f"{last['fy']}'s")),
            metric("Gross margin vs prior years", gm_tr, "pp", lerp(gm_tr, MARGIN_TREND_MAP), 10, trend_txt("Gross margin", gm_tr)),
        ], "How much of each rupee of sales survives as profit, and whether that share is widening or being squeezed."
            if fcur == "INR" else "How much of each unit of sales survives as profit, and whether that share is widening or being squeezed.")

    # ═════════════ 3 · EPS ═════════════
    eps_l, eps_b = last["eps"], base["eps"]
    eps_cagr = cagr(eps_b, eps_l, span)
    if eps_cagr is not None:
        eps_cagr_score, eps_cagr_txt = lerp(eps_cagr, G), f"EPS compounded at {pct(eps_cagr)} a year over {span} years"
    elif eps_l is not None and eps_l <= 0:
        eps_cagr_score, eps_cagr_txt = 5, f"{last['fy']} EPS is negative"
    elif eps_l is not None and eps_b is not None and eps_b <= 0:
        eps_cagr_score, eps_cagr_txt = 70, f"EPS turned positive from a loss in {base['fy']}"
    else:
        eps_cagr_score, eps_cagr_txt = None, None
    eg = last["epsGrowth"]
    eg_score = lerp(eg, G) if eg is not None else (5 if eps_l is not None and eps_l <= 0 else None)
    sum_ni = sum(r["ni"] for r in annual[-3:] if r["ni"] is not None and r["ocf"] is not None)
    sum_ocf = sum(r["ocf"] for r in annual[-3:] if r["ni"] is not None and r["ocf"] is not None)
    cash_conv = sum_ocf / sum_ni if sum_ni and sum_ni > 0 and not lender else None
    eh = raw.get("earnings_history")
    surprises = []
    try:
        if eh is not None and not eh.empty:
            for d, r in eh.iterrows():
                a, e = num(r.get("epsActual")), num(r.get("epsEstimate"))
                sp = num(r.get("surprisePercent"))
                if a is not None and e is not None:
                    surprises.append({"q": q_label(pd.Timestamp(d)), "actual": a, "estimate": e,
                                      "pct": sp * 100 if sp is not None else ((a / e - 1) * 100 if e else None)})
    except Exception:
        surprises = []
    sp_vals = [max(-40, min(40, x["pct"])) for x in surprises if x["pct"] is not None]
    avg_sp = sum(sp_vals) / len(sp_vals) if len(sp_vals) >= 2 else None
    beats = sum(1 for x in sp_vals if x > 0)
    eps_p = pillar("eps", "Earnings per share", [
        metric(f"{span}-year EPS CAGR", eps_cagr, "pct", eps_cagr_score, 30, eps_cagr_txt),
        metric(f"{last['fy']} EPS growth", eg, "spct", eg_score, 20,
               f"{last['fy']} EPS {'grew' if (eg or 0) >= 0 else 'fell'} {pct(abs(eg))}" if eg is not None else None),
        metric(f"{q_name} EPS, YoY", q_eps_yoy, "spct", lerp(q_eps_yoy, G), 20,
               f"Latest-quarter EPS {'up' if (q_eps_yoy or 0) >= 0 else 'down'} {pct(abs(q_eps_yoy)) if q_eps_yoy is not None else 'n/a'} year on year"),
        metric("Cash conversion (OCF ÷ profit, 3y)", cash_conv, "x", lerp(cash_conv, CASH_CONV_MAP), 15,
               f"Operating cash flow is {cash_conv:.2f}× reported profit over 3 years" if cash_conv is not None else None,
               good=f"Profits are backed by cash — operating cash flow is {cash_conv:.2f}× net profit over 3 years" if cash_conv is not None else None,
               bad=f"Only {cash_conv:.2f}× of reported profit showed up as operating cash over 3 years" if cash_conv is not None else None),
        metric("Average earnings surprise (4q)", avg_sp, "spct", lerp(avg_sp, SURPRISE_MAP), 15,
               f"Beat consensus in {beats} of the last {len(sp_vals)} quarters (average surprise {pct(avg_sp, True)})" if avg_sp is not None else None,
               bad=f"Missed consensus in {len(sp_vals) - beats} of the last {len(sp_vals)} quarters (average surprise {pct(avg_sp, True)})" if avg_sp is not None else None),
    ], "Whether growth reaches shareholders per share, whether it is backed by cash, and whether results beat expectations.")

    # ═════════════ 4 · RETURNS ═════════════
    roe = last["roe"]
    roes = [r["roe"] for r in annual[-3:] if r["roe"] is not None]
    roe_avg = sum(roes) / len(roes) if roes else None
    roe_tr = roe - prior_avg("roe") if roe is not None and prior_avg("roe") is not None else None
    roce, roa = last["roce"], last["roa"]
    neg_eq = last.get("negEquity")
    ret_metrics = [
        metric(f"{last['fy']} ROE", roe, "pct", 0 if neg_eq else lerp(roe, ROE_MAP), 40,
               "Shareholders' equity is negative — accumulated losses exceed the capital put in" if neg_eq else f"Return on equity of {pct(roe)}"),
        metric("3-year average ROE", roe_avg, "pct", lerp(roe_avg, ROE_MAP), 20, f"ROE has averaged {pct(roe_avg)} over 3 years"),
        metric("ROE vs prior years", roe_tr, "pp", lerp(roe_tr, ROE_TREND_MAP), 15, trend_txt("ROE", roe_tr)),
    ]
    if lender:
        ret_metrics.append(metric(f"{last['fy']} ROA", roa, "pct2", lerp(roa, ROA_MAP), 25, f"Return on assets of {pct(roa, dp=2)}"))
    else:
        ret_metrics.append(metric(f"{last['fy']} ROCE", roce, "pct", lerp(roce, ROCE_MAP), 25, f"Return on capital employed of {pct(roce)}"))
    returns = pillar("returns", "Return on equity", ret_metrics,
                     "What the business earns on shareholders' money, and whether that comes from operations or from leverage.")
    dupont = None
    if last["netMargin"] is not None and last["turnover"] and last["leverage"]:
        dupont = {"netMargin": last["netMargin"], "turnover": last["turnover"], "leverage": last["leverage"], "roe": roe}

    # ═════════════ 5 · FORWARD ESTIMATES ═════════════
    ee, re_ = raw.get("earnings_estimate"), raw.get("revenue_estimate")
    r0, r1 = est_row(ee, "0y"), est_row(ee, "+1y")
    g = lambda r, k: num(r.get(k)) if r is not None else None
    eps0, eps1, eps_base = g(r0, "avg"), g(r1, "avg"), g(r0, "yearAgoEps")
    n_an = max(g(r0, "numberOfAnalysts") or 0, g(r1, "numberOfAnalysts") or 0) or None
    est_cur = fcur
    try:
        if ee is not None and "currency" in ee.columns and len(ee["currency"].dropna()):
            est_cur = str(ee["currency"].dropna().iloc[0])
    except Exception:
        pass
    nfy = num(info.get("nextFiscalYearEnd"))
    fy0_end = pd.Timestamp(dt.datetime.fromtimestamp(nfy, dt.timezone.utc).date()) if nfy else last["end"] + pd.DateOffset(years=1)
    if fy0_end <= last["end"]:
        fy0_end = last["end"] + pd.DateOffset(years=1)
    fy0, fy1 = fy_label(fy0_end, True), fy_label(fy0_end + pd.DateOffset(years=1), True)

    g0 = (eps0 / eps_base - 1) * 100 if eps0 is not None and eps_base and eps_base > 0 else None
    g1 = (eps1 / eps0 - 1) * 100 if eps1 is not None and eps0 and eps0 > 0 else None
    basis_gap = None
    statements_current = abs((fy0_end - pd.DateOffset(years=1) - last["end"]).days) <= 45
    if statements_current and eps_base and eps_l and eps_base > 0 and eps_l > 0 and abs(eps_base / eps_l - 1) > 0.12:
        basis_gap = (eps_base / eps_l - 1) * 100
        notes.append(f"Consensus EPS is on a different basis from reported EPS (analysts' {last['fy']} base {eps_base:.2f} vs reported {eps_l:.2f}) — "
                     "often standalone vs consolidated, or adjusted vs reported. Forward growth is measured against the analysts' own base.")

    rev0 = rev1 = rg0 = rg1 = None
    if not lender:
        q0, q1 = est_row(re_, "0y"), est_row(re_, "+1y")
        a0, a1 = g(q0, "avg"), g(q1, "avg")
        if a0 and statements_current and 0.8 <= a0 / last["revenue"] <= 2.0:
            rev0, rg0 = a0, (a0 / last["revenue"] - 1) * 100
            if a1 and abs(a1 / a0 - 1) <= 0.6:
                rev1, rg1 = a1, (a1 / a0 - 1) * 100
        elif a0:
            notes.append("Yahoo's revenue consensus is on a different basis from reported revenue, so forward sales estimates are left out.")

    up30 = dn30 = up7 = dn7 = breadth = None
    er = raw.get("eps_revisions")
    for key in ("+1y", "0y"):
        rr = est_row(er, key)
        if rr is None:
            continue
        u, d = num(rr.get("upLast30days")), num(rr.get("downLast30days"))
        if u is not None and d is not None and u + d >= 2:
            up30, dn30, up7, dn7 = u, d, num(rr.get("upLast7days")), num(rr.get("downLast7Days"))
            n = u + d
            breadth = (u - d) / n * 100 * n / (n + 2)  # shrunk toward 0 when only a few analysts revised
            break

    d30 = d90 = None
    et = raw.get("eps_trend")
    for key, ref in (("+1y", eps1), ("0y", eps0)):
        rr = est_row(et, key)
        c_ = num(rr.get("current")) if rr is not None else None
        if c_ and ref and abs(c_ / ref - 1) <= 0.05:  # Yahoo's trend table often disagrees with its own consensus
            a30, a90 = num(rr.get("30daysAgo")), num(rr.get("90daysAgo"))
            d30 = (c_ / a30 - 1) * 100 if a30 and a30 > 0 else None
            d90 = (c_ / a90 - 1) * 100 if a90 and a90 > 0 else None
            break

    apt = raw.get("analyst_price_targets") or {}
    tgt = {k: num(apt.get(k)) for k in ("mean", "high", "low", "median")} if isinstance(apt, dict) else {}
    if not tgt.get("mean"):
        tgt = {"mean": num(info.get("targetMeanPrice")), "high": num(info.get("targetHighPrice")),
               "low": num(info.get("targetLowPrice")), "median": num(info.get("targetMedianPrice"))}
    if tgt.get("mean") and not 0.3 <= tgt["mean"] / price <= 3.0:
        tgt = {}
    upside = (tgt["mean"] / price - 1) * 100 if tgt.get("mean") else None
    rec = num(info.get("recommendationMean"))
    n_rec = num(info.get("numberOfAnalystOpinions")) or n_an

    def fwd_growth(lbl, v, est, prior):
        """(score, sentence) for one forward year, including loss and turnaround cases."""
        if est is None:
            return None, None
        if est <= 0:
            return 5, f"Consensus expects a loss in {lbl}"
        if v is None:
            return (75, f"Consensus expects a swing to profit in {lbl}") if prior is not None and prior <= 0 else (None, None)
        return lerp(v, G), f"Consensus sees {lbl} EPS {'up' if v >= 0 else 'down'} {pct(abs(v))}"

    (s0, t0_), (s1, t1_) = fwd_growth(fy0, g0, eps0, eps_base), fwd_growth(fy1, g1, eps1, eps0)
    fwd = pillar("forward", "Forward estimates", [
        metric(f"{fy0} EPS growth (consensus)", g0, "spct", s0, 28, t0_),
        metric(f"{fy1} EPS growth (consensus)", g1, "spct", s1, 27, t1_),
        metric("Estimate revisions, 30 days", breadth, "breadth", lerp(breadth, REVISION_MAP), 20,
               f"{int(up30)} analysts raised and {int(dn30)} cut EPS estimates in the last 30 days" if breadth is not None else None),
        metric("Upside to mean target", upside, "spct", lerp(upside, UPSIDE_MAP), 15,
               f"Mean analyst target implies {pct(upside, True)} from here" if upside is not None else None,
               bad=f"Mean analyst target implies only {pct(upside, True)} from here" if upside is not None else None),
        metric("Analyst rating (1 buy – 5 sell)", rec, "num", lerp(rec, RATING_MAP), 10,
               f"Analysts rate it {rec:.1f} on a 1 (strong buy) to 5 (sell) scale" if rec is not None else None),
    ], "What the sell side expects next, and — more usefully — which way those expectations are moving.")
    if breadth is not None:
        fwd["metrics"][2]["display"] = f"{int(up30)}↑ {int(dn30)}↓"
    thin = fwd["score"] is not None and (n_rec or 0) < 3
    if thin:
        notes.append("Fewer than three analysts cover this stock, so the forward pillar carries half weight.")

    # NTM EPS: blend of the current and next fiscal year by how much of the current year is left.
    frac_left = max(0.0, min(1.0, (fy0_end - today).days / 365.0))
    ntm = eps0 * frac_left + eps1 * (1 - frac_left) if eps0 is not None and eps1 is not None else (eps1 if eps1 is not None else eps0)
    est_same_cur = est_cur == cur
    fwd_pe = price / ntm if ntm and ntm > 0 and est_same_cur else num(info.get("forwardPE"))
    fwd_cagr = cagr(eps_base, eps1, 2) if eps1 is not None else None
    if fwd_cagr is None and g0 is not None and g0 > 0 and eps1 is None:
        fwd_cagr = g0

    # ═════════════ 6 · VALUATION ═════════════
    pe = price / ttm["eps"] if ttm["eps"] and ttm["eps"] > 0 and same_cur else num(info.get("trailingPE"))
    loss_making = ttm["eps"] is not None and ttm["eps"] <= 0
    band = pe_band(raw.get("hist"), annual) if same_cur else None
    pe_rel = pe / band["median"] if pe and band else None
    peg = fwd_pe / fwd_cagr if fwd_pe and fwd_cagr and fwd_cagr > 0 else None
    pb = num(info.get("priceToBook"))
    ev_ebitda = num(info.get("enterpriseToEbitda"))
    mcap = num(info.get("marketCap"))
    fcf_yield = last["fcf"] / mcap * 100 if last["fcf"] is not None and mcap and same_cur and not lender else None
    use_pe, pe_lbl = (fwd_pe, "Forward P/E (next 12m)") if fwd_pe and fwd_pe > 0 else (pe, "Trailing P/E")

    if peg is not None:
        peg_score, peg_txt = lerp(peg, PEG_MAP), f"PEG of {peg:.2f} — {fwd_pe:.1f}× forward earnings for {pct(fwd_cagr)} expected EPS growth"
    elif fwd_pe and fwd_cagr is not None and fwd_cagr <= 0:
        peg_score, peg_txt = 8, f"Paying {fwd_pe:.1f}× forward earnings for EPS that consensus expects to shrink"
    else:
        peg_score, peg_txt = None, None
    if ntm is not None and ntm <= 0:
        use_pe, pe_lbl = None, "Forward P/E (next 12m)"
        pe_score, pe_txt = 5, "Consensus expects a loss over the next 12 months, so there is no forward P/E to anchor on"
    elif use_pe:
        pe_score, pe_txt = lerp(use_pe, PE_MAP), f"Trades at {use_pe:.1f}× {'forward' if use_pe is fwd_pe else 'trailing'} earnings"
    elif loss_making:
        pe_score, pe_txt = 5, "Loss-making on underlying earnings over the last 12 months, so there is no P/E to anchor on"
    else:
        pe_score, pe_txt = None, None
    val_metrics = [
        metric("PEG (forward P/E ÷ EPS growth)", peg, "x", peg_score, 30, peg_txt),
        metric(pe_lbl, use_pe, "x1", pe_score, 25, pe_txt),
        metric("P/E vs own 5-year median", pe_rel, "x", lerp(pe_rel, PE_REL_MAP), 20,
               (f"P/E of {pe:.1f}× is {abs(pe_rel - 1) * 100:.0f}% {'above' if pe_rel >= 1 else 'below'} its 5-year median of {band['median']:.1f}×")
               if pe_rel is not None else None),
    ]
    if lender:
        coe = 13.0 if fcur == "INR" else 10.0
        pb_rel = pb / (roe / coe) if pb and roe and roe > 0 else None
        val_metrics.append(metric("P/B vs ROE-justified P/B", pb_rel, "x", lerp(pb_rel, PB_REL_MAP), 25,
                                  f"P/B of {pb:.2f}× against {pct(roe)} ROE (justified ≈ {roe / coe:.2f}× at a {coe:.0f}% cost of equity)" if pb_rel is not None else None))
    else:
        val_metrics.append(metric("EV / EBITDA", ev_ebitda, "x1", lerp(ev_ebitda, EV_EBITDA_MAP) if ev_ebitda and ev_ebitda > 0 else None, 10,
                                  f"EV/EBITDA of {ev_ebitda:.1f}×" if ev_ebitda else None))
        val_metrics.append(metric(f"Free-cash-flow yield ({last['fy']})", fcf_yield, "pct", lerp(fcf_yield, FCF_YIELD_MAP), 15,
                                  f"Free-cash-flow yield of {pct(fcf_yield)}" if fcf_yield is not None else None))
    valuation = pillar("valuation", "Valuation", val_metrics,
                       "What you pay for all of the above — against growth, against the stock's own history, and against cash.")

    pillars = [sales, margins, eps_p, returns, fwd, valuation]

    # ═════════════ red flags ═════════════
    def flag(level, text):
        flags.append({"level": level, "text": text})

    if not lender:
        if last["de"] is not None and last["de"] > 1.5:
            flag("high", f"Debt is {last['de']:.1f}× equity.")
        if last["cover"] is not None and last["cover"] < 0:
            flag("high", "The business runs an operating loss, so interest is not covered at all.")
        elif last["cover"] is not None and last["cover"] < 3:
            flag("high", f"Operating profit covers interest only {last['cover']:.1f}×.")
        neg_fcf = [r["fy"] for r in annual[-3:] if r["fcf"] is not None and r["fcf"] < 0]
        if len(neg_fcf) >= 2:
            flag("med", f"Free cash flow was negative in {' and '.join(neg_fcf)}.")
        if cash_conv is not None and cash_conv < 0.6:
            flag("med", f"Only {cash_conv:.2f}× of reported profit turned into operating cash over 3 years.")
        if roe is not None and roce is not None and last["de"] and last["de"] > 1 and roe - roce > 8:
            flag("med", f"ROE ({pct(roe)}) is well above ROCE ({pct(roce)}) — leverage is doing part of the work.")
    if neg_eq:
        flag("high", "Shareholders' equity is negative.")
    if loss_making:
        flag("high", "Loss-making on underlying earnings over the trailing twelve months.")
    one = [r for r in annual[-2:] if r.get("oneoff")]
    if one:
        r_ = one[-1]
        flag("low", f"{r_['fy']} reported profit includes one-off or discontinued items (reported EPS {r_['epsReported']:.2f}, underlying {r_['eps']:.2f}).")
    nm = [r["netMargin"] for r in annual[-3:] if r["netMargin"] is not None]
    if len(nm) == 3 and nm[0] > nm[1] > nm[2]:
        flag("med", f"Net margin has fallen two years running ({pct(nm[0])} → {pct(nm[2])}).")
    if breadth is not None and breadth <= -40:
        flag("med", f"Estimates are being cut: {int(dn30)} downgrades vs {int(up30)} upgrades in 30 days.")
    if tgt.get("high") and price > tgt["high"]:
        flag("med", "Price is above the highest analyst target.")
    if basis_gap is not None:
        flag("low", "Consensus EPS is on a different basis from reported EPS — read forward growth with care.")

    # ═════════════ composite and the call ═════════════
    eff_w = {p["key"]: p["weight"] * (0.5 if p["key"] == "forward" and thin else 1.0) for p in pillars}
    scored = [p for p in pillars if p["score"] is not None]
    w_total = sum(eff_w[p["key"]] for p in scored)
    coverage = sum(eff_w[p["key"]] * p["coverage"] for p in scored) / sum(WEIGHTS.values())
    score = sum(p["score"] * eff_w[p["key"]] for p in scored) / w_total if w_total else None
    caps = []
    if score is None or coverage < 0.5 or len(scored) < 4:
        action = "NO CALL"
        caps.append("Too little usable data on Yahoo Finance to make a call.")
    else:
        action = "BUY" if score >= BUY_AT else "SELL" if score < SELL_BELOW else "HOLD"
        if action == "BUY":
            if valuation["score"] is not None and valuation["score"] < 30:
                action = "HOLD"
                caps.append("The business scores as a buy, but the valuation pillar is below 30 — capped at HOLD until the price catches down or earnings catch up.")
            elif fwd["score"] is not None and fwd["score"] < 35:
                action = "HOLD"
                caps.append("The record scores as a buy, but consensus estimates are weak or falling — capped at HOLD.")
            elif loss_making and (ntm is None or ntm <= 0):
                action = "HOLD"
                caps.append("Loss-making with no forecast profit in the next 12 months — capped at HOLD.")

    # ═════════════ narrative ═════════════
    cands = []
    for p in pillars:
        if p["score"] is None:
            continue
        mw = sum(m["weight"] for m in p["metrics"] if m["score"] is not None) or 1
        for m in p["metrics"]:
            if m["score"] is not None and m["text"]:
                cands.append((abs(m["score"] - 50) * eff_w[p["key"]] * m["weight"] / mw, m, p))
    cands.sort(key=lambda x: -x[0])
    bull = [{"text": m["good"], "pillar": p["name"]} for _, m, p in cands if m["score"] >= 66][:4]
    bear = [{"text": m["bad"], "pillar": p["name"]} for _, m, p in cands if m["score"] <= 40][:4]

    ranked = sorted(scored, key=lambda p: -p["score"])
    if action == "NO CALL":
        summary = f"There isn't enough clean data on {short} to stand behind a call."
    else:
        best, worst = ranked[0], ranked[-1]
        lead = {"BUY": "clears the bar for a buy", "HOLD": "lands in hold territory", "SELL": "falls short — a sell on these numbers"}[action]
        rnd = lambda v: int(v + 0.5)  # round half up, to match the dial
        summary = (f"{short} scores {rnd(score)} out of 100 and {lead}. "
                   f"It is strongest on {best['name'].lower()} ({rnd(best['score'])}) and weakest on {worst['name'].lower()} ({rnd(worst['score'])}).")
        if caps:
            summary += " " + caps[0]

    def headline(p):
        ms = [m for m in p["metrics"] if m["score"] is not None and m["text"]]
        if not ms:
            return "Yahoo Finance has no usable data for this pillar."
        top = max(ms, key=lambda m: m["weight"])
        rest = sorted([m for m in ms if m is not top], key=lambda m: -abs(m["score"] - 50))
        out = (top["good"] if top["score"] >= 50 else top["bad"]) + "."
        if rest:
            m = rest[0]
            t = m["good"] if m["score"] >= 50 else m["bad"]
            out += " " + t[0].upper() + t[1:] + "."
        return out

    for p in pillars:
        p["headline"] = headline(p)
        for m in p["metrics"]:
            m.pop("good", None), m.pop("bad", None)

    # ═════════════ payload ═════════════
    hist = raw.get("hist")
    price_series = []
    if hist is not None and not hist.empty:
        h = hist["Close"].dropna().iloc[-261:]
        price_series = [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in h.items()]
    prev_close = num(info.get("previousClose")) or num(info.get("regularMarketPreviousClose"))

    keep_a = ("fy", "revenue", "revGrowth", "grossMargin", "opMargin", "ebitdaMargin", "netMargin", "ni", "eps", "epsGrowth",
              "roe", "roce", "roa", "de", "ocf", "fcf", "cover", "epsReported", "oneoff")
    keep_q = ("label", "revenue", "revYoY", "opMargin", "netMargin", "eps", "epsYoY", "ni")
    out = {
        "meta": {
            "symbol": sym, "name": name, "short": short, "exchange": info.get("fullExchangeName") or info.get("exchange") or "",
            "sector": sector, "industry": industry, "currency": cur, "finCurrency": fcur, "price": price,
            "prevClose": prev_close, "changePct": (price / prev_close - 1) * 100 if prev_close else None,
            "marketCap": mcap, "low52": num(info.get("fiftyTwoWeekLow")), "high52": num(info.get("fiftyTwoWeekHigh")),
            "isLender": lender, "lastFY": last["fy"], "fy0": fy0, "fy1": fy1, "analysts": n_rec,
            "asOf": dt.datetime.now().strftime("%d %b %Y, %H:%M"), "website": info.get("website"),
        },
        "verdict": {"action": action, "score": None if score is None else round(score, 1), "summary": summary,
                    "coverage": round(coverage, 2), "caps": caps, "buyAt": BUY_AT, "sellBelow": SELL_BELOW},
        "pillars": pillars, "bull": bull, "bear": bear, "flags": flags, "notes": notes,
        "annual": [{k: r.get(k) for k in keep_a} for r in annual],
        "quarterly": [{k: r.get(k) for k in keep_q} for r in quarterly] if fresh_q else [],
        "ttm": ttm, "dupont": dupont, "surprises": surprises[-4:],
        "forward": {
            "eps": [{"period": fy0, "avg": eps0, "low": g(r0, "low"), "high": g(r0, "high"), "growth": g0, "analysts": g(r0, "numberOfAnalysts")},
                    {"period": fy1, "avg": eps1, "low": g(r1, "low"), "high": g(r1, "high"), "growth": g1, "analysts": g(r1, "numberOfAnalysts")}],
            "epsBase": eps_base, "basisGap": basis_gap, "estCurrency": est_cur,
            "revenue": [{"period": fy0, "avg": rev0, "growth": rg0}, {"period": fy1, "avg": rev1, "growth": rg1}],
            "revisions": {"up30": up30, "down30": dn30, "up7": up7, "down7": dn7, "breadth": breadth},
            "trend": {"d30": d30, "d90": d90},
            "target": {**tgt, "upside": upside}, "rating": {"mean": rec, "key": info.get("recommendationKey"), "analysts": n_rec},
            "ntmEps": ntm, "fwdPE": fwd_pe, "fwdCagr": fwd_cagr,
        },
        "valuation": {"pe": pe, "fwdPE": fwd_pe, "band": band, "peg": peg, "pb": pb, "evEbitda": ev_ebitda,
                      "fcfYield": fcf_yield, "divYield": num(info.get("dividendYield")), "de": last["de"] if not lender else None},
        "price": price_series,
    }
    return clean(out)


def clean(o):
    """Make the payload JSON-safe: NaN/inf -> None, numpy/pandas scalars -> python, floats rounded."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (str, bool)) or o is None:
        return o
    if isinstance(o, int):
        return o
    v = num(o)
    if v is None:
        return None if not isinstance(o, (pd.Timestamp, dt.date)) else str(o)
    return round(v, 4) if abs(v) < 1e6 else round(v, 0)


# ───────────────────────────── cache ─────────────────────────────
_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()
CACHE_SECONDS = 600


def analyze_cached(query: str) -> dict:
    key = (query or "").strip().upper()
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]
    res = analyze(query)
    with _lock:
        _cache[key] = (time.time(), res)
        _cache[res["meta"]["symbol"]] = (time.time(), res)
    return res


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"{'symbol':<14}{'call':<9}{'score':>6}  " + "  ".join(f"{k[:6]:>6}" for k in WEIGHTS) + "   cov")
    for a in sys.argv[1:] or ["RELIANCE"]:
        t0 = time.time()
        try:
            r = analyze(a)
        except AnalysisError as e:
            print(f"{a:<14}ERROR  {e}")
            continue
        v = r["verdict"]
        cells = "  ".join(f"{p['score']:>6.0f}" if p["score"] is not None else f"{'-':>6}" for p in r["pillars"])
        print(f"{r['meta']['symbol']:<14}{v['action']:<9}{(v['score'] or 0):>6.1f}  {cells}   {v['coverage']:.2f}  ({time.time() - t0:.1f}s)")
