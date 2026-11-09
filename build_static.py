"""
Build the hosted (GitHub Pages) version of Lodestar into docs/.

GitHub Pages serves static files only, so it can't run app.py. This script runs the same engine
over every symbol in universe.txt and writes one JSON file per company plus a search index:

    docs/index.html          static/index.html with window.LODESTAR_STATIC = 1
    docs/data/index.json     [{symbol, name, short, exchange, action, score, file}], builtAt
    docs/data/<SYMBOL>.json  the exact payload /api/analyze would return

If a symbol fails (Yahoo hiccup / rate limit), its previous JSON is kept so the site never loses a
company because of one bad run. Runs daily via .github/workflows/pages.yml.

    python build_static.py               # full universe
    python build_static.py RELIANCE AAPL # just these (others' files kept)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import engine

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
DATA = os.path.join(DOCS, "data")
WORKERS = 2


def fname(sym: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]", "_", sym) + ".json"


def universe() -> list[str]:
    with open(os.path.join(HERE, "universe.txt"), encoding="utf-8") as f:
        return [s.strip() for s in f if s.strip() and not s.startswith("#")]


def one(sym: str):
    # Yahoo rate-limiting surfaces as "couldn't find a listed company" for perfectly valid symbols,
    # so that error is retried with backoff too; other AnalysisErrors (no statements) are final.
    err = ""
    for attempt in range(4):
        try:
            return sym, engine.analyze(sym), None
        except engine.AnalysisError as e:
            err = str(e)
            if "find a listed company" not in err:
                return sym, None, err
        except Exception as e:  # transient Yahoo failure
            err = f"{type(e).__name__}: {e}"
        time.sleep(8 * (attempt + 1))
    return sym, None, err


def entry(d: dict) -> dict:
    m, v = d["meta"], d["verdict"]
    return {"symbol": m["symbol"], "name": m.get("name") or m["symbol"], "short": m.get("short"),
            "exchange": m.get("exchange") or "", "action": v.get("action"), "score": v.get("score"),
            "file": fname(m["symbol"])}


def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(DATA, exist_ok=True)
    syms = argv or universe()

    # previous index: keep entries we don't refresh (or that fail) this run
    idx_path = os.path.join(DATA, "index.json")
    old = {}
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            old = {it["symbol"]: it for it in json.load(f).get("items", [])}
    keep_syms = set(universe())
    items = {s: it for s, it in old.items() if s in keep_syms or s in syms}

    t0, ok, failed = time.time(), 0, []
    with ThreadPoolExecutor(WORKERS) as ex:
        for fut in as_completed([ex.submit(one, s) for s in syms]):
            sym, d, err = fut.result()
            if d is None:
                failed.append((sym, err))
                print(f"  FAIL {sym:<16} {err}")
                continue
            e = entry(d)
            with open(os.path.join(DATA, e["file"]), "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
            items.pop(sym, None)
            items[e["symbol"]] = e
            ok += 1
            print(f"  ok   {e['symbol']:<16} {e['action'] or '-':<5} {e['score'] if e['score'] is not None else '-'}")

    # drop index entries whose file is missing; delete orphan files no longer in the universe
    items = {s: it for s, it in items.items() if os.path.exists(os.path.join(DATA, it["file"]))}
    live = {it["file"] for it in items.values()} | {"index.json"}
    for fn in os.listdir(DATA):
        if fn not in live:
            os.remove(os.path.join(DATA, fn))

    ist = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"builtAt": ist.strftime("%d %b %Y, %H:%M IST"),
                   "items": sorted(items.values(), key=lambda x: x["symbol"])},
                  f, ensure_ascii=False, separators=(",", ":"))

    with open(os.path.join(HERE, "static", "index.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</head>", "<script>window.LODESTAR_STATIC=1</script>\n</head>", 1)
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    open(os.path.join(DOCS, ".nojekyll"), "w").close()

    print(f"\n{ok}/{len(syms)} refreshed, {len(failed)} failed, {len(items)} in index  ({time.time() - t0:.0f}s)")
    # fail the run only if Yahoo was broadly unreachable - partial failures keep yesterday's files
    return 1 if syms and ok < 0.5 * len(syms) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
