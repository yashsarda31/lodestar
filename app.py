"""
Lodestar - type a ticker, get a full fundamental read and a buy / hold / sell call.

    python app.py              # starts http://127.0.0.1:8820 and opens the browser
    python app.py --no-browser --port 9000

Needs: fastapi, uvicorn, yfinance, pandas  (pip install -r requirements.txt)
"""
from __future__ import annotations

import argparse
import os
import threading
import webbrowser

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

import engine

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Lodestar", docs_url=None, redoc_url=None)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/api/analyze")
def analyze(ticker: str = Query(..., min_length=1, max_length=60)):
    try:
        return JSONResponse(engine.analyze_cached(ticker), headers={"Cache-Control": "no-store"})
    except engine.AnalysisError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:  # Yahoo hiccup, rate limit, unexpected statement shape
        return JSONResponse({"error": f"Yahoo Finance didn't return usable data just now ({type(e).__name__}). Try again in a moment."},
                            status_code=502)


@app.get("/api/search")
def search(q: str = Query("", max_length=60)):
    q = q.strip()
    return {"results": engine.search(q) if len(q) >= 2 else []}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8820)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}")).start()
    print(f"Lodestar running at http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
