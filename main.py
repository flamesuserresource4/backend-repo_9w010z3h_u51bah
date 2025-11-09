import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Avoid heavy optional deps at import time to keep server booting even if install failed
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    from joblib import load  # type: ignore
except Exception:
    def load(*args, **kwargs):  # type: ignore
        raise ImportError('joblib not available')

from database import db

app = FastAPI(title="Fintweet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    ticker: str
    predicted_close: float
    message: str


class StockSnapshot(BaseModel):
    ticker: str
    open: float
    close: float
    volume: float
    percent_change: float


class HistoryResponse(BaseModel):
    ticker: str
    prices: List[dict]
    sentiment: List[dict]


DEFAULT_TRENDING = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
DEFAULT_USER_ID = "default"


@app.get("/")
def read_root():
    return {"message": "Fintweet Backend is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
        "optional_deps": {
            "yfinance": bool(yf),
            "numpy": bool(np),
            "pandas": bool(pd),
        }
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


def _synthetic_trending(symbols: List[str]) -> List[StockSnapshot]:
    out: List[StockSnapshot] = []
    for i, sym in enumerate(symbols):
        base = 100 + i * 10
        close = base * 1.01
        open_p = base
        volume = 1_000_000 + i * 50_000
        pct = ((close - open_p) / open_p) * 100
        out.append(StockSnapshot(ticker=sym, open=open_p, close=close, volume=volume, percent_change=pct))
    return out


@app.get("/trending", response_model=List[StockSnapshot])
def get_trending(tickers: Optional[str] = None):
    symbols = [t.strip().upper() for t in (tickers.split(",") if tickers else DEFAULT_TRENDING)]
    if yf is None or pd is None:
        return _synthetic_trending(symbols)

    data: List[StockSnapshot] = []
    for sym in symbols:
        try:
            info = yf.Ticker(sym)
            hist = info.history(period="5d")
            if hist.empty:
                continue
            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else latest
            open_p = float(latest.get("Open", float("nan")))
            close_p = float(latest.get("Close", float("nan")))
            volume = float(latest.get("Volume", 0.0))
            prev_close = float(prev.get("Close", close_p)) or close_p
            pct = ((close_p - prev_close) / prev_close) * 100 if prev_close else 0.0
            data.append(StockSnapshot(ticker=sym, open=open_p, close=close_p, volume=volume, percent_change=pct))
        except Exception:
            continue
    return data


@app.get("/predict", response_model=PredictionResponse)
def predict_close(ticker: str = Query(..., description="Ticker symbol, e.g., AAPL")):
    sym = ticker.upper()

    # If yfinance not available, return a deterministic synthetic value
    if yf is None:
        base = 150.0
        pred = base
        return PredictionResponse(
            ticker=sym,
            predicted_close=round(pred, 2),
            message=f"Predicted next-day closing price for {sym}: {round(pred, 2)}",
        )

    # Attempt to load a trained model; fallback to naive prediction when missing
    model_path_candidates = [
        Path("model.joblib"),
        Path("models/model.joblib"),
        Path("/app/model.joblib"),
        Path("/workspace/model.joblib"),
    ]

    model = None
    for p in model_path_candidates:
        if p.exists():
            try:
                model = load(p)
                break
            except Exception:
                model = None

    # Prepare simple features from recent history
    try:
        hist = yf.Ticker(sym).history(period="30d")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {sym}")
        last_close = float(hist["Close"].iloc[-1])
        last_return = float((hist["Close"].pct_change().iloc[-1] or 0.0))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch market data")

    if model is not None and np is not None:
        try:
            X = np.array([[last_close, last_return]], dtype=float)
            pred = float(model.predict(X)[0])
            return PredictionResponse(
                ticker=sym,
                predicted_close=round(pred, 2),
                message=f"Predicted next-day closing price for {sym}: {round(pred, 2)}",
            )
        except Exception:
            pass

    # Naive heuristic fallback
    naive_pred = last_close * (1 + last_return * 0.5)
    return PredictionResponse(
        ticker=sym,
        predicted_close=round(float(naive_pred), 2),
        message=f"Predicted next-day closing price for {sym}: {round(float(naive_pred), 2)}",
    )


@app.get("/history", response_model=HistoryResponse)
def history(ticker: str = Query(...)):
    sym = ticker.upper()

    if yf is None or pd is None:
        # synthetic series
        dates = [f"2024-09-{str(i).zfill(2)}" for i in range(1, 31)]
        prices = [{"date": d, "close": 100 + i * 0.5} for i, d in enumerate(dates)]
        sentiment = []
        for i, d in enumerate(dates):
            pos = 40 + (i % 10)
            neg = 20 - (i % 5)
            neu = 100 - pos - max(0, neg)
            sentiment.append({"date": d, "positive": float(pos), "negative": float(max(0, neg)), "neutral": float(neu)})
        return {"ticker": sym, "prices": prices, "sentiment": sentiment}

    try:
        hist = yf.Ticker(sym).history(period="3mo")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {sym}")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch market data")

    hist = hist.tail(60)
    prices = (
        hist.reset_index()[["Date", "Close"]]
        .rename(columns={"Date": "date", "Close": "close"})
    )
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")

    returns = hist["Close"].pct_change().fillna(0)
    sentiment = []
    for dt, r in zip(hist.index.strftime("%Y-%m-%d"), returns):
        pos = float(max(0.0, r) * 100)
        neg = float(max(0.0, -r) * 100)
        neu = float(max(0.0, 1.0 - (abs(r) * 100)))
        total = pos + neg + neu
        if total == 0:
            pos = neg = 0.0
            neu = 100.0
            total = 100.0
        sentiment.append({
            "date": dt,
            "positive": round(100 * pos / total, 2),
            "negative": round(100 * neg / total, 2),
            "neutral": round(100 * neu / total, 2),
        })

    return {
        "ticker": sym,
        "prices": prices.to_dict(orient="records"),
        "sentiment": sentiment,
    }


@app.get("/watchlist")
def get_watchlist(user_id: str = DEFAULT_USER_ID):
    if db is None:
        return {"user_id": user_id, "tickers": []}
    doc = db["watchlist"].find_one({"user_id": user_id})
    if not doc:
        db["watchlist"].insert_one({"user_id": user_id, "tickers": []})
        return {"user_id": user_id, "tickers": []}
    return {"user_id": user_id, "tickers": doc.get("tickers", [])}


@app.post("/watchlist")
def toggle_watchlist(ticker: str, user_id: str = DEFAULT_USER_ID):
    sym = ticker.upper()
    if db is None:
        return {"user_id": user_id, "tickers": [sym]}
    doc = db["watchlist"].find_one({"user_id": user_id})
    if not doc:
        db["watchlist"].insert_one({"user_id": user_id, "tickers": [sym]})
        return {"user_id": user_id, "tickers": [sym]}
    tickers = set(doc.get("tickers", []))
    if sym in tickers:
        tickers.remove(sym)
    else:
        tickers.add(sym)
    db["watchlist"].update_one({"user_id": user_id}, {"$set": {"tickers": sorted(list(tickers))}})
    return {"user_id": user_id, "tickers": sorted(list(tickers))}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
