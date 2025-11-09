"""
Database Schemas for Fintweet

Each Pydantic model represents a collection in MongoDB.
Class name lowercased = collection name.
"""

from pydantic import BaseModel, Field
from typing import Optional, List

class WatchItem(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g., AAPL")
    notes: Optional[str] = Field(None, description="Optional notes for the ticker")

class Prediction(BaseModel):
    ticker: str
    predicted_close: float
    message: str

class SentimentPoint(BaseModel):
    date: str
    positive: float
    negative: float
    neutral: float

class PricePoint(BaseModel):
    date: str
    close: float

class StockSnapshot(BaseModel):
    ticker: str
    open: float
    close: float
    volume: float
    percent_change: float

class Watchlist(BaseModel):
    user_id: str
    tickers: List[str] = []
