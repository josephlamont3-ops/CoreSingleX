"""
data_fetch.py
------------------------------------
CoreSignalX Market, Macro, and News Data Collectors

This module fetches:
 • Equity & crypto OHLCV data (yfinance + Alpha Vantage)
 • Options summary data (open interest / put–call ratio)
 • Macro indicators (FRED)
 • News headlines (NewsAPI)
"""

import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


# --------------------------------------------------
#  Utility
# --------------------------------------------------
def _safe_request(url: str, params: dict = None, max_retries: int = 3) -> Optional[dict]:
    """Helper with retries and error trapping."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[WARN] {e}")
        time.sleep(1.5)
    return None


# --------------------------------------------------
#  Alpha Vantage price fetch
# --------------------------------------------------
def get_alpha_vantage_prices(symbol: str, api_key: str) -> pd.DataFrame:
    """
    Intraday prices for a ticker via Alpha Vantage.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "5min",
        "apikey": api_key,
    }
    data = _safe_request(url, params)
    if not data or "Time Series (5min)" not in data:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(data["Time Series (5min)"], orient="index", dtype=float)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# --------------------------------------------------
#  yfinance fallback
# --------------------------------------------------
def get_yf_prices(symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Generic OHLCV via Yahoo Finance."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"[ERROR] yfinance failed for {symbol}: {e}")
        return pd.DataFrame()


# --------------------------------------------------
#  Options data summary
# --------------------------------------------------
def get_options_summary(symbol: str) -> Dict[str, Any]:
    """
    Basic options-market snapshot using yfinance.
    Returns aggregated open interest and call/put ratios.
    """
    try:
        tk = yf.Ticker(symbol)
        exps = tk.options
        if not exps:
            return {}
        exp = exps[0]
        opt = tk.option_chain(exp)
        calls, puts = opt.calls, opt.puts
        call_oi = calls["openInterest"].sum()
        put_oi = puts["openInterest"].sum()
        total_oi = call_oi + put_oi
        ratio = (call_oi / total_oi) if total_oi > 0 else 0
        return {
            "expiration": exp,
            "call_open_interest": int(call_oi),
            "put_open_interest": int(put_oi),
            "put_call_ratio": round(put_oi / call_oi, 3) if call_oi else None,
            "call_ratio": round(ratio, 3),
        }
    except Exception as e:
        print(f"[ERROR] get_options_summary({symbol}) → {e}")
        return {}


# --------------------------------------------------
#  Macro (FRED)
# --------------------------------------------------
def get_macro_data(fred_api_key: str) -> pd.DataFrame:
    """
    Downloads a few macro indicators:
        CPI, Fed Funds Rate, and 10Y Treasury Yield
    """
    try:
        series = {
            "CPI": "CPIAUCSL",
            "FEDFUNDS": "FEDFUNDS",
            "GS10": "GS10",
        }
        frames = []
        for name, sid in series.items():
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {"series_id": sid, "api_key": fred_api_key, "file_type": "json"}
            data = _safe_request(url, params)
            if not data:
                continue
            df = pd.DataFrame(data["observations"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df.rename(columns={"value": name}, inplace=True)
            df = df[["date", name]]
            frames.append(df)
        if frames:
            out = frames[0]
            for f in frames[1:]:
                out = pd.merge(out, f, on="date", how="outer")
            return out.sort_values("date")
    except Exception as e:
        print(f"[ERROR] get_macro_data: {e}")
    return pd.DataFrame()


# --------------------------------------------------
#  NewsAPI
# --------------------------------------------------
def get_news_headlines(api_key: str, query: str = "stocks") -> List[str]:
    """
    Top 10 business headlines for given query.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": 10,
        "sortBy": "relevancy",
        "apiKey": api_key,
    }
    data = _safe_request(url, params)
    if not data or "articles" not in data:
        return []
    return [a["title"] for a in data["articles"] if a.get("title")]


# --------------------------------------------------
#  High-level aggregator
# --------------------------------------------------
def get_live_data(symbol: str,
                  alpha_key: str = "",
                  fred_key: str = "",
                  news_key: str = "") -> Dict[str, Any]:
    """
    Fetch all relevant data for one ticker symbol.
    """
    result = {"symbol": symbol, "timestamp": datetime.utcnow().isoformat()}
    # Price
    if alpha_key:
        df = get_alpha_vantage_prices(symbol, alpha_key)
        if df.empty:
            df = get_yf_prices(symbol)
    else:
        df = get_yf_prices(symbol)
    result["price_data"] = df.tail(200)
    # Options
    result["options_summary"] = get_options_summary(symbol)
    # Macro
    result["macro_data"] = get_macro_data(fred_key) if fred_key else pd.DataFrame()
    # News
    result["news"] = get_news_headlines(news_key, query=symbol) if news_key else []
    return result


# --------------------------------------------------
#  Test harness
# --------------------------------------------------
if __name__ == "__main__":
    print("Testing data_fetch module ...")
    ticker = "AAPL"
    data = get_live_data(ticker)
    print(f"{ticker}: {len(data['price_data'])} rows of price data")
    print("Options:", data["options_summary"])
    print("News samples:", data["news"][:2])
