"""
signal_engines.py
------------------------------------
CoreSignalX Technical Indicator Engine

Computes RSI, MACD, Bollinger Bands,
Stochastic Oscillator, and Moving Average Crossovers.
Returns normalized 0–100 signal strengths and direction flags.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any

warnings.filterwarnings("ignore")


# --------------------------------------------------
#  Individual Indicator Functions
# --------------------------------------------------

def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(period).mean()
    roll_down = pd.Series(loss).rolling(period).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    df["RSI"] = rsi
    return df["RSI"]


def calc_macd(df: pd.DataFrame,
              short: int = 12,
              long: int = 26,
              signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence."""
    exp1 = df["Close"].ewm(span=short, adjust=False).mean()
    exp2 = df["Close"].ewm(span=long, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]
    return df[["MACD", "Signal", "MACD_Hist"]]


def calc_bollinger(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """Bollinger Bands."""
    rolling_mean = df["Close"].rolling(window).mean()
    rolling_std = df["Close"].rolling(window).std()
    df["BB_Upper"] = rolling_mean + (rolling_std * num_std)
    df["BB_Lower"] = rolling_mean - (rolling_std * num_std)
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]) * 100
    return df[["BB_Upper", "BB_Lower", "BB_Position"]]


def calc_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator."""
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    df["%K"] = (df["Close"] - low_min) * 100 / (high_max - low_min)
    df["%D"] = df["%K"].rolling(window=d_period).mean()
    return df[["%K", "%D"]]


def calc_ma_cross(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.Series:
    """Moving average crossover signal."""
    df["MA_Short"] = df["Close"].rolling(window=short).mean()
    df["MA_Long"] = df["Close"].rolling(window=long).mean()
    df["MA_Diff"] = df["MA_Short"] - df["MA_Long"]
    df["MA_Signal"] = np.where(df["MA_Diff"] > 0, 1, -1)
    return df["MA_Signal"]


# --------------------------------------------------
#  Composite Engine
# --------------------------------------------------

def generate_signal_scores(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all indicators and return numeric 0–100 scores + textual summary.
    """

    if df.empty or len(df) < 50:
        return {"error": "Insufficient data"}

    try:
        calc_rsi(df)
        calc_macd(df)
        calc_bollinger(df)
        calc_stochastic(df)
        calc_ma_cross(df)

        latest = df.iloc[-1]

        # RSI
        rsi_val = latest["RSI"]
        rsi_score = np.clip(100 - abs(50 - rsi_val) * 2, 0, 100)

        # MACD
        macd_score = np.clip((latest["MACD_Hist"] * 500) + 50, 0, 100)

        # Bollinger
        bb_pos = latest["BB_Position"]
        bb_score = 100 - abs(50 - bb_pos)

        # Stochastic
        stoch_val = latest["%K"]
        stoch_score = np.clip(100 - abs(50 - stoch_val), 0, 100)

        # Moving Average
        ma_signal = latest["MA_Signal"]
        ma_score = 80 if ma_signal > 0 else 20

        # Weighted composite
        weights = {
            "RSI": 0.25,
            "MACD": 0.25,
            "Bollinger": 0.15,
            "Stochastic": 0.15,
            "MA": 0.20,
        }
        total_score = (
            rsi_score * weights["RSI"]
            + macd_score * weights["MACD"]
            + bb_score * weights["Bollinger"]
            + stoch_score * weights["Stochastic"]
            + ma_score * weights["MA"]
        )

        verdict = "Neutral"
        if total_score >= 70:
            verdict = "Bullish"
        elif total_score <= 30:
            verdict = "Bearish"

        summary = (
            f"RSI {rsi_val:.1f}, MACD Hist {latest['MACD_Hist']:.4f}, "
            f"Bollinger pos {bb_pos:.1f}, Stoch {stoch_val:.1f}, MA signal {'Bullish' if ma_signal>0 else 'Bearish'}"
        )

        return {
            "RSI": round(rsi_val, 2),
            "MACD_Hist": round(latest["MACD_Hist"], 4),
            "BB_Position": round(bb_pos, 2),
            "%K": round(stoch_val, 2),
            "MA_Signal": int(ma_signal),
            "score": round(total_score, 2),
            "verdict": verdict,
            "summary": summary,
        }

    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------
#  Example usage
# --------------------------------------------------
if __name__ == "__main__":
    import yfinance as yf
    print("Testing signal_engines...")
    df = yf.download("AAPL", period="6mo", interval="1d", progress=False)
    out = generate_signal_scores(df)
    print(out)
