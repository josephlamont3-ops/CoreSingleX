"""
signal_engines.py
------------------------------------
CoreSignalX Technical Indicator Engine

Calculates RSI, MACD, Bollinger Bands,
Stochastic Oscillator, and Moving Average Crossovers.
Returns normalized 0–100 signal strengths and direction flags.
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# --------------------------------------------------
#  Helper indicators (no external dependency needed)
# --------------------------------------------------
def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (
