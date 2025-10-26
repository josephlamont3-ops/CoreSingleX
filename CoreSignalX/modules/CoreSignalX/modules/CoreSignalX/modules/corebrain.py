"""
corebrain.py
------------------------------------
CoreSignalX "CoreBrain" AI fusion engine.

Combines multiple analytical layers:
 - Technical indicators (from signal_engines)
 - Options flow (from data_fetch)
 - Macro metrics (from data_fetch)
 - Sentiment data (from sentiment_nlp)

Outputs:
 - CoreScore: 0–100 composite confidence
 - Core Verdict label: Favorable / Watch / Neutral / Avoid
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from datetime import datetime

MODEL_PATH = "database/corebrain_model.pkl"

# --------------------------------------------------
#  Helper functions
# --------------------------------------------------
def _init_model() -> GradientBoostingRegressor:
    """Creates a small default model if no trained model is found."""
    model = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    return model


def _train_baseline_model(model_path: str = MODEL_PATH) -> GradientBoostingRegressor:
    """
    Builds a quick baseline model on synthetic data so that CoreBrain
    always has something to use even before real training.
    """
    np.random.seed(42)
    X = np.random.rand(1000, 6)
    y = (
        0.3 * X[:, 0]
        + 0.2 * X[:, 1]
        + 0.1 * X[:, 2]
        + 0.15 * X[:, 3]
        + 0.15 * X[:, 4]
        + 0.1 * X[:, 5]
    ) * 100
    model = _init_model()
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model


def _load_model() -> GradientBoostingRegressor:
    """Loads the CoreBrain model or trains a baseline if missing."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    return _train_baseline_model()


def _normalize_features(features: Dict[str, float]) -> np.ndarray:
    """Converts dict of raw features to scaled array for model input."""
    vals = np.array(list(features.values())).reshape(1, -1)
    scaler = MinMaxScaler()
    scaler.fit(np.vstack([np.zeros_like(vals), np.ones_like(vals) * 100]))
    return scaler.transform(vals)


# --------------------------------------------------
#  Main CoreBrain Engine
# --------------------------------------------------
def run_corebrain(
    tech_signals: Dict[str, Any],
    options_summary: Dict[str, Any],
    macro_df: pd.DataFrame,
    sentiment_score: float = 50.0,
) -> Dict[str, Any]:
    """
    CoreSignalX AI fusion function.

    tech_signals: output from signal_engines.generate_signal_scores()
    options_summary: dict from data_fetch.get_options_summary()
    macro_df: macroeconomic indicators
    sentiment_score: -100 to +100 (neutral = 0)

    Returns: dict with CoreScore, label, and feature importance summary.
    """

    # --- Defensive defaults
    if "score" not in tech_signals:
        return {"error": "Invalid technical input"}

    model = _load_model()

    # --- Build feature vector
    features = {
        "tech_score": tech_signals.get("score", 50),
        "put_call_ratio": options_summary.get("put_call_ratio", 1.0),
        "call_ratio": options_summary.get("call_ratio", 0.5) * 100,
        "macro_rate": macro_df["FEDFUNDS"].iloc[-1] if not macro_df.empty and "FEDFUNDS" in macro_df else 5.0,
        "macro_cpi": macro_df["CPI"].iloc[-1] if not macro_df.empty and "CPI" in macro_df else 300.0,
        "sentiment": sentiment_score + 50,  # shift from -100–100 → 0–100
    }

    X_scaled = _normalize_features(features)
    core_score = float(np.clip(model.predict(X_scaled)[0], 0, 100))

    # --- Derive verdict
    if core_score >= 75:
        label = "🟢 Favorable Setup"
    elif core_score >= 60:
        label = "🔵 Strong Watch"
    elif core_score >= 40:
        label = "🟡 Neutral / Wait"
    else:
        label = "🔴 Avoid"

    # --- Feature breakdown (explainability)
    imp = getattr(model, "feature_importances_", None)
    if imp is not None:
        imp_map = dict(zip(features.keys(), imp))
    else:
        imp_map = {k: 1 / len(features) for k in features}

    breakdown = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)

    return {
        "CoreScore": round(core_score, 2),
        "label": label,
        "timestamp": datetime.utcnow().isoformat(),
        "features": features,
        "importance": breakdown,
    }


# --------------------------------------------------
#  Public convenience wrapper
# --------------------------------------------------
def get_core_verdict(ticker: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for app.py — combines all layers.
    """
    from modules.signal_engines import generate_signal_scores

    price_data = market_data.get("price_data", pd.DataFrame())
    tech = generate_signal_scores(price_data)
    options_summary = market_data.get("options_summary", {})
    macro_df = market_data.get("macro_data", pd.DataFrame())
    sentiment_score = market_data.get("sentiment", 0.0)

    result = run_corebrain(tech, options_summary, macro_df, sentiment_score)
    result["summary"] = (
        f"{ticker}: {result['label']} "
        f"— CoreScore {result['CoreScore']} "
        f"(Tech {tech.get('score',0):.1f}, Sent {sentiment_score:+.1f})"
    )
    return result


# --------------------------------------------------
#  Example test
# --------------------------------------------------
if __name__ == "__main__":
    import yfinance as yf
    from modules.signal_engines import generate_signal_scores
    from modules.data_fetch import get_options_summary, get_macro_data

    print("Testing corebrain ...")
    df = yf.download("AAPL", period="6mo", interval="1d", progress=False)
    tech = generate_signal_scores(df)
    opts = get_options_summary("AAPL")
    macro = pd.DataFrame({"FEDFUNDS": [5.25], "CPI": [305.4]})
    out = run_corebrain(tech, opts, macro, sentiment_score=20)
    print(out)
