# src/predict.py

import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "creditfraud_pipeline.pkl"
)

model = joblib.load(MODEL_PATH)


def predict(transaction_df: pd.DataFrame):
    """
    transaction_df: DataFrame with same columns as training data (except target)
    """
    return model.predict(transaction_df)


def predict_proba(transaction_df: pd.DataFrame):
    """
    Get fraud probabilities for transactions.

    Parameters:
    -----------
    transaction_df : pd.DataFrame
        DataFrame with same columns as training data (except target)

    Returns:
    --------
    probabilities : array
        Array of fraud probabilities (0.0 to 1.0)
    """
    return model.predict_proba(transaction_df)[:, 1]
