from __future__ import annotations

import pandas as pd


def fraud_by_gender(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("gender")["is_fraud"]
        .agg(transactions="count", fraud_count="sum", fraud_rate="mean")
        .reset_index()
        .sort_values("fraud_rate", ascending=False)
    )
