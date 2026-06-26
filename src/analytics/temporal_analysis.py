from __future__ import annotations

import pandas as pd


def fraud_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["trans_date_trans_time"] = pd.to_datetime(work["trans_date_trans_time"], errors="coerce")
    work["hour"] = work["trans_date_trans_time"].dt.hour
    return (
        work.groupby("hour")["is_fraud"]
        .agg(transactions="count", fraud_count="sum", fraud_rate="mean")
        .reset_index()
        .sort_values("hour")
    )
