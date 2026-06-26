from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BEHAVIORAL_NUMERIC_COLUMNS = [
    "time_since_last_trx",
    "count_trx_1h",
    "count_trx_24h",
    "avg_amt_7d",
    "amt_ratio_to_avg",
]

NUMERIC_COLUMNS = [
    "amt",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "age",
    "hour",
    "dayofweek",
    "month",
    "distance_km",
    *BEHAVIORAL_NUMERIC_COLUMNS,
]

CATEGORICAL_COLUMNS = ["category", "gender", "state"]


@dataclass
class FeatureArtifacts:
    encoder: OneHotEncoder
    scaler: StandardScaler
    feature_names: list[str]


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class Preprocessor:
    """Feature extraction pipeline for stateless and behavioral fraud features."""

    def __init__(
        self,
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> None:
        self.numeric_columns = numeric_columns or NUMERIC_COLUMNS
        self.categorical_columns = categorical_columns or CATEGORICAL_COLUMNS

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["_original_order"] = np.arange(len(out))

        self._ensure_base_columns(out)
        out["trans_date_trans_time"] = pd.to_datetime(out["trans_date_trans_time"], errors="coerce")
        out["dob"] = pd.to_datetime(out["dob"], errors="coerce")
        out["amt"] = pd.to_numeric(out["amt"], errors="coerce").fillna(0)

        # Chronological ordering is mandatory for shift and rolling windows.
        out = out.sort_values(["trans_date_trans_time", "cc_num", "_original_order"], kind="mergesort")

        out["hour"] = out["trans_date_trans_time"].dt.hour.fillna(0).astype(int)
        out["dayofweek"] = out["trans_date_trans_time"].dt.dayofweek.fillna(0).astype(int)
        out["month"] = out["trans_date_trans_time"].dt.month.fillna(0).astype(int)
        out["age"] = ((out["trans_date_trans_time"] - out["dob"]).dt.days / 365.25).fillna(0).clip(lower=0)
        out["distance_km"] = haversine_km(out["lat"], out["long"], out["merch_lat"], out["merch_long"]).fillna(0)

        out = self._add_behavioral_features(out)

        for col in self.numeric_columns:
            if col not in out:
                out[col] = 0
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

        for col in self.categorical_columns:
            if col not in out:
                out[col] = "unknown"
            out[col] = out[col].astype(str).fillna("unknown")

        return out.sort_values("_original_order").drop(columns=["_original_order"])

    def fit_transform_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureArtifacts]:
        enriched = self.add_features(df)
        scaler = StandardScaler()
        encoder = make_encoder()

        numeric = scaler.fit_transform(enriched[self.numeric_columns])
        categorical = encoder.fit_transform(enriched[self.categorical_columns])
        cat_names = encoder.get_feature_names_out(self.categorical_columns).tolist()
        feature_names = self.numeric_columns + cat_names

        X = pd.DataFrame(np.hstack([numeric, categorical]), columns=feature_names, index=enriched.index)
        return X, FeatureArtifacts(encoder=encoder, scaler=scaler, feature_names=feature_names)

    def transform_features(
        self,
        df: pd.DataFrame,
        encoder: OneHotEncoder,
        scaler: StandardScaler,
    ) -> pd.DataFrame:
        enriched = self.add_features(df)
        numeric = scaler.transform(enriched[self.numeric_columns])
        categorical = encoder.transform(enriched[self.categorical_columns])
        cat_names = encoder.get_feature_names_out(self.categorical_columns).tolist()
        feature_names = self.numeric_columns + cat_names
        return pd.DataFrame(np.hstack([numeric, categorical]), columns=feature_names, index=enriched.index)

    def _ensure_base_columns(self, out: pd.DataFrame) -> None:
        defaults = {
            "trans_date_trans_time": pd.NaT,
            "dob": pd.NaT,
            "cc_num": "unknown",
            "amt": 0,
            "city_pop": 0,
            "lat": 0,
            "long": 0,
            "merch_lat": 0,
            "merch_long": 0,
            "category": "unknown",
            "gender": "unknown",
            "state": "unknown",
        }
        for col, default in defaults.items():
            if col not in out:
                out[col] = default

        out["cc_num"] = out["cc_num"].astype(str).fillna("unknown")
        for col in ["city_pop", "lat", "long", "merch_lat", "merch_long"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    def _add_behavioral_features(self, out: pd.DataFrame) -> pd.DataFrame:
        out["time_since_last_trx"] = (
            out.groupby("cc_num", sort=False)["trans_date_trans_time"]
            .diff()
            .dt.total_seconds()
            .fillna(0)
            .clip(lower=0)
        )

        valid_time = out["trans_date_trans_time"].notna()
        out["count_trx_1h"] = 0.0
        out["count_trx_24h"] = 0.0
        out["avg_amt_7d"] = out["amt"]

        if valid_time.any():
            valid = out.loc[valid_time].copy()
            out.loc[valid.index, "count_trx_1h"] = self._rolling_count(valid, "1h")
            out.loc[valid.index, "count_trx_24h"] = self._rolling_count(valid, "24h")
            out.loc[valid.index, "avg_amt_7d"] = self._rolling_mean(valid, "7d")

        out["avg_amt_7d"] = out["avg_amt_7d"].replace([np.inf, -np.inf], np.nan)
        out["avg_amt_7d"] = out["avg_amt_7d"].fillna(out["amt"])
        denominator = out["avg_amt_7d"].where(out["avg_amt_7d"].abs() > 1e-9, out["amt"])
        out["amt_ratio_to_avg"] = (out["amt"] / denominator.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        out["amt_ratio_to_avg"] = out["amt_ratio_to_avg"].fillna(1.0)

        return out

    @staticmethod
    def _rolling_count(df: pd.DataFrame, window: str) -> pd.Series:
        def compute(group: pd.DataFrame) -> pd.Series:
            rolled = (
                group.set_index("trans_date_trans_time")["amt"]
                .rolling(window, closed="left")
                .count()
            )
            return pd.Series(rolled.to_numpy(), index=group.index)

        return (
            df.groupby("cc_num", sort=False, group_keys=False)
            .apply(compute)
            .reindex(df.index)
            .fillna(0)
        )

    @staticmethod
    def _rolling_mean(df: pd.DataFrame, window: str) -> pd.Series:
        def compute(group: pd.DataFrame) -> pd.Series:
            rolled = (
                group.set_index("trans_date_trans_time")["amt"]
                .rolling(window, closed="left")
                .mean()
            )
            return pd.Series(rolled.to_numpy(), index=group.index)

        return df.groupby("cc_num", sort=False, group_keys=False).apply(compute).reindex(df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    return Preprocessor().add_features(df)


def fit_transform_features(df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureArtifacts]:
    return Preprocessor().fit_transform_features(df)


def transform_features(df: pd.DataFrame, encoder: OneHotEncoder, scaler: StandardScaler) -> pd.DataFrame:
    return Preprocessor().transform_features(df, encoder, scaler)


# Public exports are intentionally rebound to the stateful implementation.
# Existing imports from src.inference.preprocessing keep working unchanged.
from src.inference.stateful_preprocessing import (  # noqa: E402,F401
    BEHAVIORAL_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    FeatureArtifacts,
    Preprocessor,
    add_features,
    fit_transform_features,
    haversine_km,
    make_encoder,
    transform_features,
)
