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
    "amt_zscore_card",
    "is_new_merchant",
    "is_new_category",
    "is_new_state",
    "geo_velocity_kmh",
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
    preprocessor: Preprocessor | None = None


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
    """Stateful feature extraction pipeline for card-level fraud behavior."""

    def __init__(
        self,
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        history_window_days: int = 8,
    ) -> None:
        self.numeric_columns = numeric_columns or NUMERIC_COLUMNS
        self.categorical_columns = categorical_columns or CATEGORICAL_COLUMNS
        self.history_window_days = history_window_days
        self.is_fitted = False
        self.history_tail_: pd.DataFrame | None = None
        self.card_amount_profile_: pd.DataFrame | None = None
        self.seen_pairs_: dict[str, pd.MultiIndex] = {}

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        base = self._prepare_base(df)
        self._fit_customer_state(base)
        self.is_fitted = True
        return self

    def add_features(self, df: pd.DataFrame, use_history: bool = True) -> pd.DataFrame:
        current = self._prepare_base(df)
        current["_is_current"] = True

        if use_history and self.is_fitted and self.history_tail_ is not None and not self.history_tail_.empty:
            history = self.history_tail_.copy()
            history["_original_order"] = -1
            history["_is_current"] = False
            combined = pd.concat([history, current], ignore_index=True, sort=False)
        else:
            combined = current

        combined = combined.sort_values(
            ["trans_date_trans_time", "cc_num", "_is_current", "_original_order"],
            kind="mergesort",
        )
        combined = self._add_calendar_geo_features(combined)
        combined = self._add_behavioral_features(combined)
        combined = self._add_profile_features(combined, use_history=use_history)
        combined = self._add_seen_flags(combined, use_history=use_history)
        combined = self._finalize_columns(combined)

        out = combined.loc[combined["_is_current"]].copy()
        return out.sort_values("_original_order").drop(columns=["_original_order", "_is_current"])

    def fit_transform_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureArtifacts]:
        enriched = self.add_features(df, use_history=False)
        self._fit_customer_state(self._prepare_base(df))
        self.is_fitted = True

        scaler = StandardScaler()
        encoder = make_encoder()

        numeric = scaler.fit_transform(enriched[self.numeric_columns])
        categorical = encoder.fit_transform(enriched[self.categorical_columns])
        cat_names = encoder.get_feature_names_out(self.categorical_columns).tolist()
        feature_names = self.numeric_columns + cat_names

        X = pd.DataFrame(np.hstack([numeric, categorical]), columns=feature_names, index=enriched.index)
        return X, FeatureArtifacts(
            encoder=encoder,
            scaler=scaler,
            feature_names=feature_names,
            preprocessor=self,
        )

    def transform_features(
        self,
        df: pd.DataFrame,
        encoder: OneHotEncoder,
        scaler: StandardScaler,
    ) -> pd.DataFrame:
        enriched = self.add_features(df, use_history=True)
        numeric = scaler.transform(enriched[self.numeric_columns])
        categorical = encoder.transform(enriched[self.categorical_columns])
        cat_names = encoder.get_feature_names_out(self.categorical_columns).tolist()
        feature_names = self.numeric_columns + cat_names
        return pd.DataFrame(np.hstack([numeric, categorical]), columns=feature_names, index=enriched.index)

    def _prepare_base(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["_original_order"] = np.arange(len(out))
        self._ensure_base_columns(out)
        out["trans_date_trans_time"] = pd.to_datetime(out["trans_date_trans_time"], errors="coerce")
        out["dob"] = pd.to_datetime(out["dob"], errors="coerce")
        out["cc_num"] = out["cc_num"].astype(str).fillna("unknown")
        out["amt"] = pd.to_numeric(out["amt"], errors="coerce").fillna(0)
        out["merchant_key"] = self._merchant_key(out)
        return out.sort_values(["trans_date_trans_time", "cc_num", "_original_order"], kind="mergesort")

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

        for col in ["city_pop", "lat", "long", "merch_lat", "merch_long"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

        for col in ["category", "gender", "state"]:
            out[col] = out[col].astype(str).fillna("unknown")

    @staticmethod
    def _merchant_key(out: pd.DataFrame) -> pd.Series:
        if "merchant" in out.columns:
            return out["merchant"].astype(str).fillna("unknown")
        lat = pd.to_numeric(out["merch_lat"], errors="coerce").fillna(0).round(3).astype(str)
        lon = pd.to_numeric(out["merch_long"], errors="coerce").fillna(0).round(3).astype(str)
        return lat + "," + lon

    def _fit_customer_state(self, base: pd.DataFrame) -> None:
        valid = base.loc[base["trans_date_trans_time"].notna()].copy()
        if valid.empty:
            self.history_tail_ = valid
            self.card_amount_profile_ = pd.DataFrame(columns=["cc_num", "card_amt_mean", "card_amt_std"])
            self.seen_pairs_ = {}
            return

        max_time = valid["trans_date_trans_time"].max()
        cutoff = max_time - pd.Timedelta(days=self.history_window_days)
        recent = valid.loc[valid["trans_date_trans_time"].ge(cutoff)].copy()
        last_per_card = valid.groupby("cc_num", sort=False).tail(1)
        self.history_tail_ = (
            pd.concat([recent, last_per_card], ignore_index=True)
            .drop_duplicates(subset=["cc_num", "trans_date_trans_time", "amt", "merchant_key"])
            .sort_values(["trans_date_trans_time", "cc_num"], kind="mergesort")
            [self._history_columns()]
        )

        profile = (
            valid.groupby("cc_num")["amt"]
            .agg(card_amt_mean="mean", card_amt_std="std")
            .reset_index()
        )
        profile["card_amt_std"] = profile["card_amt_std"].replace(0, np.nan).fillna(1.0)
        self.card_amount_profile_ = profile

        self.seen_pairs_ = {
            "merchant_key": pd.MultiIndex.from_frame(valid[["cc_num", "merchant_key"]].drop_duplicates()),
            "category": pd.MultiIndex.from_frame(valid[["cc_num", "category"]].drop_duplicates()),
            "state": pd.MultiIndex.from_frame(valid[["cc_num", "state"]].drop_duplicates()),
        }

    @staticmethod
    def _history_columns() -> list[str]:
        return [
            "cc_num",
            "trans_date_trans_time",
            "amt",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "merchant_key",
            "category",
            "state",
            "dob",
            "city_pop",
            "gender",
        ]

    def _add_calendar_geo_features(self, out: pd.DataFrame) -> pd.DataFrame:
        out["hour"] = out["trans_date_trans_time"].dt.hour.fillna(0).astype(int)
        out["dayofweek"] = out["trans_date_trans_time"].dt.dayofweek.fillna(0).astype(int)
        out["month"] = out["trans_date_trans_time"].dt.month.fillna(0).astype(int)
        out["age"] = ((out["trans_date_trans_time"] - out["dob"]).dt.days / 365.25).fillna(0).clip(lower=0)
        out["distance_km"] = haversine_km(out["lat"], out["long"], out["merch_lat"], out["merch_long"]).fillna(0)
        return out

    def _add_behavioral_features(self, out: pd.DataFrame) -> pd.DataFrame:
        grouped = out.groupby("cc_num", sort=False)
        prev_time = grouped["trans_date_trans_time"].shift(1)
        out["time_since_last_trx"] = (
            (out["trans_date_trans_time"] - prev_time)
            .dt.total_seconds()
            .fillna(0)
            .clip(lower=0)
        )

        prev_lat = grouped["merch_lat"].shift(1)
        prev_lon = grouped["merch_long"].shift(1)
        geo_delta = haversine_km(
            prev_lat.fillna(out["merch_lat"]),
            prev_lon.fillna(out["merch_long"]),
            out["merch_lat"],
            out["merch_long"],
        )
        hours_delta = out["time_since_last_trx"] / 3600
        out["geo_velocity_kmh"] = (
            geo_delta / hours_delta.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        valid_time = out["trans_date_trans_time"].notna()
        out["count_trx_1h"] = 0.0
        out["count_trx_24h"] = 0.0
        out["avg_amt_7d"] = out["amt"]

        if valid_time.any():
            valid = out.loc[valid_time].copy()
            out.loc[valid.index, "count_trx_1h"] = self._rolling_count(valid, "1h")
            out.loc[valid.index, "count_trx_24h"] = self._rolling_count(valid, "24h")
            out.loc[valid.index, "avg_amt_7d"] = self._rolling_mean(valid, "7d")

        out["avg_amt_7d"] = out["avg_amt_7d"].replace([np.inf, -np.inf], np.nan).fillna(out["amt"])
        denominator = out["avg_amt_7d"].where(out["avg_amt_7d"].abs() > 1e-9, out["amt"])
        out["amt_ratio_to_avg"] = (
            out["amt"] / denominator.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        return out

    def _add_profile_features(self, out: pd.DataFrame, use_history: bool) -> pd.DataFrame:
        if use_history and self.card_amount_profile_ is not None and not self.card_amount_profile_.empty:
            out = out.merge(self.card_amount_profile_, on="cc_num", how="left")
        else:
            expanding = (
                out.groupby("cc_num", sort=False)["amt"]
                .expanding()
                .agg(["mean", "std"])
                .reset_index(level=0, drop=True)
            )
            out["card_amt_mean"] = expanding["mean"].groupby(out["cc_num"], sort=False).shift(1)
            out["card_amt_std"] = expanding["std"].groupby(out["cc_num"], sort=False).shift(1)

        out["card_amt_mean"] = out.get("card_amt_mean", pd.Series(index=out.index, dtype=float)).fillna(out["avg_amt_7d"])
        out["card_amt_std"] = out.get("card_amt_std", pd.Series(index=out.index, dtype=float)).replace(0, np.nan).fillna(1.0)
        out["amt_zscore_card"] = (
            (out["amt"] - out["card_amt_mean"]) / out["card_amt_std"]
        ).replace([np.inf, -np.inf], np.nan).fillna(0)
        return out

    def _add_seen_flags(self, out: pd.DataFrame, use_history: bool) -> pd.DataFrame:
        mapping = {
            "merchant_key": "is_new_merchant",
            "category": "is_new_category",
            "state": "is_new_state",
        }
        # safe empty MultiIndex with two levels (cc_num, value) to avoid pandas errors
        empty_seen = pd.MultiIndex.from_arrays([[], []], names=["cc_num", "value"])
        for field, output_col in mapping.items():
            pairs = pd.MultiIndex.from_frame(out[["cc_num", field]])
            if use_history and isinstance(self.seen_pairs_, dict):
                seen_ref = self.seen_pairs_.get(field)
                compare_index = empty_seen if (seen_ref is None or len(seen_ref) == 0) else seen_ref
                seen_profile = pairs.isin(compare_index)
            else:
                seen_profile = np.zeros(len(out), dtype=bool)
            seen_current = out.duplicated(["cc_num", field], keep="first").to_numpy()
            out[output_col] = (~(seen_profile | seen_current)).astype(int)
        return out

    def _finalize_columns(self, out: pd.DataFrame) -> pd.DataFrame:
        for col in self.numeric_columns:
            if col not in out:
                out[col] = 0
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

        for col in self.categorical_columns:
            if col not in out:
                out[col] = "unknown"
            out[col] = out[col].astype(str).fillna("unknown")
        return out

    @staticmethod
    def _rolling_count(df: pd.DataFrame, window: str) -> pd.Series:
        def compute(group: pd.DataFrame) -> pd.Series:
            rolled = group.set_index("trans_date_trans_time")["amt"].rolling(window, closed="left").count()
            return pd.Series(rolled.to_numpy(), index=group.index)

        return df.groupby("cc_num", sort=False, group_keys=False).apply(compute).reindex(df.index).fillna(0)

    @staticmethod
    def _rolling_mean(df: pd.DataFrame, window: str) -> pd.Series:
        def compute(group: pd.DataFrame) -> pd.Series:
            rolled = group.set_index("trans_date_trans_time")["amt"].rolling(window, closed="left").mean()
            return pd.Series(rolled.to_numpy(), index=group.index)

        return df.groupby("cc_num", sort=False, group_keys=False).apply(compute).reindex(df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    return Preprocessor().add_features(df, use_history=False)


def fit_transform_features(df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureArtifacts]:
    return Preprocessor().fit_transform_features(df)


def transform_features(df: pd.DataFrame, encoder: OneHotEncoder, scaler: StandardScaler) -> pd.DataFrame:
    return Preprocessor().transform_features(df, encoder, scaler)
