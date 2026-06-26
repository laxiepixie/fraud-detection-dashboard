from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOTLY_TEMPLATE = "plotly_dark"

LENS_OPTIONS = ["Aktual", "C++ Mahalanobis", "LightGBM", "SVM (Linear)", "Isolation Forest", "LOF"]

LENS_COLUMNS = {
    "Aktual": "is_fraud",
    "C++ Mahalanobis": "pred_cpp",
    "LightGBM": "pred_lgbm",
    "SVM (Linear)": "pred_svm",
    "Isolation Forest": "pred_if",
    "LOF": "pred_lof",
}

LENS_INFO = {
    "Aktual": "Lensa aktual memakai label asli dataset sebagai ground truth fraud.",
    "C++ Mahalanobis": "Mahalanobis menandai transaksi yang jaraknya ekstrem dari pola normal multivariat.",
    "LightGBM": "LightGBM memakai supervised learning untuk mengenali pola fraud dari data berlabel.",
    "SVM (Linear)": "SVM mencari margin garis lurus (hyperplane) paling optimal untuk memisahkan transaksi aman dan fraud.",
    "Isolation Forest": "Isolation Forest mencari transaksi yang mudah dipisahkan dari populasi normal.",
    "LOF": "LOF membandingkan kepadatan lokal transaksi untuk menemukan titik yang tidak wajar.",
}

FEATURE_IMPORTANCE = pd.DataFrame(
    {
        "feature": [
            "amt",
            "distance_km",
            "hour",
            "category",
            "job",
            "age",
            "city_pop",
            "merchant_location",
        ],
        "importance": [0.31, 0.18, 0.14, 0.12, 0.09, 0.07, 0.05, 0.04],
    }
)


def lens_column(lensa: str) -> str:
    return LENS_COLUMNS.get(lensa, "is_fraud")


def lens_description(lensa: str) -> str:
    return LENS_INFO.get(lensa, LENS_INFO["Aktual"])


def lens_mask(df: pd.DataFrame, lensa: str) -> pd.Series:
    col = lens_column(lensa)
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).eq(1)
    return pd.Series(False, index=df.index)


def prepare_dashboard_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["amt", "lat", "long", "merch_lat", "merch_long", "city_pop"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["category", "job", "state", "cc_num"]:
        if col not in out.columns:
            out[col] = "Unknown"
        out[col] = out[col].astype(str).fillna("Unknown")

    if "trans_date_trans_time" not in out.columns:
        out["trans_date_trans_time"] = pd.NaT
    out["trans_date_trans_time"] = pd.to_datetime(out["trans_date_trans_time"], errors="coerce")
    out["hour"] = out["trans_date_trans_time"].dt.hour.fillna(0).astype(int)
    out["day_name"] = out["trans_date_trans_time"].dt.day_name().fillna("Unknown")
    out["period"] = out["trans_date_trans_time"].dt.to_period("M").astype(str)
    out.loc[out["period"].eq("NaT"), "period"] = "Unknown"

    if "dob" not in out.columns:
        out["dob"] = pd.NaT
    out["dob"] = pd.to_datetime(out["dob"], errors="coerce")
    age = (out["trans_date_trans_time"] - out["dob"]).dt.days / 365.25
    out["age"] = age.fillna(0).clip(lower=0, upper=110)
    out["age_band"] = pd.cut(
        out["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        include_lowest=True,
    ).astype(str)

    out["distance_km"] = _haversine_km(out["lat"], out["long"], out["merch_lat"], out["merch_long"])

    for col in LENS_COLUMNS.values():
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    return out


def _haversine_km(lat1, lon1, lat2, lon2) -> pd.Series:
    radius = 6371.0
    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return pd.Series(2 * radius * np.arcsin(np.sqrt(a)), index=getattr(lat1, "index", None)).fillna(0)


def _fraud_subset(df: pd.DataFrame, lensa: str) -> pd.DataFrame:
    return df.loc[lens_mask(df, lensa)].copy()


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=PLOTLY_TEMPLATE, title=title)
    fig.add_annotation(
        text="Data tidak tersedia untuk lensa ini.",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
    )
    return fig


def plot_roi_area(df: pd.DataFrame, lensa: str) -> go.Figure:
    work = prepare_dashboard_frame(df)
    actual = work["is_fraud"].eq(1) if "is_fraud" in work.columns else pd.Series(False, index=work.index)
    predicted = lens_mask(work, lensa)
    work["Dana Terselamatkan"] = np.where(actual & predicted, work["amt"].fillna(0), 0)
    work["Kerugian Finansial"] = np.where(actual & ~predicted, work["amt"].fillna(0), 0)

    grouped = (
        work.groupby("period")[["Dana Terselamatkan", "Kerugian Finansial"]]
        .sum()
        .sort_index()
        .reset_index()
    )
    if grouped.empty:
        return _empty_figure(f"ROI Finansial - {lensa}")

    melted = grouped.melt("period", var_name="Komponen", value_name="Nominal")
    fig = px.area(
        melted,
        x="period",
        y="Nominal",
        color="Komponen",
        title=f"ROI Finansial - {lensa}",
        template=PLOTLY_TEMPLATE,
        color_discrete_map={"Dana Terselamatkan": "#10b981", "Kerugian Finansial": "#ef4444"},
    )
    fig.update_layout(xaxis_title="Periode", yaxis_title="Nominal Transaksi")
    return fig


def plot_feature_importance(df: pd.DataFrame, lensa: str) -> go.Figure:
    importance = FEATURE_IMPORTANCE.copy().sort_values("importance", ascending=True)
    fig = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Mockup Explainable AI - {lensa}",
        template=PLOTLY_TEMPLATE,
        labels={"importance": "Kontribusi relatif", "feature": "Fitur"},
        color="importance",
        color_continuous_scale="Tealrose",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_spatial_map(df: pd.DataFrame, lensa: str) -> go.Figure:
    work = prepare_dashboard_frame(df)
    fraud = _fraud_subset(work, lensa).dropna(subset=["merch_lat", "merch_long"])
    if fraud.empty:
        return _empty_figure(f"Radar Lokasi Merchant Fraud - {lensa}")

    if len(fraud) > 15000:
        fraud = fraud.sample(15000, random_state=42)

    fig = px.scatter_mapbox(
        fraud,
        lat="merch_lat",
        lon="merch_long",
        color="amt",
        size="amt",
        size_max=14,
        hover_data=["amt", "category", "job", "state", "distance_km"],
        zoom=3,
        center={"lat": 39.5, "lon": -98.35},
        title=f"Scatter Mapbox Kasus Fraud - {lensa}",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="Reds",
    )
    fig.update_layout(mapbox_style="carto-darkmatter", margin={"r": 0, "t": 50, "l": 0, "b": 0})
    return fig


def plot_temporal_heatmap(df: pd.DataFrame, lensa: str) -> go.Figure:
    work = prepare_dashboard_frame(df)
    fraud = _fraud_subset(work, lensa)
    if fraud.empty:
        return _empty_figure(f"Heatmap Hari vs Jam - {lensa}")

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    matrix = (
        fraud.groupby(["day_name", "hour"])
        .size()
        .reset_index(name="fraud_count")
        .pivot(index="day_name", columns="hour", values="fraud_count")
        .reindex(day_order)
        .reindex(columns=list(range(24)), fill_value=0)
        .fillna(0)
    )

    fig = px.imshow(
        matrix,
        aspect="auto",
        title=f"Heatmap Pola Waktu Fraud - {lensa}",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="YlOrRd",
        labels={"x": "Jam", "y": "Hari", "color": "Jumlah"},
    )
    fig.update_xaxes(side="top")
    return fig


def plot_category_bar(df: pd.DataFrame, lensa: str) -> go.Figure:
    work = prepare_dashboard_frame(df)
    work["_flag"] = lens_mask(work, lensa).astype(int)
    grouped = (
        work.groupby("category")
        .agg(total=("amt", "size"), flagged=("_flag", "sum"))
        .assign(fraud_rate=lambda x: np.where(x["total"] > 0, x["flagged"] / x["total"], 0))
        .sort_values("fraud_rate", ascending=False)
        .head(15)
        .reset_index()
        .sort_values("fraud_rate")
    )
    if grouped.empty:
        return _empty_figure(f"Persentase Fraud per Kategori - {lensa}")

    fig = px.bar(
        grouped,
        x="fraud_rate",
        y="category",
        orientation="h",
        title=f"Persentase Fraud per Kategori - {lensa}",
        template=PLOTLY_TEMPLATE,
        labels={"fraud_rate": "Persentase fraud", "category": "Kategori"},
        color="fraud_rate",
        color_continuous_scale="Reds",
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(tickformat=".1%")
    return fig


def plot_velocity_scatter(df: pd.DataFrame, lensa: str) -> go.Figure:
    work = prepare_dashboard_frame(df).dropna(subset=["trans_date_trans_time"])
    if work.empty:
        return _empty_figure(f"Velocity Gesek Kartu - {lensa}")

    work["_flag"] = lens_mask(work, lensa).astype(int)
    candidate_cards = work.loc[work["_flag"].eq(1), "cc_num"].dropna().unique()
    if len(candidate_cards) == 0:
        candidate_cards = work["cc_num"].dropna().unique()
    if len(candidate_cards) == 0:
        return _empty_figure(f"Velocity Gesek Kartu - {lensa}")

    seed = int(hashlib.sha256(lensa.encode("utf-8")).hexdigest()[:8], 16)
    selected_card = candidate_cards[seed % len(candidate_cards)]
    card_df = work.loc[work["cc_num"].eq(selected_card)].sort_values("trans_date_trans_time")

    fig = px.scatter(
        card_df,
        x="trans_date_trans_time",
        y="amt",
        color=card_df["_flag"].map({1: "Fraud", 0: "Aman"}),
        size="amt",
        hover_data=["category", "job", "state", "distance_km"],
        title=f"Velocity Transaksi Kartu {selected_card} - {lensa}",
        template=PLOTLY_TEMPLATE,
        color_discrete_map={"Fraud": "#ef4444", "Aman": "#38bdf8"},
    )
    fig.update_layout(xaxis_title="Waktu Transaksi", yaxis_title="Nominal")
    return fig


def plot_demographic_dist(df: pd.DataFrame, lensa: str) -> go.Figure:
    work = prepare_dashboard_frame(df)
    work["_status"] = np.where(lens_mask(work, lensa), "Fraud", "Aman")
    grouped = work.groupby(["age_band", "_status"]).size().reset_index(name="count")
    if grouped.empty:
        return _empty_figure(f"Distribusi Umur Korban - {lensa}")

    fig = px.bar(
        grouped,
        x="age_band",
        y="count",
        color="_status",
        barmode="group",
        title=f"Distribusi Umur Korban - {lensa}",
        template=PLOTLY_TEMPLATE,
        labels={"age_band": "Rentang umur", "count": "Jumlah transaksi", "_status": "Status"},
        color_discrete_map={"Fraud": "#ef4444", "Aman": "#38bdf8"},
    )
    return fig


def plot_job_treemap(df: pd.DataFrame, lensa: str) -> go.Figure:
    work = prepare_dashboard_frame(df)
    work["_flag"] = lens_mask(work, lensa).astype(int)
    grouped = (
        work.groupby("job")
        .agg(total=("amt", "size"), fraud_count=("_flag", "sum"), total_amt=("amt", "sum"))
        .assign(fraud_rate=lambda x: np.where(x["total"] > 0, x["fraud_count"] / x["total"], 0))
        .query("fraud_count > 0")
        .sort_values("fraud_count", ascending=False)
        .head(50)
        .reset_index()
    )
    if grouped.empty:
        return _empty_figure(f"Treemap Kerentanan Profesi - {lensa}")

    fig = px.treemap(
        grouped,
        path=["job"],
        values="fraud_count",
        color="fraud_rate",
        hover_data=["total", "total_amt"],
        title=f"Treemap Profesi Paling Rentan - {lensa}",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="Reds",
    )
    return fig
