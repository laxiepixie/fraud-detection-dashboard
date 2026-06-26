from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference.storage import TEMP_PROCESSED_DIR
from src.visualization import charts


ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "dashboard_predictions.csv",
    TEMP_PROCESSED_DIR / "dashboard_predictions.csv",
    ROOT / "data" / "raw" / "fraudTest.csv",
    ROOT / "data" / "raw" / "fraudTrain.csv",
]


@st.cache_data(show_spinner="Memuat data dashboard...")
def load_data() -> pd.DataFrame:
    source = next((path for path in DATA_CANDIDATES if path.exists()), None)
    if source is None:
        raise FileNotFoundError("Tidak ada data. Letakkan fraudTest.csv atau dashboard_predictions.csv.")

    required = [
        "amt",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "category",
        "job",
        "dob",
        "trans_date_trans_time",
        "cc_num",
        "state",
        "city_pop",
        "is_fraud",
        "pred_cpp",
        "pred_lgbm",
        "pred_if",
        "pred_lof",
        "pred_svm"
    ]
    available = pd.read_csv(source, nrows=0).columns.tolist()
    usecols = [col for col in required if col in available]
    return charts.prepare_dashboard_frame(pd.read_csv(source, usecols=usecols))


st.set_page_config(page_title="Spatiotemporal Radar", layout="wide")
st.title("Spatiotemporal Radar")

try:
    df = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

lensa = st.radio("Lensa analisis", charts.LENS_OPTIONS, horizontal=True)
st.info(charts.lens_description(lensa))

fraud_df = df.loc[charts.lens_mask(df, lensa)].copy()
state_risk = "Unknown"
if not fraud_df.empty and "state" in fraud_df.columns:
    state_risk = fraud_df["state"].value_counts().idxmax()
avg_distance = float(fraud_df["distance_km"].mean()) if not fraud_df.empty else 0

k1, k2 = st.columns(2)
k1.metric("State Paling Rawan", state_risk)
k2.metric("Rata-rata Jarak Anomali", f"{avg_distance:,.2f} km")

left, right = st.columns([7, 3])
with left:
    st.plotly_chart(charts.plot_spatial_map(df, lensa), use_container_width=True)
with right:
    st.subheader("Metrik Jarak Kasar")
    st.metric("Median jarak", f"{fraud_df['distance_km'].median() if not fraud_df.empty else 0:,.2f} km")
    st.metric("Jarak maksimum", f"{fraud_df['distance_km'].max() if not fraud_df.empty else 0:,.2f} km")
    st.metric("Jumlah titik fraud", f"{len(fraud_df):,}")

st.plotly_chart(charts.plot_temporal_heatmap(df, lensa), use_container_width=True)
