from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
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


st.set_page_config(page_title="Victim Profiling", layout="wide")
st.title("Victim Profiling")

try:
    df = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

lensa = st.radio("Lensa analisis", charts.LENS_OPTIONS, horizontal=True)
st.info(charts.lens_description(lensa))

fraud_df = df.loc[charts.lens_mask(df, lensa)].copy()
top_job = fraud_df["job"].value_counts().idxmax() if not fraud_df.empty else "Unknown"
avg_age = float(fraud_df["age"].mean()) if not fraud_df.empty else 0

k1, k2 = st.columns(2)
k1.metric("Profesi Paling Diserang", top_job)
k2.metric("Rata-rata Umur Korban", f"{avg_age:,.1f} tahun")

left, right = st.columns([5, 5])
with left:
    st.plotly_chart(charts.plot_demographic_dist(df, lensa), use_container_width=True)
with right:
    scatter_df = df.copy()
    scatter_df["Status"] = charts.lens_mask(scatter_df, lensa).map({True: "Fraud", False: "Aman"})
    if len(scatter_df) > 20000:
        scatter_df = scatter_df.sample(20000, random_state=42)
    fig = px.scatter(
        scatter_df,
        x="city_pop",
        y="amt",
        color="Status",
        hover_data=["job", "category", "age"],
        title=f"Sebaran Populasi Kota vs Nominal - {lensa}",
        template=charts.PLOTLY_TEMPLATE,
        color_discrete_map={"Fraud": "#ef4444", "Aman": "#38bdf8"},
    )
    st.plotly_chart(fig, use_container_width=True)

st.plotly_chart(charts.plot_job_treemap(df, lensa), use_container_width=True)
