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


st.set_page_config(page_title="Behavior and Velocity", layout="wide")
st.title("Behavior and Velocity")

try:
    df = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

lensa = st.radio("Lensa analisis", charts.LENS_OPTIONS, horizontal=True)
st.info(charts.lens_description(lensa))

fraud_df = df.loc[charts.lens_mask(df, lensa)].copy()
category_risk = fraud_df["category"].value_counts().idxmax() if not fraud_df.empty else "Unknown"
avg_amt = float(fraud_df["amt"].mean()) if not fraud_df.empty else 0

k1, k2 = st.columns(2)
k1.metric("Kategori Paling Rawan", category_risk)
k2.metric("Rata-rata Nominal Anomali", f"${avg_amt:,.2f}")

left, right = st.columns([5, 5])
with left:
    st.plotly_chart(charts.plot_category_bar(df, lensa), use_container_width=True)
with right:
    box_df = df.copy()
    box_df["Status"] = charts.lens_mask(box_df, lensa).map({True: "Fraud", False: "Aman"})
    fig = px.box(
        box_df,
        x="Status",
        y="amt",
        color="Status",
        title=f"Sebaran Nominal Transaksi - {lensa}",
        template=charts.PLOTLY_TEMPLATE,
        color_discrete_map={"Fraud": "#ef4444", "Aman": "#38bdf8"},
    )
    st.plotly_chart(fig, use_container_width=True)

st.plotly_chart(charts.plot_velocity_scatter(df, lensa), use_container_width=True)
