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


st.set_page_config(page_title="Business and Logic", layout="wide")
st.title("Business and Logic")

try:
    df = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

lensa = st.radio("Lensa analisis", charts.LENS_OPTIONS, horizontal=True)
st.info(charts.lens_description(lensa))

actual = df["is_fraud"].eq(1)
predicted = charts.lens_mask(df, lensa)
tp = actual & predicted
fp = ~actual & predicted
fn = actual & ~predicted

saved_funds = float(df.loc[tp, "amt"].sum())
loss_funds = float(df.loc[fn, "amt"].sum())
precision = tp.sum() / max((tp | fp).sum(), 1)

k1, k2, k3 = st.columns(3)
k1.metric("Total Dana Terselamatkan", f"${saved_funds:,.2f}")
k2.metric("Total Kerugian", f"${loss_funds:,.2f}")
k3.metric("Rasio Presisi", f"{precision:.2%}")

left, right = st.columns([6, 4])
with left:
    st.plotly_chart(charts.plot_roi_area(df, lensa), use_container_width=True)
with right:
    st.plotly_chart(charts.plot_feature_importance(df, lensa), use_container_width=True)
