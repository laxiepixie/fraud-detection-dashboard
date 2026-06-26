from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Optional
import os
from src.training.train_models import run_training # Asumsikan skrip ini ada

def initialize_data():
    file_path = 'data/processed/dashboard_predictions.csv'
    
    # Cek apakah file ada
    if not os.path.exists(file_path):
        st.info("Data tidak ditemukan. Menjalankan skrip training untuk generate data...")
        try:
            # Pastikan folder ada
            os.makedirs('data/processed/', exist_ok=True)
            # Jalankan skrip training-mu
            run_training() 
            st.success("Data berhasil di-generate!")
        except Exception as e:
            st.error(f"Gagal generate data: {e}")

# Panggil fungsi ini sebelum kode dashboard lainnya
initialize_data()

ROOT = Path(__file__).parent

# Ensure src can be imported when running the app from the project root or other locations
src_path = ROOT / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    # Import the stateful Preprocessor that creates behavioral features
    from inference.preprocessing import Preprocessor  # type: ignore
except Exception as e:
    st.warning(f"Failed to import Preprocessor: {e}. Using identity Preprocessor as fallback.")

    class Preprocessor:
        def add_features(self, df):
            return df
        def fit_transform_features(self, df):
            return df, None

# ROOT already defined above

st.set_page_config(
    page_title="Anomali Fraud Detection",
    page_icon="💳",
    layout="wide",
)

st.title("Anomali Fraud Detection")
st.caption("Dashboard analisis fraud dan deteksi anomali transaksi.")

model_dir = ROOT / "models"
raw_dir = ROOT / "data" / "raw"

left, mid, right = st.columns(3)
left.metric("Model folder", "Ada" if model_dir.exists() else "Belum ada")
mid.metric("fraudTrain.csv", "Ada" if (raw_dir / "fraudTrain.csv").exists() else "Belum ada")
right.metric("fraudTest.csv", "Ada" if (raw_dir / "fraudTest.csv").exists() else "Belum ada")

st.subheader("Alur kerja")
st.write(
    "Mulai dari menaruh dataset di `data/raw`, lalu jalankan training, kemudian buka halaman analisis "
    "dan halaman deteksi fraud dari sidebar."
)

st.code(
    "pip install -r requirements.txt\n"
    "python src/training/train_models.py\n"
    "streamlit run app.py",
    language="bash",
)

st.info("Kalau model belum dilatih, jalankan script training dulu sebelum memakai halaman Fraud Detection.")

@st.cache_data
def load_and_preprocess_data(nrows: int = 50000) -> Optional[pd.DataFrame]:
    """Load raw data, run through Preprocessor.add_features, and return processed dataframe.

    Caches the result to avoid repeated preprocessing during interaction.
    """
    csv_path = ROOT / "data" / "raw" / "fraudTrain.csv"
    if not csv_path.exists():
        st.error(f"File not found: {csv_path}")
        return None
    try:
        df_raw = pd.read_csv(csv_path, nrows=nrows)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    try:
        pre = Preprocessor()
        # Prefer add_features to generate stateful behavioral features
        if hasattr(pre, "add_features"):
            df_processed = pre.add_features(df_raw)
        elif hasattr(pre, "fit_transform_features"):
            df_processed, _ = pre.fit_transform_features(df_raw)
        else:
            # fallback to returning raw data if Preprocessor lacks expected API
            df_processed = df_raw

        # Convert numpy arrays to DataFrame if necessary
        if isinstance(df_processed, np.ndarray):
            df_processed = pd.DataFrame(df_processed)
        df_processed = pd.DataFrame(df_processed)

    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return None

    return df_processed


with st.expander('Analisis Distribusi & Skewness Data (Post-Preprocessing)', expanded=True):
    df = load_and_preprocess_data()
    if df is None:
        st.warning("Data terproses tidak tersedia. Pastikan file ada di data/raw dan Preprocessor dapat dijalankan.")
    else:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not num_cols:
            st.info("Tidak ada kolom numerik di data terproses untuk dianalisis.")
        else:
            selected = st.multiselect("Pilih fitur numerik untuk analisis", options=num_cols, default=num_cols[:5])
            if not selected:
                st.info("Pilih paling sedikit satu fitur untuk menampilkan analisis.")
            for feat in selected:
                skew_val = float(df[feat].skew(skipna=True))
                col_left, col_right = st.columns([1, 3])
                col_left.metric(label=f"Skewness: {feat}", value=f"{skew_val:.3f}")

                if abs(skew_val) > 1:
                    col_left.warning(
                        f"Skewness {skew_val:.3f} (|skew|>1): distribusi sangat miring — dapat merugikan model seperti Mahalanobis."
                    )
                elif abs(skew_val) > 0.5:
                    col_left.info(f"Skewness {skew_val:.3f} (moderate skew).")

                fig = px.histogram(df, x=feat, marginal='box', nbins=50, title=f"Distribusi fitur: {feat}")
                col_right.plotly_chart(fig, use_container_width=True)
