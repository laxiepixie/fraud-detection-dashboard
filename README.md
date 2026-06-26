# Fraud Detection & Algorithm Comparison Dashboard

A machine learning pipeline and Streamlit dashboard designed to detect financial transaction anomalies. This project evaluates and compares five different algorithms (Supervised and Unsupervised) to measure their effectiveness on highly skewed behavioral data.

## 🧠 Model Architecture & Analysis

The system trains and evaluates the following 5 models using ~555,000 test records:

1. **LightGBM (Supervised Tree):** Achieves the highest precision and recall. It is natively robust against unscaled features and extreme data skewness.
2. **Local Outlier Factor / LOF (Unsupervised Spatial):** Utilizes `StandardScaler` to equalize feature weights (e.g., transaction amount vs. frequency), effectively identifying isolated anomalies within normal transaction clusters.
3. **SVM with RBFSampler (Supervised Kernel Approximation):** Projects data into 500 dimensions using Random Fourier Features before applying a Linear SVM (`SGDClassifier`). This approximates non-linear decision boundaries while bypassing the $O(n^3)$ memory complexity of a standard RBF SVC.
4. **Isolation Forest (Unsupervised Tree):** Isolates anomalies using random orthogonal splits. Generates higher false positives as it tends to flag legitimate high-value transactions.
5. **Mahalanobis Distance (Unsupervised Parametric):** Assumes a multivariate normal distribution. Performs poorly on this specific dataset due to the extreme skewness and non-parametric nature of human transaction behavior.

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                  # Place fraudTrain.csv & fraudTest.csv here
│   ├── processed/            # Processed features (Parquet) & dashboard_predictions.csv
│   └── analytics/            # Aggregated summary data for the dashboard
├── models/                   # Serialized model artifacts (.pkl) and threshold.json
├── src/
│   ├── inference/            # Preprocessing, data scaling, and prediction pipelines
│   ├── training/             # Model training scripts & threshold optimization
│   └── visualization/        # Plotly chart rendering modules
├── app.py                    # Streamlit Dashboard entry point
└── requirements.txt
```

## ⚙️ Data Requirements
The system requires a dataset formatted similarly to standard Credit Card Fraud datasets. The raw CSV files must include the following columns:
trans_date_trans_time, cc_num, category, amt, gender, job, state, lat, long, city_pop, dob, merch_lat, merch_long, and is_fraud.


## 🚀 How to Run
1. Environment Setup
Ensure Python is installed, then install the required dependencies:

```text
pip install -r requirements.txt
```
2. Model Training & Prediction
Execute the backend pipeline. This script trains all 5 models, optimizes classification thresholds, and generates the dashboard_predictions.csv file required by the UI.

```text
python src/training/train_models.py
```
3. Launch the Dashboard
Start the Streamlit interface:

```text
streamlit run app.py
```
<<<<<<< HEAD
You can copy and paste this directly into your repository. Are you ready to init
=======
You can copy and paste this directly into your repository. Are you ready to init
>>>>>>> 5aed35c (Deploy: Arsitektur Deteksi Fraud dan Dashboard UI)
