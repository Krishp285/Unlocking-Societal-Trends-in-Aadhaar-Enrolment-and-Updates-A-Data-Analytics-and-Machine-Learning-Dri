"""
aadhaar_ml_pipeline.py

Run: python aadhaar_ml_pipeline.py

This script:
- extracts the three ZIP files (enrolment, demographic, biometric)
- loads CSV(s), normalizes column names
- creates aggregated monthly time series per state
- performs EDA charts
- runs anomaly detection (IsolationForest) on monthly state series
- trains per-state ML regressors with lag features to forecast next months
- saves outputs and models to /mnt/data/aadhaar_outputs/

Adjust paths or parameters at the top as needed.
"""

import os
from pathlib import Path
from turtle import st
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------- PARAMETERS ----------
DATA_DIR = Path("mnt/data")
ZIP_FILES = {
    "enrolment": DATA_DIR / "api_data_aadhar_enrolment.zip",
    "demographic": DATA_DIR / "api_data_aadhar_demographic.zip",
    "biometric": DATA_DIR / "api_data_aadhar_biometric.zip",
}
OUTPUT_DIR = DATA_DIR / "aadhaar_outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FORECAST_MONTHS = 3  # how many months ahead to forecast
RANDOM_STATE = 42
# --------------------------------

def extract_zip(zippath: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zippath, "r") as zf:
        zf.extractall(dest)
    # pick first CSV/XLSX file
    files = list(dest.rglob("*.csv")) + list(dest.rglob("*.xlsx"))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV/XLSX found in {zippath}")
    return files[0]

def load_table(file_path: Path):
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, low_memory=False)
    else:
        df = pd.read_excel(file_path)
    return df

def standardize_columns(df: pd.DataFrame):
    # Lowercase and replace spaces
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def ensure_date(df: pd.DataFrame, candidates=["date","day","enrol_date","update_date"]):
    for c in candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            return df, c
    # fallback: first datetime-like
    for c in df.columns:
        if "date" in c:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            return df, c
    return df, None

def safe_numeric_sum(df, includes_prefix):
    cols = [c for c in df.columns if c.startswith(includes_prefix)]
    if len(cols) == 0:
        # fallback: numeric columns excluding pincode
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        numcols = [c for c in numcols if "pin" not in c and "pincode" not in c]
        cols = numcols
    if len(cols) == 0:
        return None  # nothing found
    return df[cols].sum(axis=1), cols

def prepare_dataset(zip_path: Path, role: str):
    print(f"Processing {role} from {zip_path} ...")
    dest = OUTPUT_DIR / role
    csv_path = extract_zip(zip_path, dest)
    df = load_table(csv_path)
    df = standardize_columns(df)
    df, date_col = ensure_date(df)
    # ensure state and district columns exist
    for col in ["state","district","pincode","pin"]:
        if col not in df.columns:
            # create placeholder if missing
            df[col] = np.nan
    # compute total counts depending on dataset type
    if role == "enrolment":
        s, cols = safe_numeric_sum(df, "age_")
        if s is None:
            # try columns common names
            s = df.select_dtypes(include=[np.number]).sum(axis=1)
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df["total_enrolments"] = s
        key_total = "total_enrolments"
    elif role == "demographic":
        s, cols = safe_numeric_sum(df, "demo_")
        if s is None:
            s = df.select_dtypes(include=[np.number]).sum(axis=1)
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df["total_demo_updates"] = s
        key_total = "total_demo_updates"
    elif role == "biometric":
        s, cols = safe_numeric_sum(df, "bio_")
        if s is None:
            s = df.select_dtypes(include=[np.number]).sum(axis=1)
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df["total_bio_updates"] = s
        key_total = "total_bio_updates"
    else:
        raise ValueError("Unknown role")
    # create month column for aggregation
    if date_col:
        df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    else:
        # if no date col, try to create synthetic monthly bucket from index
        df["month"] = pd.NaT
    return df, date_col, key_total

# ---------- Load datasets ----------
datasets = {}
date_cols = {}
total_cols = {}
for role, zip_path in ZIP_FILES.items():
    if not zip_path.exists():
        print(f"Warning: {zip_path} not found. Skipping {role}.")
        datasets[role] = None
        date_cols[role] = None
        total_cols[role] = None
        continue
    df, date_col, key_total = prepare_dataset(zip_path, role)
    datasets[role] = df
    date_cols[role] = date_col
    total_cols[role] = key_total
    # save sample
    df.head(10).to_csv(OUTPUT_DIR / f"{role}_sample_head.csv", index=False)

# ---------- Aggregate monthly per-state ----------
def aggregate_state_month(df, month_col, total_col, role):
    if df is None:
        return None
    if month_col is None:
        print(f"No date column for {role}. Can't create monthly series.")
        return None
    agg = df.groupby(["state", month_col])[total_col].sum().reset_index()
    agg = agg.rename(columns={month_col: "month", total_col: "count"})
    return agg

agg_enrol = aggregate_state_month(datasets["enrolment"], date_cols["enrolment"], total_cols["enrolment"], "enrolment")
agg_demo = aggregate_state_month(datasets["demographic"], date_cols["demographic"], total_cols["demographic"], "demographic")
agg_bio  = aggregate_state_month(datasets["biometric"], date_cols["biometric"], total_cols["biometric"], "biometric")

# Save aggregated CSVs
if agg_enrol is not None: agg_enrol.to_csv(OUTPUT_DIR / "agg_enrol_state_month.csv", index=False)
if agg_demo  is not None: agg_demo.to_csv(OUTPUT_DIR / "agg_demo_state_month.csv", index=False)
if agg_bio   is not None: agg_bio.to_csv(OUTPUT_DIR / "agg_bio_state_month.csv", index=False)

# ---------- EDA plots: top states and time series ----------
def save_bar_top_states(agg_df, role, top_n=15):
    if agg_df is None: return
    top = agg_df.groupby("state")["count"].sum().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10,5))
    plt.bar(top.index, top.values)
    plt.xticks(rotation=70)
    plt.title(f"Top {top_n} States by {role} (total counts)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"top_states_{role}.png")
    plt.close()

def save_monthly_global_series(agg_df, role):
    if agg_df is None: return
    monthly = agg_df.groupby("month")["count"].sum().reset_index()
    plt.figure(figsize=(10,4))
    plt.plot(monthly["month"], monthly["count"], marker="o")
    plt.title(f"Monthly total {role} (all states combined)")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"monthly_series_{role}.png")
    plt.close()
    monthly.to_csv(OUTPUT_DIR / f"monthly_series_{role}.csv", index=False)
    return monthly

save_bar_top_states(agg_enrol, "enrolment")
save_bar_top_states(agg_demo, "demographic_updates")
save_bar_top_states(agg_bio, "biometric_updates")

monthly_enrol = save_monthly_global_series(agg_enrol, "enrolment")
monthly_demo  = save_monthly_global_series(agg_demo,  "demographic_updates")
monthly_bio   = save_monthly_global_series(agg_bio,   "biometric_updates")

# ---------- Anomaly detection per-state monthly series ----------
def detect_anomalies_state_month(agg_df, role):
    if agg_df is None: return None
    out_rows = []
    states = agg_df["state"].dropna().unique()
    for st in states:
        s_df = agg_df[agg_df["state"] == st].sort_values("month")
        if len(s_df) < 6:
            # skip too-short series
            continue
        values = s_df["count"].fillna(0).values.reshape(-1,1)
        # IsolationForest
        iso = IsolationForest(contamination=0.02, random_state=RANDOM_STATE)
        iso.fit(values)
        preds = iso.predict(values)  # -1 anomaly, 1 normal
        s_df = s_df.copy()
        s_df["anomaly"] = (preds == -1)
        anomalies = s_df[s_df["anomaly"]]
        for _, r in anomalies.iterrows():
            out_rows.append({
                "state": st,
                "month": r["month"],
                "count": r["count"],
                "role": role
            })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_DIR / f"anomalies_{role}.csv", index=False)
    return out_df

anoms_enrol = detect_anomalies_state_month(agg_enrol, "enrolment")
anoms_demo  = detect_anomalies_state_month(agg_demo, "demographic")
anoms_bio   = detect_anomalies_state_month(agg_bio, "biometric")

# ---------- Forecasting: simple lag-feature model (single model trained per-state) ----------
def create_lag_features(dff, lags=[1,2,3,6,12]):
    d = dff.copy().sort_values("month")
    for lag in lags:
        d[f"lag_{lag}"] = d["count"].shift(lag)
    d["month_index"] = np.arange(len(d))
    d = d.dropna().reset_index(drop=True)
    return d

def train_forecast_models(agg_df, role, forecast_months=FORECAST_MONTHS):
    if agg_df is None:
        return {}
    models = {}
    states = agg_df["state"].dropna().unique()
    summary_rows = []
    for st in states:
        st_df = agg_df[agg_df["state"] == st].sort_values("month")
        # Need minimum data for lag features + train/test split
        MIN_ROWS = 15

        if len(st_df) < MIN_ROWS:
            # Not enough data → skip ML for this state
            continue

        st_lagged = create_lag_features(st_df, lags=[1,2,3,6,12])
        feature_cols = [c for c in st_lagged.columns if c.startswith("lag_")] + ["month_index"]
        X = st_lagged[feature_cols]
        y = st_lagged["count"]
        # Train/test split
        # Safe train-test split
        split_idx = int(len(X) * 0.8)

        if split_idx < 1:
            continue  # skip if not enough rows

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        ypred = model.predict(X_test)
        mae = mean_absolute_error(y_test, ypred)
        rmse = mean_squared_error(y_test, ypred, squared=False)
        # ---- FIXED ACTUAL vs PREDICTED PLOT ----
        if len(y_test) > 1:
            plt.figure(figsize=(7,4))

            # reset index to align x-axis
            y_test_plot = y_test.reset_index(drop=True)
            y_pred_plot = pd.Series(ypred)

            x = range(len(y_test_plot))

            plt.plot(x, y_test_plot, marker='o', label="Actual")
            plt.plot(x, y_pred_plot, marker='x', linestyle='--', label="Predicted")

            plt.xlabel("Test Time Index")
            plt.ylabel("Enrolment Count")
            plt.title(f"Actual vs Predicted Enrolment ({st})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"actual_vs_pred_{st}.png")
            plt.close()

    
        # Save model
        models[st] = {"model": model, "feature_cols": feature_cols, "last_df": st_lagged, "mae": mae, "rmse": rmse}
        summary_rows.append({"state": st, "mae": mae, "rmse": rmse, "trained_on": len(X_train), "tested_on": len(X_test)})
        # Forecast N months forward by iterative prediction
        last_row = st_df.sort_values("month").iloc[-1:]
        # prepare a working array of last known values
        recent = st_df["count"].values.tolist()
        preds = []
        temp_series = recent.copy()
        for m in range(forecast_months):
            # construct lag features from temp_series
            lags = [1,2,3,6,12]
            feat = {}
            for lag in lags:
                if len(temp_series) - lag >= 0:
                    feat[f"lag_{lag}"] = temp_series[-lag]
                else:
                    feat[f"lag_{lag}"] = 0.0
            feat["month_index"] = len(temp_series)
            Xpred = pd.DataFrame([feat])[models[st]["feature_cols"]]
            p = model.predict(Xpred)[0]
            preds.append(max(0, float(p)))  # enforce non-negative
            temp_series.append(p)
        models[st]["forecast_next"] = preds
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / f"forecast_summary_{role}.csv", index=False)
    # save models
    joblib.dump(models, OUTPUT_DIR / f"models_{role}.joblib")
    return models

models_enrol = train_forecast_models(agg_enrol, "enrolment", FORECAST_MONTHS)
models_demo  = train_forecast_models(agg_demo,  "demographic_updates", FORECAST_MONTHS)
models_bio   = train_forecast_models(agg_bio,   "biometric_updates", FORECAST_MONTHS)


# Save a small human-readable forecast CSV (state, next n months preds)
def dump_forecasts(models_dict, role, months_ahead=FORECAST_MONTHS):
    rows = []
    for st, info in models_dict.items():
        preds = info.get("forecast_next", [])
        row = {"state": st}
        for i,p in enumerate(preds,1):
            row[f"pred_month_{i}"] = p
        row["mae"] = info.get("mae")
        row["rmse"] = info.get("rmse")
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"forecasts_{role}.csv", index=False)

dump_forecasts(models_enrol, "enrolment")
dump_forecasts(models_demo, "demographic_updates")
dump_forecasts(models_bio, "biometric_updates")

# ---------- Save summary report csvs ----------
if agg_enrol is not None:
    agg_enrol.groupby("state")["count"].sum().sort_values(ascending=False).to_csv(OUTPUT_DIR / "state_totals_enrolment.csv")
if agg_demo is not None:
    agg_demo.groupby("state")["count"].sum().sort_values(ascending=False).to_csv(OUTPUT_DIR / "state_totals_demographic_updates.csv")
if agg_bio is not None:
    agg_bio.groupby("state")["count"].sum().sort_values(ascending=False).to_csv(OUTPUT_DIR / "state_totals_biometric_updates.csv")

print("Completed pipeline. Outputs in:", OUTPUT_DIR)
print("Notable files:")
for f in OUTPUT_DIR.glob("*"):
    print("-", f.name)

def plot_age_state_heatmap(df):
    age_cols = [c for c in df.columns if c.startswith("age_")]
    if not age_cols:
        return
    pivot = df.groupby("state")[age_cols].sum()
    plt.figure(figsize=(8,10))
    plt.imshow(pivot.values, aspect="auto")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(age_cols)), age_cols, rotation=45)
    plt.title("Age-wise enrolment distribution across states")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_state_heatmap.png")
    plt.close()

plot_age_state_heatmap(datasets["enrolment"])

def plot_stacked_age_state(df, state_name):
    age_cols = [c for c in df.columns if c.startswith("age_")]
    sdf = df[df["state"] == state_name]
    if sdf.empty:
        return
    grouped = sdf.groupby("month")[age_cols].sum()
    grouped.plot.area(figsize=(10,5))
    plt.title(f"Stacked age-wise enrolment trend: {state_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"stacked_age_{state_name}.png")
    plt.close()

plot_stacked_age_state(datasets["enrolment"], "Uttar Pradesh")
