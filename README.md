# Unlocking Societal Trends in Aadhaar Enrolment and Updates

A Data Analytics and Machine Learning–Driven Decision Support System for understanding Aadhaar enrolment and update patterns, detecting anomalies, and forecasting service demand.

---

## 📌 Project Overview

Aadhaar enrolment and update activities generate large-scale administrative data that reflects societal behavior, mobility, and system load.  
This project analyses **Aadhaar Enrolment**, **Demographic Updates**, and **Biometric Updates** datasets to extract meaningful insights, detect anomalies, and generate short-term predictive indicators that support informed decision-making.

Rather than treating Aadhaar data as static counts, this system interprets activity as **administrative and behavioural signals** useful for governance and operational planning.

---

## 🎯 Objectives

- Perform univariate, bivariate, and multivariate analysis on Aadhaar datasets
- Identify temporal and regional trends in enrolment and updates
- Detect anomalous activity indicating operational stress or special drives
- Forecast short-term Aadhaar service demand using machine learning
- Translate analytical findings into actionable administrative insights

---

## 📂 Datasets Used

1. **Aadhaar Enrolment Dataset**
   - State, district, PIN code
   - Age-wise enrolment categories (0–5, 5–17, 18+)
   - Temporal enrolment patterns

2. **Aadhaar Demographic Update Dataset**
   - Address, mobile number, name, DOB, gender updates
   - Reflects migration, corrections, and life events

3. **Aadhaar Biometric Update Dataset**
   - Fingerprint, iris, and face updates
   - Indicates lifecycle-based biometric revalidation

> ⚠️ All datasets are **aggregated and anonymized**.  
> No personal or sensitive data is used.

---

## 🏗️ System Architecture

The system follows a **layered and modular architecture**:

1. Data Ingestion Layer  
2. Data Preprocessing & Feature Engineering  
3. Descriptive Analytics (EDA)  
4. Anomaly Detection Engine  
5. Forecasting Engine  
6. Insight & Decision Support Layer  
7. Visualization & Reporting Layer  

This design ensures **reproducibility, explainability, and governance readiness**.

---

## 🧪 Methodology

### 🔹 Data Processing
- ZIP extraction and schema validation
- Column normalization and date standardization
- State–month aggregation
- Lag-based feature engineering for forecasting

### 🔹 Exploratory Data Analysis
- Univariate analysis: distributions and dominance
- Bivariate analysis: state vs time relationships
- Multivariate perspective across datasets

### 🔹 Anomaly Detection
- Isolation Forest (unsupervised)
- Detects unusual spikes or drops in activity
- Acts as an operational stress indicator

### 🔹 Forecasting
- Random Forest Regressor (state-wise models)
- Lag features: 1, 2, 3, 6, 12 months
- Fallback logic for sparse data regions
- Focus on short-term planning support

---

## 🤖 Machine Learning Models Used

| Task | Model |
|----|----|
| Anomaly Detection | Isolation Forest |
| Forecasting | Random Forest Regressor |

**Why Random Forest?**
- Robust to noise
- Works well with limited data
- Interpretable for administrative use

---

## 📊 Outputs Generated

All outputs are saved to:


### Key Files:
- `top_states_*.png` → State-wise distributions
- `monthly_series_*.png` → Time-series trends
- `anomalies_*.csv` → Detected anomalies
- `forecasts_*.csv` → Future demand predictions
- `forecast_summary_*.csv` → Model performance metrics
- `models_*.joblib` → Saved ML models
- `state_totals_*.csv` → Aggregated statistics

---

## 📈 Visualization & Reporting

- PNG charts suitable for reports and presentations
- CSV outputs for tabular analysis
- Word/PDF report and SIH-style PPT supported

Visualizations are designed for **non-technical stakeholders**.

---

## 💡 Key Insights

- Enrolment spikes align with policy initiatives and welfare schemes
- High demographic update ratios indicate population mobility
- Biometric update peaks correspond to Aadhaar lifecycle stages
- Significant regional variation exists in administrative workload
- Anomaly detection helps identify operational issues early

---

## 🏛️ Impact & Applicability

The system supports:
- Proactive enrolment center planning
- Temporary staff and infrastructure allocation
- Migration-aware administrative decisions
- Early warning of system stress or outages
- Improved citizen service delivery

This framework is directly applicable to **UIDAI-style governance workflows**.

---

## ⚖️ Design Decisions & Trade-offs

- Prioritized explainability over complex deep learning models
- Avoided individual-level inference to preserve privacy
- Used fallback statistical methods for sparse regions
- Focused on decision support rather than prediction alone

---

## 🚀 How to Run the Project

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn joblib

Folder Structure

uidai/
├── aadhaar_ml_pipeline.py
└── mnt/
    └── data/
        ├── api_data_aadhar_enrolment.zip
        ├── api_data_aadhar_demographic.zip
        ├── api_data_aadhar_biometric.zip


