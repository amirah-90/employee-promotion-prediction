import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# make src/ importable so the dashboard can LOAD the frozen saved model
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
from predict_utils import load_model_bundle, predict_promotion

DATA_PATH = BASE_DIR / "data" / "dashboard_data.csv"

st.set_page_config(page_title="Employee Promotion Dashboard", layout="wide")
st.title("Employee Promotion Prediction Dashboard")
st.write("Explore employee data, predict promotion likelihood, and monitor the deployed model.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_bundle():
    return load_model_bundle()

df = load_data()
bundle = load_bundle()   # frozen model loaded from file (not retrained)

# ---------- Monitoring Overview (Q5a) ----------
st.header("Monitoring Overview")
m1, m2, m3 = st.columns(3)
m1.metric("Records Monitored", f"{len(df):,}")
m1.caption("Operational: number of records the dashboard is serving.")
m2.metric("Current Promotion Rate", f"{df['promoted'].mean() * 100:.1f}%")
m2.caption("Business: share of employees promoted (watch for sudden changes).")
m3.metric("Model Recall", f"{bundle['full_model_recall']:.2f}")
m3.caption("Model performance: recall of the evaluated full model (Q2/Q3).")

# ---------- Interactive filters (Q4a-ii) ----------
st.sidebar.header("Filter Options")
dept = st.sidebar.selectbox("Select Department", ["All"] + sorted(df["department"].unique()))
min_perf = st.sidebar.slider("Minimum Performance Score", 40, 100, 40)

view = df.copy()
if dept != "All":
    view = view[view["department"] == dept]
view = view[view["performance_score"] >= min_perf]
st.caption(f"Showing {len(view):,} employees")

# ---------- Visualizations (Q4a-i) ----------
st.subheader("1. Promotion Rate by Department (%)")
rate = view.groupby("department")["promoted"].mean().sort_values(ascending=False) * 100
st.bar_chart(rate)

st.subheader("2. Performance Score Distribution")
fig, ax = plt.subplots()
ax.hist(view["performance_score"], bins=30, color="#d2691e", edgecolor="white")
ax.set_xlabel("Performance Score")
ax.set_ylabel("Number of Employees")
st.pyplot(fig)

# ---------- Live prediction using the frozen model (Q4a) ----------
st.subheader("3. Predict Promotion for an Employee")
cols = st.columns(len(bundle["features"]))
inputs = {}
for c, feat in zip(cols, bundle["features"]):
    inputs[feat] = c.number_input(feat, value=float(df[feat].median()))

if st.button("Predict"):
    pred, prob = predict_promotion(inputs, bundle)
    label = "Likely to be Promoted" if pred == 1 else "Not Likely to be Promoted"
    st.success(f"{label}  (probability = {prob:.2f}, threshold = {bundle['threshold']})")
