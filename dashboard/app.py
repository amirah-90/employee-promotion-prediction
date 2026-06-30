import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

st.set_page_config(page_title="Employee Promotion Dashboard", layout="wide")
st.title("Employee Promotion Prediction Dashboard")
st.write("Explore employee data, predict promotion likelihood, and monitor the model.")

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "dashboard_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(data_path)

df = load_data()
feats = ["performance_score", "kpi_achievement_percent", "manager_rating",
         "tasks_completed", "years_at_company"]

@st.cache_resource
def train_model(data):
    Xtr, Xte, ytr, yte = train_test_split(data[feats], data["promoted"], test_size=0.2,
                                          random_state=42, stratify=data["promoted"])
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
    rec = recall_score(yte, model.predict(Xte))
    return model, rec

model, model_recall = train_model(df)

# ---------- Monitoring Overview (Q5a) ----------
st.header("Monitoring Overview")
m1, m2, m3 = st.columns(3)
m1.metric("Records Monitored", f"{len(df):,}")
m1.caption("Operational: number of records the dashboard is serving.")
m2.metric("Current Promotion Rate", f"{df['promoted'].mean()*100:.1f}%")
m2.caption("Business: share of employees promoted (watch for sudden changes).")
m3.metric("Model Recall", f"{model_recall:.2f}")
m3.caption("Model performance: share of real promotions the model catches.")
st.divider()

# ---------- Filters ----------
st.sidebar.header("Filter Options")
dept = st.sidebar.selectbox("Select Department", ["All"] + sorted(df["department"].unique()))
min_perf = st.sidebar.slider("Minimum Performance Score", 40, 100, 40)
view = df.copy()
if dept != "All":
    view = view[view["department"] == dept]
view = view[view["performance_score"] >= min_perf]
st.caption(f"Showing {len(view):,} employees")

# ---------- Visualizations ----------
st.subheader("1. Promotion Rate by Department (%)")
rate = view.groupby("department")["promoted"].mean().sort_values(ascending=False) * 100
st.bar_chart(rate)

st.subheader("2. Performance Score Distribution")
fig, ax = plt.subplots()
ax.hist(view["performance_score"], bins=30, color="#2c6fbb", edgecolor="white")
ax.set_xlabel("Performance Score")
ax.set_ylabel("Number of Employees")
st.pyplot(fig)

st.subheader("3. Promotion Outcome Counts")
counts = view["promoted"].value_counts().rename({0: "Not Promoted", 1: "Promoted"})
st.bar_chart(counts)

# ---------- Prediction ----------
st.subheader("4. Predict Promotion Likelihood")
c1, c2, c3 = st.columns(3)
perf = c1.slider("Performance Score", 40, 100, 70)
kpi = c2.slider("KPI Achievement %", 30, 100, 70)
mgr = c3.slider("Manager Rating", 1.0, 5.0, 3.0, step=0.1)
tasks = c1.slider("Tasks Completed", 0, 130, 35)
years = c2.slider("Years at Company", 0, 30, 5)
if st.button("Predict"):
    row = pd.DataFrame([[perf, kpi, mgr, tasks, years]], columns=feats)
    prob = model.predict_proba(row)[0][1]
    st.metric("Promotion Probability", f"{prob*100:.1f}%")
    if prob >= 0.5:
        st.success("Prediction: Likely to be promoted")
    else:
        st.info("Prediction: Unlikely to be promoted")
