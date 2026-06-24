import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Employee Promotion Dashboard", layout="wide")
st.title("Employee Promotion Prediction Dashboard")
st.write("Explore employee data and predict promotion likelihood.")

# absolute path for deployment reliability (same approach as Practical 5.3)
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "dashboard_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(data_path)

df = load_data()

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
ax.hist(view["performance_score"], bins=30, color="#2c6fbb", edgecolor="white")
ax.set_xlabel("Performance Score")
ax.set_ylabel("Number of Employees")
st.pyplot(fig)

st.subheader("3. Promotion Outcome Counts")
counts = view["promoted"].value_counts().rename({0: "Not Promoted", 1: "Promoted"})
st.bar_chart(counts)

# ---------- Predictive output (Q4a-iii) ----------
st.subheader("4. Predict Promotion Likelihood")

@st.cache_resource
def train_model(data):
    feats = ["performance_score", "kpi_achievement_percent", "manager_rating",
             "tasks_completed", "years_at_company"]
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(data[feats], data["promoted"])
    return model, feats

model, feats = train_model(df)

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
