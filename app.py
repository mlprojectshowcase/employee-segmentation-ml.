import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Employee Segmentation", page_icon="👥", layout="wide")
st.title("👥 Employee Segmentation using Machine Learning")
st.caption("Unsupervised learning project | K-Means Clustering")
st.markdown("""
This dashboard groups employees with similar work-related characteristics.
It uses **K-Means clustering**, an unsupervised machine-learning algorithm.
The included CSV is synthetic demonstration data; replace it with approved,
anonymized data for a real organizational analysis.
""")

FEATURES = ["Experience_Years", "Monthly_Salary", "Performance_Score",
            "Job_Satisfaction", "Training_Hours"]

@st.cache_data
def load_sample():
    return pd.read_csv("employee_data.csv")

uploaded = st.sidebar.file_uploader("Upload employee CSV (optional)", type=["csv"])
try:
    data = pd.read_csv(uploaded) if uploaded else load_sample()
except Exception as e:
    st.error(f"Could not read the CSV: {e}")
    st.stop()

st.sidebar.header("Clustering settings")
k = st.sidebar.slider("Number of clusters (K)", 2, 8, 3)
st.sidebar.info("Choose K based on business understanding and evaluation metrics.")

missing = [c for c in FEATURES if c not in data.columns]
if missing:
    st.error("Your CSV is missing required numeric columns: " + ", ".join(missing))
    st.stop()

work = data.copy()
for col in FEATURES:
    work[col] = pd.to_numeric(work[col], errors="coerce")
work[FEATURES] = work[FEATURES].replace([np.inf, -np.inf], np.nan)
work = work.dropna(subset=FEATURES).reset_index(drop=True)

if len(work) <= k:
    st.error(f"At least {k+1} complete employee records are required.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(work[FEATURES])
model = KMeans(n_clusters=k, random_state=42, n_init=10)
work["Cluster"] = model.fit_predict(X_scaled)
score = silhouette_score(X_scaled, work["Cluster"]) if len(set(work["Cluster"])) > 1 else 0

# Human-readable cluster labels based on average feature profile
means = work.groupby("Cluster")[FEATURES].mean()
order = means["Performance_Score"].sort_values().index.tolist()
labels = {}
names = ["Developing profile", "Experienced contributors", "High-performing specialists",
         "Emerging talent", "Core workforce", "Growth-oriented employees",
         "Specialized contributors", "Balanced contributors"]
for rank, cluster_id in enumerate(order):
    labels[cluster_id] = names[min(rank, len(names)-1)]
work["Segment"] = work["Cluster"].map(labels)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees analyzed", f"{len(work):,}")
c2.metric("Segments created", k)
c3.metric("Silhouette score", f"{score:.3f}")
c4.metric("Departments", work["Department"].nunique() if "Department" in work else "N/A")

st.subheader("Segment distribution")
counts = work["Segment"].value_counts().rename_axis("Segment").reset_index(name="Employees")
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots()
    sns.barplot(data=counts, x="Employees", y="Segment", ax=ax)
    ax.set_title("Employees in each segment")
    st.pyplot(fig)
    plt.close(fig)
with right:
    fig, ax = plt.subplots()
    sns.scatterplot(data=work, x="Experience_Years", y="Monthly_Salary",
                    hue="Segment", palette="tab10", s=75, ax=ax)
    ax.set_title("Experience vs monthly salary")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Cluster profiles")
profile = work.groupby("Segment")[FEATURES].mean().round(2)
st.dataframe(profile, use_container_width=True)

st.subheader("Explore employee records")
display_cols = [c for c in ["Employee_ID", "Department"] if c in work.columns] + FEATURES + ["Segment"]
st.dataframe(work[display_cols], use_container_width=True)

csv = work.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download segmented employee data", csv,
                   "employee_segments.csv", "text/csv")

st.subheader("Assign a new employee to a segment")
with st.form("new_employee"):
    a, b, c = st.columns(3)
    exp = a.number_input("Experience (years)", min_value=0.0, max_value=50.0, value=3.0)
    salary = b.number_input("Monthly salary", min_value=0.0, max_value=1000000.0, value=35000.0, step=1000.0)
    performance = c.number_input("Performance score (1–10)", min_value=1.0, max_value=10.0, value=7.0)
    d, e = st.columns(2)
    satisfaction = d.number_input("Job satisfaction (1–10)", min_value=1.0, max_value=10.0, value=7.0)
    training = e.number_input("Training hours", min_value=0.0, max_value=1000.0, value=20.0)
    submitted = st.form_submit_button("Find employee segment")
if submitted:
    new_x = pd.DataFrame([[exp, salary, performance, satisfaction, training]], columns=FEATURES)
    cluster_id = int(model.predict(scaler.transform(new_x))[0])
    st.success(f"Predicted segment: **{labels[cluster_id]}**")

with st.expander("How the machine learning works"):
    st.markdown("""
    1. **Data collection:** load employee records from a CSV file.
    2. **Preprocessing:** convert required columns to numeric values and remove incomplete rows.
    3. **Feature scaling:** StandardScaler puts numeric features on comparable scales.
    4. **Clustering:** K-Means assigns each employee to one of K groups using distance to cluster centroids.
    5. **Evaluation:** silhouette score gives an indication of how well-separated the clusters are.
    6. **Interpretation:** compare average feature values to understand each segment.

    **Important:** Clusters are patterns in this dataset, not objective judgments of employee worth.
    Do not use this demo to make hiring, firing, promotion, or compensation decisions.
    """)
st.caption("Academic demonstration • K-Means is unsupervised learning • Demo data is synthetic")
