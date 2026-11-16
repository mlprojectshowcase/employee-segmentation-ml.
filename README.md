# Employee Segmentation using Machine Learning

A viva-ready academic project that segments employees with **K-Means clustering** and presents the results in an interactive Streamlit dashboard.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

## Free online hosting (Streamlit Community Cloud)

1. Create a free account at https://github.com/ and sign in.
2. Create a **public** GitHub repository, for example `employee-segmentation-ml`.
3. Upload `app.py`, `employee_data.csv`, and `requirements.txt` to the repository root.
4. Sign in at https://share.streamlit.io/ using GitHub.
5. Select **Create app**, choose your repository, branch (`main`), and main file path (`app.py`).
6. Deploy. Streamlit will install packages from `requirements.txt` and provide a public app URL.
7. Open that URL on the presentation computer or phone for your viva.

Free hosting availability and limits can change. Do not upload confidential employee data to a public repository or public demo.

## Dataset columns

- `Employee_ID`: synthetic identifier
- `Department`: department category (shown for exploration)
- `Experience_Years`: years of experience
- `Monthly_Salary`: monthly salary
- `Performance_Score`: illustrative score from 1 to 10
- `Job_Satisfaction`: illustrative score from 1 to 10
- `Training_Hours`: training hours

The supplied dataset is **synthetic** and created only for demonstration. It does not represent real employees.

## Algorithm

The app standardizes five numeric features with `StandardScaler`, then fits K-Means with a user-selected number of clusters. The app reports a silhouette score and cluster-level feature averages. K-Means is an unsupervised algorithm: it discovers groups without pre-existing target labels.

## Viva points

- **Segmentation:** dividing observations into groups with similar characteristics.
- **Clustering:** an unsupervised machine-learning task.
- **K-Means:** partitions observations into K clusters by minimizing within-cluster squared distances.
- **Centroid:** the mean feature vector of a cluster.
- **Why scaling?** Salary values can be much larger numerically than scores; scaling prevents a large-scale feature from dominating distance calculations.
- **Silhouette score:** measures how close a point is to its own cluster compared with other clusters; higher is generally better.
- **Why choose K?** Compare silhouette scores for different K values and consider whether the groups are interpretable and useful.
- **Limitations:** results depend on selected features, scaling, K, and data quality; K-Means favors roughly spherical clusters and can be affected by outliers.

## Responsible use

This is an educational demonstration, not a validated HR decision system. Cluster names are descriptive labels only. Do not use the output to make employment decisions. Use anonymized, authorized data only.
