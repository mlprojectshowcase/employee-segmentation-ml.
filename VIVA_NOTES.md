# Viva quick preparation

**Project title:** Employee Segmentation using Machine Learning

**Problem statement:** Organizations may have employee data but lack a simple way to explore groups with similar work-related characteristics. This project demonstrates how clustering can reveal patterns in employee records.

**Objective:** Apply K-Means to group employees using experience, salary, performance score, job satisfaction, and training hours, then visualize and explore the groups.

**Why machine learning?** The groups are not given beforehand. The algorithm discovers patterns from feature similarities, so this is unsupervised learning.

**Why K-Means?** It is relatively simple, efficient, widely taught, and produces understandable cluster assignments and centroids.

**Why StandardScaler?** K-Means uses distance. Scaling stops salary's larger numeric range from overwhelming scores and training hours.

**What is K?** The number of clusters requested from K-Means. In this app it can be changed in the sidebar.

**What is a centroid?** The mean position of the observations assigned to a cluster in feature space.

**What does the silhouette score mean?** It ranges from -1 to +1. A higher score generally indicates better-separated clusters; near zero suggests overlap, and negative values may indicate poor assignments.

**Is this supervised or unsupervised?** Unsupervised, because there is no target label supplied during training.

**Can it predict a new employee's cluster?** Yes. The app scales the new employee's feature values with the fitted scaler and uses the fitted K-Means model to assign the nearest cluster.

**Limitations:** Synthetic sample data, chosen features and K influence results; K-Means can be sensitive to outliers and assumes roughly compact clusters. It should not be used for consequential HR decisions.

**Demo sequence:** Open the app → show dataset → change K → explain distribution chart → explain profile table → enter a new employee → download results.
