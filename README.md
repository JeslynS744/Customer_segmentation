# Customer Segmentation

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0+-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

**Customer Segmentation** is an unsupervised machine learning pipeline that clusters retail mall customers into distinct personas using K-Means clustering based on annual income and spending score.

---

## 🌟 Key Features

* **Automated Data Ingestion & EDA:** Loads mall customer datasets and performs statistical audits, missing value checks, and distribution profiling across demographic variables.
* **Optimal Cluster Evaluation (The Elbow Method):** Evaluates Within-Cluster Sum of Squares (WCSS / Inertia) across candidate cluster counts ($K \in [1, 10]$) to mathematically determine the optimal elbow point ($K = 5$).
* **K-Means Clustering Pipeline:** Deploys K-Means with `k-means++` initialization for fast, robust convergence and consistent customer partitioning.
* **Actionable Retail Personas:** Maps cluster centroids to strategic business segments (Target Stars, Middle Ground, Trendsetters, Conservative Spenders, Budget Shoppers).
* **Automated Visualization Suite:** Generates and exports high-resolution visual assets for initial data scatter, WCSS elbow curve, and final clustered scatter plots with centroids.
* **Segmented Dataset & Metrics Export:** Exports labeled customer records (`Mall_Customers_Segmented.csv`) and demographic metric summaries (`customer_segments_summary.csv`) for downstream marketing workflows.

---

## 🚀 Quick Start

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/JeslynS744/Customer_segmentation.git
cd Customer_segmentation
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1 | Mac/Linux: source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python customer_segmentation_kmeans.py
```

---

## 📁 Project Structure

```text
Customer_segmentation/
├── .gitignore
├── Mall_Customers.csv
├── Mall_Customers_Segmented.csv
├── README.md
├── customer_segmentation_clusters.png
├── customer_segmentation_kmeans.py
├── customer_segments_summary.csv
├── elbow_method_curve.png
├── initial_scatter_plot.png
└── requirements.txt
```
