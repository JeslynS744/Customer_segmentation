# 🛍️ Customer Segmentation using K-Means Clustering

A data science project that groups retail mall customers into distinct personas using **K-Means Clustering** based on **Annual Income** and **Spending Score**.

---

## 📌 Project Overview
Customer segmentation enables businesses to understand their target demographics, optimize promotional campaigns, and allocate resources efficiently. This project implements an end-to-end unsupervised learning pipeline on the classic **Mall Customers Dataset** from Kaggle.

### Key Highlights:
- **Data Ingestion & EDA**: Statistical analysis and distribution inspection across demographic variables.
- **Optimal Cluster Selection**: Systematic evaluation using the **Elbow Method** (Within-Cluster Sum of Squares / Inertia).
- **Model Training**: K-Means clustering with `k-means++` initialization ($K = 5$).
- **Business Personas**: Translating geometric cluster centroids into concrete retail marketing strategies.

---

## 📂 Project Structure

```text
customer_segmentation/
├── Mall_Customers.csv                # Original dataset
├── Mall_Customers_Segmented.csv      # Enriched dataset with cluster assignments
├── customer_segments_summary.csv     # Demographic & behavioral metrics per cluster
├── customer_segmentation_kmeans.py   # Complete 4-step pipeline script
├── initial_scatter_plot.png          # Raw income vs. spending distribution
├── elbow_method_curve.png            # WCSS elbow curve confirming K=5
├── customer_segmentation_clusters.png# Final clustered visualization with centroids
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure Python 3.9+ is installed. Clone the repository and install required packages:

```bash
git clone <your-repo-url>
cd customer_segmentation
pip install -r requirements.txt
```

### 2. Running the Pipeline
Execute the full pipeline with a single command:

```bash
python customer_segmentation_kmeans.py
```

---

## 📊 Methodology & Findings

### Phase 1: Exploratory Data Analysis (EDA)
- **Dataset Size**: 200 records, 5 columns (`CustomerID`, `Genre`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`).
- **Data Quality**: 0 missing values, clean data distributions.
- **Initial Observation**: A scatter plot of *Annual Income* vs. *Spending Score* exhibits natural 2D clustering patterns.

### Phase 2: The Elbow Method ($K = 5$)
Evaluating WCSS (Inertia) for $K \in [1, 10]$ reveals a clear elbow inflection point at **$K = 5$**:
- $K=1 \rightarrow \text{WCSS} \approx 269,981$
- $K=5 \rightarrow \text{WCSS} \approx 44,448$
- Beyond $K=5$, reduction in error plateaus, indicating diminishing returns.

### Phase 3 & 4: Customer Personas & Business Profiling

| Cluster | Persona | Size (% Base) | Avg Age | Avg Income | Avg Spend | Strategic Focus |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **0** | **Middle Ground** | 40.5% | 42.7 yrs | \$55.3k | 49.5 | Retention via general loyalty rewards and seasonal sales |
| **1** | **VIP / Target Stars** | 19.5% | 32.7 yrs | \$86.5k | 82.1 | Premium offerings, concierge service, luxury brand pop-ups |
| **2** | **Trendsetters / Impulsive** | 11.0% | 25.3 yrs | \$25.7k | 79.4 | Social media marketing, trendy items, flexible BNPL options |
| **3** | **Conservative Spenders** | 17.5% | 41.1 yrs | \$88.2k | 17.1 | Value propositions, warranties, long-term durability |
| **4** | **Budget Shoppers** | 11.5% | 45.2 yrs | \$26.3k | 20.9 | Clearance events, discount coupons, economical bundles |

---

## 🛠️ Built With
- **Python 3**
- **Pandas** & **NumPy** - Data manipulation and mathematical operations
- **Scikit-Learn** - K-Means clustering algorithm & metrics
- **Matplotlib** & **Seaborn** - Data visualization and charts
