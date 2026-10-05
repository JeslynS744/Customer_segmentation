"""
Customer Segmentation using K-Means Clustering
Dataset: Mall Customers (Kaggle)

Complete 4-Step Pipeline:
  Step 1: Setup & Data Exploration
  Step 2: Finding Optimal K (Elbow Method)
  Step 3: Model Training (K-Means with K=5)
  Step 4: Visualization & Business Profiling
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# Global Aesthetics & Configuration
# ---------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'Segoe UI',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

DATASET_PATH = "Mall_Customers.csv"

# =========================================================
# STEP 1: Setup & Data Exploration
# =========================================================
print("=" * 70)
print("STEP 1: SETUP & DATA EXPLORATION")
print("=" * 70)

# 1.1 Load the dataset
df = pd.read_csv(DATASET_PATH)

print("\n--- 1. Dataset Preview (First 5 Rows) ---")
print(df.head())

print("\n--- 2. Dataset Information & Schema ---")
print(df.info())

print("\n--- 3. Missing Value Audit ---")
missing_counts = df.isnull().sum()
print(missing_counts)

print("\n--- 4. Numerical Summary Statistics ---")
print(df.describe())

# 1.2 Initial Scatter Plot (Raw unclustered data)
plt.figure(figsize=(9, 6), dpi=120)
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="#1f77b4",
    s=75,
    alpha=0.85,
    edgecolor="w",
    linewidth=0.8
)
plt.title("Initial Exploration: Annual Income vs. Spending Score", weight="bold", pad=15)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.tight_layout()
initial_plot_file = "initial_scatter_plot.png"
plt.savefig(initial_plot_file)
plt.close()
print(f"\n[Saved] Initial scatter plot: '{initial_plot_file}'")


# =========================================================
# STEP 2: Finding Optimal K (The Elbow Method)
# =========================================================
print("\n" + "=" * 70)
print("STEP 2: FINDING OPTIMAL K (ELBOW METHOD)")
print("=" * 70)

# Extract features: Annual Income (col 3) & Spending Score (col 4)
X = df.iloc[:, [3, 4]].values

wcss = []
k_range = range(1, 11)

print("\nCalculating Within-Cluster Sum of Squares (WCSS)...")
for k in k_range:
    kmeans_test = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans_test.fit(X)
    wcss.append(kmeans_test.inertia_)
    print(f"  K = {k:2d} -> WCSS (Inertia): {kmeans_test.inertia_:10.2f}")

# Plot the Elbow Curve
plt.figure(figsize=(9, 5.5), dpi=120)
plt.plot(k_range, wcss, marker='o', markersize=8, color="#2b5c8f", linewidth=2.5, label="WCSS")

optimal_k = 5
plt.axvline(x=optimal_k, color="#d9534f", linestyle="--", linewidth=1.8, label=f"Optimal Elbow (K = {optimal_k})")
plt.scatter(optimal_k, wcss[optimal_k - 1], color="#d9534f", s=130, zorder=5)

plt.title("Elbow Method for Selecting Optimal Clusters", weight="bold", pad=15)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.xticks(k_range)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True, facecolor="white", loc="upper right")
plt.tight_layout()
elbow_plot_file = "elbow_method_curve.png"
plt.savefig(elbow_plot_file)
plt.close()
print(f"\n[Saved] Elbow curve plot: '{elbow_plot_file}'")


# =========================================================
# STEP 3: Model Training (K-Means with Optimal K)
# =========================================================
print("\n" + "=" * 70)
print(f"STEP 3: MODEL TRAINING (K-Means with K = {optimal_k})")
print("=" * 70)

# Initialize KMeans with optimal clusters
kmeans = KMeans(
    n_clusters=optimal_k,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42
)

# Fit model and predict cluster assignments
cluster_labels = kmeans.fit_predict(X)

# Assign cluster indices to dataframe
df["Cluster"] = cluster_labels
centroids = kmeans.cluster_centers_

print(f"\nModel successfully trained on {len(X)} customer samples.")
print("\nCentroid Coordinates [Annual Income (k$), Spending Score (1-100)]:")
for i, center in enumerate(centroids):
    print(f"  Cluster {i} Centroid: Income = ${center[0]:5.2f}k, Spending Score = {center[1]:5.2f}")


# =========================================================
# STEP 4: Visualization & Business Profiling
# =========================================================
print("\n" + "=" * 70)
print("STEP 4: VISUALIZATION & BUSINESS PROFILING")
print("=" * 70)

# Cluster Persona Mapping based on Income & Spending coordinates:
# Let's dynamically map based on centroid position:
persona_map = {}
for i, center in enumerate(centroids):
    inc, spd = center[0], center[1]
    if inc > 65 and spd > 60:
        persona = "VIP / Target Stars (High Income, High Spend)"
    elif inc > 65 and spd < 40:
        persona = "Careful / Conservative (High Income, Low Spend)"
    elif inc < 45 and spd > 60:
        persona = "Impulsive / Trendsetters (Low Income, High Spend)"
    elif inc < 45 and spd < 40:
        persona = "Budget Shoppers (Low Income, Low Spend)"
    else:
        persona = "Middle Ground (Moderate Income, Moderate Spend)"
    persona_map[i] = persona

df["Segment_Name"] = df["Cluster"].map(persona_map)

# 4.1 Cluster Scatter Plot Visualization
palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
cluster_order = sorted(df["Cluster"].unique())

plt.figure(figsize=(11, 7), dpi=130)

for idx, cluster_id in enumerate(cluster_order):
    cluster_points = df[df["Cluster"] == cluster_id]
    plt.scatter(
        cluster_points["Annual Income (k$)"],
        cluster_points["Spending Score (1-100)"],
        s=85,
        color=palette[idx % len(palette)],
        alpha=0.85,
        edgecolor="w",
        linewidth=0.7,
        label=f"Cluster {cluster_id}: {persona_map[cluster_id]}"
    )

# Plot Centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=260,
    color="#111111",
    marker="X",
    edgecolor="#ffffff",
    linewidth=1.8,
    zorder=10,
    label="Cluster Centroids"
)

plt.title("Customer Segmentation: 5 Distinct Customer Groups", weight="bold", pad=15)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., frameon=True)
plt.tight_layout()
clusters_plot_file = "customer_segmentation_clusters.png"
plt.savefig(clusters_plot_file)
plt.close()
print(f"\n[Saved] Final clustered plot: '{clusters_plot_file}'")

# 4.2 Business Profile Summary Table
profile_summary = df.groupby(["Cluster", "Segment_Name"]).agg(
    Customer_Count=("CustomerID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income_k=("Annual Income (k$)", "mean"),
    Avg_Spending_Score=("Spending Score (1-100)", "mean"),
    Female_Ratio=("Genre", lambda g: round((g == "Female").mean() * 100, 1))
).reset_index()

print("\n--- Business Profile Summary ---")
print(profile_summary.to_string(index=False))

# Export segmented dataset and summary
segmented_dataset_file = "Mall_Customers_Segmented.csv"
df.to_csv(segmented_dataset_file, index=False)
print(f"\n[Saved] Enriched segmented dataset: '{segmented_dataset_file}'")

summary_file = "customer_segments_summary.csv"
profile_summary.to_csv(summary_file, index=False)
print(f"[Saved] Customer segmentation summary: '{summary_file}'")
print("\n" + "=" * 70)
print("PROJECT PIPELINE EXECUTION COMPLETE!")
print("=" * 70)
