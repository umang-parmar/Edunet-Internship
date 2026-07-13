# Smart Energy Usage Analysis using K-Means Clustering
# Dataset : smart_energy_usage_dataset.csv

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Dataset

energy_df = pd.read_csv("smart_energy_usage_dataset.csv")

print(energy_df.head())

# Basic Information

print("\nDataset Information")
energy_df.info()

print("\nShape")
print(energy_df.shape)

print("\nColumns")
print(energy_df.columns.tolist())

print("\nStatistical Summary")
print(energy_df.describe())

print("\nMissing Values")
print(energy_df.isnull().sum())

# Human Observations
# Handle Missing Values

energy_df.fillna(
    energy_df.mean(numeric_only=True),
    inplace=True
)

energy_df["location"] = energy_df["location"].fillna(
    energy_df["location"].mode()[0]
)

# Feature Engineering

energy_df["Weather_Index"] = (
    energy_df["temperature_c"]
    * energy_df["humidity_pct"]
) / 100

# Weather Index gives a simple indication
# of environmental conditions.

energy_df["Usage_Per_Person"] = (
    energy_df["energy_usage_kwh"]
    / energy_df["occupancy"]
)

# This feature compares energy usage
# with the number of people present.

# Select Features for Clustering

X = energy_df[
    [
        "energy_usage_kwh",
        "temperature_c",
        "humidity_pct",
        "solar_output_kw",
        "Usage_Per_Person",
        "Weather_Index"
    ]
]

# Standardization

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

# Scaling ensures that every feature
# contributes equally during clustering.

# Train KMeans Model

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    init="k-means++"
)

energy_df["Cluster"] = kmeans.fit_predict(X_scaled)

# Cluster Quality

score = silhouette_score(
    X_scaled,
    energy_df["Cluster"]
)

print("\nSilhouette Score :", score)

# A higher silhouette score indicates
# better separated clusters.

# Scatter Plot

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=energy_df,
    x="energy_usage_kwh",
    y="temperature_c",
    hue="Cluster",
    palette="Set2",
    s=70
)

plt.title("Energy Usage Clusters")
plt.xlabel("Energy Usage (kWh)")
plt.ylabel("Temperature (°C)")
plt.show()

# Box Plot

sns.boxplot(
    x="Cluster",
    y="energy_usage_kwh",
    data=energy_df
)

plt.title("Energy Usage Distribution by Cluster")
plt.show()

# Histogram

sns.histplot(
    energy_df["energy_usage_kwh"],
    bins=20,
    kde=True
)

plt.title("Distribution of Energy Usage")
plt.show()

# Correlation Heatmap

plt.figure(figsize=(10,7))
numeric_df = energy_df.select_dtypes(include=np.number)
sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=True
)

plt.title("Correlation Matrix")
plt.show()

# Cluster Summary

print("\nAverage Values for Each Cluster")
print(
    energy_df.groupby("Cluster")[
        [
            "energy_usage_kwh",
            "temperature_c",
            "humidity_pct",
            "solar_output_kw",
            "occupancy"
        ]
    ].mean()
)

# # OUTPUT:
# PS C:\Users\UMANG\Desktop\day-1_task-main\Internship\Day-07_(13-07-2026)> py d7.py
#           timestamp    location  energy_usage_kwh  ...  solar_output_kw  occupancy  peak_hour
# 0  01-01-2024 00:00  Industrial            620.28  ...           205.32         58          0
# 1  01-01-2024 01:00  Industrial            412.19  ...            21.49        224          0
# 2  01-01-2024 02:00       Urban            217.18  ...           141.36        495          0
# 3  01-01-2024 03:00       Rural            439.46  ...           146.39         35          0
# 4  01-01-2024 04:00       Rural            106.55  ...            36.06         70          0

# [5 rows x 8 columns]

# Dataset Information
# <class 'pandas.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 8 columns):
#  #   Column            Non-Null Count  Dtype  
# ---  ------            --------------  -----  
#  0   timestamp         1000 non-null   str    
#  1   location          1000 non-null   str    
#  2   energy_usage_kwh  1000 non-null   float64
#  3   temperature_c     1000 non-null   float64
#  4   humidity_pct      1000 non-null   float64
#  5   solar_output_kw   1000 non-null   float64
#  6   occupancy         1000 non-null   int64  
#  7   peak_hour         1000 non-null   int64  
# dtypes: float64(4), int64(2), str(2)
# memory usage: 62.6 KB

# Shape
# (1000, 8)

# Columns
# ['timestamp', 'location', 'energy_usage_kwh', 'temperature_c', 'humidity_pct', 'solar_output_kw', 'occupancy', 'peak_hour']

# Statistical Summary
#        energy_usage_kwh  temperature_c  humidity_pct  solar_output_kw    occupancy    peak_hour
# count       1000.000000    1000.000000   1000.000000      1000.000000  1000.000000  1000.000000
# mean         360.574640      26.740680     56.043560       124.378990   254.218000     0.164000
# std          163.355635       8.785039     21.827753        73.570038   141.763081     0.370461
# min           80.130000      12.050000     20.100000         0.230000    10.000000     0.000000
# 25%          217.840000      18.957500     37.355000        60.020000   133.750000     0.000000
# 50%          355.825000      26.695000     54.565000       125.200000   248.000000     0.000000
# 75%          500.257500      34.395000     75.065000       189.267500   376.250000     0.000000
# max          649.130000      41.980000     94.860000       249.970000   500.000000     1.000000

# Missing Values
# timestamp           0
# location            0
# energy_usage_kwh    0
# temperature_c       0
# humidity_pct        0
# solar_output_kw     0
# occupancy           0
# peak_hour           0
# dtype: int64
# [[ 1.59061125 -0.33034929 -1.4861715   1.10074038  1.77965706 -1.17770626]
#  [ 0.31612737  1.42920247 -0.91413622 -1.39921735 -0.20165411 -0.10447732]
#  [-0.87824574  0.20378001 -1.44904421  0.2309297  -0.51523817 -1.00969871]
#  ...
#  [-0.33082218  0.35524951 -0.77433593 -1.62265371 -0.28519522 -0.4173729 ]
#  [-0.39433507 -1.64118689 -0.37418625  0.89661501 -0.24520677 -1.14349969]
#  [ 0.65402576  0.45546994 -1.00443345 -1.62224573 -0.31882454 -0.56778212]]

# Silhouette Score : 0.17462940168913124
