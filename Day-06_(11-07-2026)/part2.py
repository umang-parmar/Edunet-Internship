# Air Quality Classification using Logistic Regression
# Dataset : air_quality_synthetic.csv

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load Dataset

air_df = pd.read_csv("air_quality_synthetic.csv")

print(air_df.head())

# Basic Information

print("\nInformation")
air_df.info()

print("\nShape")
print(air_df.shape)

print("\nColumns")
print(air_df.columns.tolist())

print("\nMissing Values")
print(air_df.isnull().sum())

print("\nStatistics")
print(air_df.describe())

# Observations

# 1. Dataset contains air pollution information.
# 2. Every row represents one city in a specific month.
# 3. Dataset has numerical and categorical features.
# 4. AQI depends on pollutant concentration.
# 5. Temperature and humidity may affect pollution.
# 6. PM2.5 and PM10 are major pollution indicators.
# 7. Dataset is useful for classification.
# 8. City is a categorical feature.
# 9. AQI will be converted into pollution classes.
# 10. No unnecessary columns exist.

# Missing Value Handling

numeric_cols = air_df.select_dtypes(include=np.number).columns

air_df[numeric_cols] = air_df[numeric_cols].fillna(
    air_df[numeric_cols].mean()
)

air_df["City"] = air_df["City"].fillna(
    air_df["City"].mode()[0]
)

# Feature Engineering

air_df["Pollution_Index"] = (

    air_df["PM2_5"]

    + air_df["PM10"]

    + air_df["NO2"]

    + air_df["SO2"]

)

# Pollution Index combines major pollutants
# into one useful feature.

# Create Target Variable

air_df["Air_Quality"] = pd.cut(

    air_df["AQI"],

    bins=[0,100,200,500],

    labels=["Good","Moderate","Poor"]

)

print(air_df["Air_Quality"].head())

# Label Encoding

label = LabelEncoder()

air_df["Air_Quality"] = label.fit_transform(
    air_df["Air_Quality"]
)

print(air_df["Air_Quality"].head())

# One Hot Encoding

air_df = pd.get_dummies(

    air_df,

    columns=["City"],

    drop_first=True

)

print("\nColumns After Encoding")
print(air_df.columns.tolist())

# Feature Scaling

scaler = StandardScaler()

num_cols = air_df.select_dtypes(include=np.number).columns

num_cols = num_cols.drop("Air_Quality")

air_df[num_cols] = scaler.fit_transform(

    air_df[num_cols]

)

# Scaling makes all numerical values
# come to a similar range.

# Visualization

sns.countplot(

    x="Air_Quality",

    data=air_df

)

plt.show()

# ==========================================================

sns.histplot(

    air_df["AQI"],

    kde=True

)

plt.show()

# ==========================================================

sns.boxplot(

    x=air_df["Pollution_Index"]

)

plt.show()


plt.figure(figsize=(12,8))

sns.heatmap(

    air_df.corr(),

    cmap="coolwarm",

    annot=False

)

plt.show()

# Prepare Features

X = air_df.drop(

    "Air_Quality",

    axis=1

)

y = air_df["Air_Quality"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

# Train Model

model = LogisticRegression(

    max_iter=500

)

model.fit(

    X_train,

    y_train

)

# Prediction

prediction = model.predict(

    X_test

)

print("\nPredictions")

print(prediction)

# Evaluation

print("\nAccuracy")

print(

    accuracy_score(

        y_test,

        prediction

    )

)

print("\nConfusion Matrix")

print(

    confusion_matrix(

        y_test,

        prediction

    )

)

print("\nClassification Report")

print(

    classification_report(

        y_test,

        prediction

    )

)

# 1. Loaded the dataset using pandas.
# 2. Checked dataset structure and missing values.
# 3. Filled missing numerical values using the mean.
# 4. Filled missing city names using the most common city.
# 5. Created Pollution_Index by combining important pollutants.
# 6. Converted AQI into three pollution categories.
# 7. Encoded categorical labels into numerical values.
# 8. Converted City into dummy variables for machine learning.
# 9. Standardized numerical features before training.
# 10. Used Logistic Regression because this is a classification problem.
# 11. Accuracy shows the percentage of correct predictions.
# 12. Confusion Matrix explains where the model makes mistakes.
# 13. Classification Report provides precision, recall and F1-score.
# 14. This workflow is suitable for practicing preprocessing and classification.

# OUTPUT:-
#  py part2.py
#      City  Year  Month  PM2_5   PM10    NO2   SO2    CO     O3  Temperature_C  Humidity_pct  AQI
# 0  MetroB  2024      2   49.6  128.9   56.7  39.8  3.96   26.0           10.9          82.7  251
# 1  MetroE  2020     12   82.9  183.2   31.3  56.8  4.52   15.2           10.8          60.6  225
# 2  MetroB  2023     12   10.1   63.2   55.4  30.8  1.24   49.2           17.0          54.5  178
# 3  MetroA  2023      9  166.4   34.0   77.4  44.0  1.55  136.3           38.7          93.0  286
# 4  MetroD  2024     11   38.2   78.2  116.9  31.0  4.71   76.9           37.3          56.0  410

# Information
# <class 'pandas.DataFrame'>
# RangeIndex: 250 entries, 0 to 249
# Data columns (total 12 columns):
#  #   Column         Non-Null Count  Dtype  
# ---  ------         --------------  -----  
#  0   City           250 non-null    str    
#  1   Year           250 non-null    int64  
#  2   Month          250 non-null    int64  
#  3   PM2_5          250 non-null    float64
#  4   PM10           250 non-null    float64
#  5   NO2            250 non-null    float64
#  6   SO2            250 non-null    float64
#  7   CO             250 non-null    float64
#  8   O3             250 non-null    float64
#  9   Temperature_C  250 non-null    float64
#  10  Humidity_pct   250 non-null    float64
#  11  AQI            250 non-null    int64  
# dtypes: float64(8), int64(3), str(1)
# memory usage: 23.6 KB

# Shape
# (250, 12)

# Columns
# ['City', 'Year', 'Month', 'PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature_C', 'Humidity_pct', 'AQI']

# Missing Values
# City             0
# Year             0
# Month            0
# PM2_5            0
# PM10             0
# NO2              0
# SO2              0
# CO               0
# O3               0
# Temperature_C    0
# Humidity_pct     0
# AQI              0
# dtype: int64

# Statistics
#               Year       Month       PM2_5        PM10  ...          O3  Temperature_C  Humidity_pct         AQI
# count   250.000000  250.000000  250.000000  250.000000  ...  250.000000     250.000000    250.000000  250.000000
# mean   2022.644000    6.792000   88.215200  131.479600  ...   96.902000      25.845600     58.333600  245.840000
# std       1.644409    3.576577   50.582682   70.866914  ...   48.691321       9.194992     21.398909  123.175416
# min    2020.000000    1.000000    6.300000   10.400000  ...   11.100000      10.000000     20.600000   30.000000
# 25%    2021.000000    3.000000   45.025000   72.675000  ...   55.650000      18.000000     40.000000  137.500000
# 50%    2023.000000    7.000000   82.950000  132.800000  ...   96.400000      25.350000     59.450000  251.000000
# 75%    2024.000000   10.000000  130.525000  188.275000  ...  136.825000      33.900000     77.675000  352.750000
# max    2025.000000   12.000000  178.800000  249.300000  ...  179.700000      41.900000     94.700000  450.000000

# [8 rows x 11 columns]
# 0        Poor
# 1        Poor
# 2    Moderate
# 3        Poor
# 4        Poor
# Name: Air_Quality, dtype: category
# Categories (3, str): ['Good' < 'Moderate' < 'Poor']
# 0    2
# 1    2
# 2    1
# 3    2
# 4    2
# Name: Air_Quality, dtype: int64

# Columns After Encoding
# ['Year', 'Month', 'PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature_C', 'Humidity_pct', 'AQI', 'Pollution_Index', 'Air_Quality', 'City_MetroB', 'City_MetroC', 'City_MetroD', 'City_MetroE']

# Predictions
# [1 1 2 2 2 1 1 2 2 2 2 2 2 1 1 2 2 0 0 2 0 1 0 2 1 0 2 0 0 0 2 0 2 1 2 2 2
#  1 2 1 2 0 2 2 2 2 2 0 2 2]

# Accuracy
# 0.84

# Confusion Matrix
# [[ 9  4  0]
#  [ 2  6  1]
#  [ 0  1 27]]

# Classification Report
#               precision    recall  f1-score   support

#            0       0.82      0.69      0.75        13
#            1       0.55      0.67      0.60         9
#            2       0.96      0.96      0.96        28

#     accuracy                           0.84        50
#    macro avg       0.78      0.77      0.77        50
# weighted avg       0.85      0.84      0.84        50

# PS C:\Users\UMANG\Desktop\day-1_task-main\Internship\Day-06_(11-07-2026)>  