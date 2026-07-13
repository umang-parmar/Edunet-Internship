# Renewable Energy Prediction using Linear Regression

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset

energy_df = pd.read_csv("renewable_energy_synthetic.csv")

print(energy_df.head())
print("\nDataset Info")
energy_df.info()

print("\nStatistical Summary")
print(energy_df.describe())

print("\nShape :", energy_df.shape)
print("\nColumns :", energy_df.columns.tolist())

print("\nMissing Values")
print(energy_df.isnull().sum())

# Observations

# 1. Dataset contains renewable energy information.
# 2. Each row represents a country in a particular year.
# 3. Dataset contains numerical and categorical columns.
# 4. CO2 emissions vary with renewable energy production.
# 5. GDP may influence energy generation.
# 6. Renewable share differs among countries.
# 7. Population impacts energy demand.
# 8. Dataset is suitable for regression.
# 9. Country is categorical.
# 10. Carbon emissions are the target variable.

# Select Required Columns

energy_df = energy_df[
    [
        "country",
        "year",
        "population",
        "gdp_billion_usd",
        "solar_twh",
        "wind_twh",
        "hydro_twh",
        "renewable_share_pct",
        "co2_mt"
    ]
]

# Handle Missing Values

energy_df.fillna(energy_df.mean(numeric_only=True), inplace=True)

energy_df["country"].fillna(
    energy_df["country"].mode()[0],
    inplace=True
)

# Feature Engineering

energy_df["Total_Renewable_TWH"] = (
    energy_df["solar_twh"]
    + energy_df["wind_twh"]
    + energy_df["hydro_twh"]
)

# Encode Categorical Column

energy_df = pd.get_dummies(
    energy_df,
    columns=["country"],
    drop_first=True
)

print("\nColumns after Encoding")
print(energy_df.columns.tolist())

# Feature Scaling

scaler = StandardScaler()

numerical = energy_df.select_dtypes(include=np.number).columns
numerical = numerical.drop("co2_mt")

energy_df[numerical] = scaler.fit_transform(
    energy_df[numerical]
)

# Correlation Heatmap

plt.figure(figsize=(12,8))
sns.heatmap(
    energy_df.corr(),
    cmap="coolwarm",
    annot=False
)
plt.show()

# Boxplot

sns.boxplot(
    x=energy_df["Total_Renewable_TWH"]
)

plt.show()

# Pairplot

sns.pairplot(
    energy_df[
        [
            "gdp_billion_usd",
            "population",
            "Total_Renewable_TWH",
            "renewable_share_pct",
            "co2_mt"
        ]
    ]
)

plt.show()

# Distribution Plot

sns.histplot(
    energy_df["co2_mt"],
    kde=True
)

plt.show()

# Prepare Data

X = energy_df.drop("co2_mt", axis=1)

y = energy_df["co2_mt"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Model Training

model = LinearRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("\nPredictions")
print(prediction[:10])

# Evaluation

mae = mean_absolute_error(y_test, prediction)

mse = mean_squared_error(y_test, prediction)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, prediction)

print("\nModel Performance")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

# Conclusion

print("\nConclusion")
print("- Missing values handled.")
print("- Country encoded using one-hot encoding.")
print("- Numerical features standardized.")
print("- Created Total_Renewable_TWH feature.")
print("- Linear Regression trained successfully.")
print("- Model evaluated using MAE, MSE, RMSE and R2 Score.")

# # OUTPUT:
#  py part1.py
#   country  year  population  gdp_billion_usd  solar_twh  wind_twh  hydro_twh  renewable_share_pct  co2_mt
# 0  Brazil  2015   184995841          1923.69       7.70    130.90     283.18                23.87  640.17
# 1  Brazil  2023   182651062          1258.27     158.70     43.56     199.31                69.37  633.37
# 2  Canada  2024   162506646          1208.50     100.80     91.19     276.52                67.25  439.48
# 3   Chile  2012    31358631          1927.01     164.74     63.90     195.86                 8.99  787.05
# 4  Norway  2020   117549530           282.74      25.55     39.96     225.32                55.22  553.25

# Dataset Info
# <class 'pandas.DataFrame'>
# RangeIndex: 200 entries, 0 to 199
# Data columns (total 9 columns):
#  #   Column               Non-Null Count  Dtype  
# ---  ------               --------------  -----  
#  0   country              200 non-null    str    
#  1   year                 200 non-null    int64  
#  2   population           200 non-null    int64  
#  3   gdp_billion_usd      200 non-null    float64
#  4   solar_twh            200 non-null    float64
#  5   wind_twh             200 non-null    float64
#  6   hydro_twh            200 non-null    float64
#  7   renewable_share_pct  200 non-null    float64
#  8   co2_mt               200 non-null    float64
# dtypes: float64(6), int64(2), str(1)
# memory usage: 14.2 KB

# Statistical Summary
#              year    population  gdp_billion_usd   solar_twh    wind_twh   hydro_twh  renewable_share_pct      co2_mt
# count   200.00000  2.000000e+02       200.000000  200.000000  200.000000  200.000000           200.000000  200.000000
# mean   2016.93500  1.047020e+08      1226.680100  134.571600   82.844400  155.332500            47.433450  415.091250
# std       4.66687  5.904564e+07       731.371477   72.927829   52.406252   86.961358            24.511059  224.088024
# min    2010.00000  3.164372e+06        32.470000    1.950000    1.280000    5.650000             5.420000   10.540000
# 25%    2013.00000  5.197677e+07       577.790000   74.102500   39.240000   78.105000            26.837500  236.865000
# 50%    2017.00000  1.053248e+08      1171.580000  138.480000   76.390000  160.250000            48.425000  421.825000
# 75%    2021.00000  1.593158e+08      1886.907500  199.662500  128.547500  229.702500            67.600000  606.755000
# max    2025.00000  1.999412e+08      2499.020000  249.450000  177.870000  298.430000            94.490000  799.920000

# Shape : (200, 9)

# Columns : ['country', 'year', 'population', 'gdp_billion_usd', 'solar_twh', 'wind_twh', 'hydro_twh', 'renewable_share_pct', 'co2_mt']

# Missing Values
# country                0
# year                   0
# population             0
# gdp_billion_usd        0
# solar_twh              0
# wind_twh               0
# hydro_twh              0
# renewable_share_pct    0
# co2_mt                 0
# dtype: int64
# C:\Users\UMANG\Desktop\day-1_task-main\Internship\Day-06_(11-07-2026)\part1.py:74: ChainedAssignmentError: A value is being set on a copy of a DataFrame or Series through chained assignment using an inplace method.
# Such inplace method never works to update the original DataFrame or Series, because the intermediate object on which we are setting values always behaves as a copy (due to Copy-on-Write).

# For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' instead, to perform the operation inplace on the original object, or try to avoid an inplace operation using 'df[col] = df[col].method(value)'.

# See the documentation for a more detailed explanation: https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html
#   energy_df["country"].fillna(

# Columns after Encoding
# ['year', 'population', 'gdp_billion_usd', 'solar_twh', 'wind_twh', 'hydro_twh', 'renewable_share_pct', 'co2_mt', 'Total_Renewable_TWH', 'country_Canada', 'country_Chile', 'country_India', 'country_Japan', 'country_Kenya', 'country_Norway', 'country_Spain']

# Predictions
# [461.36557723 566.47972785 182.3154785  347.8990928  404.11028703
#  489.90843617 323.69223187 464.76655714 451.26682828 324.78577054]

# Model Performance
# MAE : 214.92499408454643
# MSE : 63050.79508491617
# RMSE : 251.09917380373074
# R2 Score : -0.2451873867263712

# Conclusion
# - Missing values handled.
# - Country encoded using one-hot encoding.
# - Numerical features standardized.
# - Created Total_Renewable_TWH feature.
# - Linear Regression trained successfully.
# - Model evaluated using MAE, MSE, RMSE and R2 Score.
# PS C:\Users\UMANG\Desktop\day-1_task-main\Internship\Day-06_(11-07-2026)> 