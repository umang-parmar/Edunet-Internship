# Import Libraries

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Dataset

df = pd.read_csv("renewable_energy.csv")

print("Original Dataset:\n")
print(df)

# Basic Information

print("\n1. Information:")
df.info()

print("\n2. Describe Data:")
print(df.describe())

print("\n3. Shape:")
print(df.shape)

print("\n4. Columns:")
print(df.columns)

# Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

# Handle Missing Values

# Numerical Columns -> Mean
df["Energy Production (GWh)"] = df["Energy Production (GWh)"].fillna(
    df["Energy Production (GWh)"].mean())

df["Investment (Million $)"] = df["Investment (Million $)"].fillna(
    df["Investment (Million $)"].mean())

df["CO2 Reduction (Tons)"] = df["CO2 Reduction (Tons)"].fillna(
    df["CO2 Reduction (Tons)"].mean())

df["Jobs Created"] = df["Jobs Created"].fillna(
    df["Jobs Created"].mean())

df["Efficiency (%)"] = df["Efficiency (%)"].fillna(
    df["Efficiency (%)"].mean())

# Categorical Columns -> Mode
df["Energy Source"] = df["Energy Source"].fillna(
    df["Energy Source"].mode()[0])

df["Region"] = df["Region"].fillna(
    df["Region"].mode()[0])

print("\nDataset After Handling Missing Values:")
print(df)

# Feature Engineering

# Investment per GWh
df["Investment_per_GWh"] = (
    df["Investment (Million $)"] /
    df["Energy Production (GWh)"]
)

# CO2 Reduction per Job
df["CO2_per_Job"] = (
    df["CO2 Reduction (Tons)"] /
    df["Jobs Created"]
)

print("\nDataset After Feature Engineering:")
print(df)

# Generalization

print("\nAverage Energy Production by Energy Source:")
print(df.groupby("Energy Source")["Energy Production (GWh)"].mean())

print("\nAverage Efficiency by Region:")
print(df.groupby("Region")["Efficiency (%)"].mean())

print("\nTotal Investment by Energy Source:")
print(df.groupby("Energy Source")["Investment (Million $)"].sum())

# Correlation

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# Label Encoding

encoder = LabelEncoder()

df["Energy Source"] = encoder.fit_transform(df["Energy Source"])
df["Region"] = encoder.fit_transform(df["Region"])

print("\nAfter Label Encoding:")
print(df)

# One Hot Encoding

dummy_df = pd.get_dummies(
    df,
    columns=["Energy Source", "Region"]
)

print("\nOne Hot Encoded Dataset:")
print(dummy_df)

# Standard Scaling

scaler = StandardScaler()

numerical_columns = [
    "Energy Production (GWh)",
    "Investment (Million $)",
    "CO2 Reduction (Tons)",
    "Jobs Created",
    "Efficiency (%)",
    "Investment_per_GWh",
    "CO2_per_Job"
]

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("\nAfter Standard Scaling:")
print(df)

# Final Dataset Information

print("\nFinal Dataset Information:")
df.info()

print("\nFinal Dataset Description:")
print(df.describe())

print("\nAssignment Completed Successfully")

#-----Generalizations------#

# 1. Geothermal and Hydropower projects often show stable energy generation across different years.
# 2. Renewable energy investments differ significantly across regions and energy sources.
# 3. Projects with greater energy production tend to contribute more to CO2 emission reduction.
# 4. Higher-efficiency renewable projects generally provide better environmental benefits.
# 5. Job creation varies depending on the type and scale of the renewable energy project.
# 6. Wind and Solar projects are among the most commonly represented renewable energy sources.
# 7. Investment alone does not always guarantee the highest energy production, indicating the impact of project efficiency.
# 8. Different regions demonstrate varying levels of renewable energy performance and adoption.
# 9. Missing values in the dataset should be handled before performing analysis or building machine learning models.
# 10. Renewable energy development supports sustainable growth through cleaner power generation and employment opportunities.

#------Why Mean Imputation?------#

# Mean imputation fills missing numerical values using the column average.
# It prevents data loss by keeping all rows in the dataset.
# It is effective when only a few numerical values are missing.

#--------Why Feature Engineering?-------#

# Feature engineering creates meaningful variables from existing data.
# New features help identify hidden patterns and improve data analysis.
# These engineered features can increase the performance of machine learning models.

#------Why Label Encoding?------#

# Label Encoding assigns a unique integer to each category.
# It transforms categorical features into a machine-readable format.
# This allows machine learning models to process categorical data efficiently.

#------Why One-Hot Encoding?------#

# Represents each category using 0s and 1s.
# Ensures all categories are treated equally by the model.
# Commonly used for nominal variables such as Region and Energy Source.

#--------Why StandardScaler?-------#

# StandardScaler transforms numerical features to have a mean of 0 and a standard deviation of 1.
# It prevents features with large values from dominating the model.
# This results in faster and more stable model training.

#------Final Insights------#

# More investment usually produces more renewable energy.
# Better efficiency helps generate more energy.
# Renewable energy projects help reduce CO2 pollution.
# These projects also create more job opportunities.
# New features helped us understand the data better.
# Scaling made all numerical values similar in range.
# Encoding changed text data into numbers so the model can use it.

