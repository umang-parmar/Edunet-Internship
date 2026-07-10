import pandas as pd

df = pd.read_csv("CO2_Emissions_1960-2018.csv")


print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())


print("\nShape of Dataset:")
print(df.shape)


print("\nColumn Names:")
print(df.columns)


print("\nNull Values:")
print(df.isnull().sum())


numeric_columns = df.columns[1:]
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

print("\nNull values after filling:")
print(df.isnull().sum())


print("\nSummary Statistics:")
print(df.describe())


print("\nAverage CO2 Emission of Each Year:")
print(df[numeric_columns].mean())


df.to_csv("Cleaned_CO2_Emissions.csv", index=False)

# ANALYSIS


# 1. Every row represents one country.
# 2. Every column from 1960 to 2018 represents CO2 emission for that year.
# 3. Country Name is the only text column.
# 4. All year columns contain numeric values.
# 5. Countries with higher industrial growth generally have higher CO2 emissions.
# 6. CO2 emission values change over time for every country.
# 7. The year columns are related because they show emissions of the same country over different years.
# 8. Missing values were replaced with the average value of their respective year.
# 9. describe() gives count, mean, standard deviation, minimum and maximum values.
# 10. mean() tells the average CO2 emission for each year.

# SUMMARY

# 1. Imported the dataset using Pandas.
# 2. Displayed dataset information.
# 3. Checked rows, columns and data types.
# 4. Found missing values.
# 5. Filled missing values using column mean.
# 6. Generated summary statistics.
# 7. Calculated average CO2 emission for each year.
# 8. Wrote observations about the dataset.
# 9. Saved the cleaned dataset.