import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Generate Student Dataset

def create_student_dataset(records=1000):

    np.random.seed(100)

    streams = ['CSE', 'IT', 'ECE', 'Mechanical', 'Civil']
    genders = ['Male', 'Female']

    df = pd.DataFrame({
        "Department": np.random.choice(streams, records),
        "Gender": np.random.choice(genders, records)
    })

    df["Maths"] = np.round(np.clip(np.random.normal(7, 1.2, records), 1, 10))
    df["English"] = np.round(np.clip(np.random.normal(6.5, 1.1, records), 1, 10))
    df["Science"] = np.round(np.clip(np.random.normal(5.8, 1.3, records), 1, 10))

    df["Total Marks"] = df["Maths"] + df["English"] + df["Science"]

    def grade(total):
        if total >= 25:
            return "A"
        elif total >= 19:
            return "B"
        elif total >= 15:
            return "C"
        return "D"

    df["Grade"] = df["Total Marks"].apply(grade)

    df["Fuel Consumption"] = np.round(
        np.clip(np.random.normal(11, 2.8, records), 2, 20), 2
    )

    return df

# Create Dataset

student_df = create_student_dataset(1000)

print("Original Dataset")
print(student_df.head())

# Copy Dataset

processed_df = student_df.copy()

# One-Hot Encoding

processed_df = pd.get_dummies(
    processed_df,
    columns=["Department", "Gender"],
    dtype=int
)

# Ordinal Encoding

encoder = OrdinalEncoder(categories=[["D", "C", "B", "A"]])

processed_df[["Grade"]] = encoder.fit_transform(processed_df[["Grade"]])

# Standard Scaling

scaler = StandardScaler()

numerical_columns = [
    "Maths",
    "English",
    "Science",
    "Total Marks",
    "Fuel Consumption"
]

processed_df[numerical_columns] = scaler.fit_transform(
    processed_df[numerical_columns]
)

# Display Processed Data

print("\nProcessed Dataset")
print(processed_df.head())