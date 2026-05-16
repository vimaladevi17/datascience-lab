# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load dataset (change file name as needed)
df = pd.read_csv("raw_merged_heart_dataset.csv")

# Display first 5 rows
print("Original Data:")
print(df.head())

# -----------------------------
# 🔹 1. DATA CLEANING
# -----------------------------

# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values

# For numerical columns → fill with mean
num_imputer = SimpleImputer(strategy='mean')
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# For categorical columns → fill with most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# -----------------------------
# 🔹 2. FEATURE ENCODING
# -----------------------------

# Label Encoding (for ordinal or simple categories)
label_encoder = LabelEncoder()

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

# OR use One-Hot Encoding (better for non-ordinal data)
# df = pd.get_dummies(df, columns=cat_cols)

# -----------------------------
# 🔹 FINAL OUTPUT
# -----------------------------

print("\nCleaned & Encoded Data:")
print(df.head())

# Save cleaned dataset
df.to_csv("cleaned_data.csv", index=False)