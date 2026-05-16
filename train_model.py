import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully!")