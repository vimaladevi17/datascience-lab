# ===============================
# Import Required Libraries
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt

# ===============================
# Load Dataset (Replace with your file path)
# ===============================
df = pd.read_csv(r"C:\Users\VIMALA\Downloads\datascience lab\cleaned_data.csv")

# ===============================
# Define Features and Target
# ===============================
X = df.drop("target", axis=1)   # Replace 'target' with your target column
y = df["target"]

# ===============================
# Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===============================
# 1️⃣ Random Forest Model
# ===============================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

# Metrics
rf_auc = roc_auc_score(y_test, rf_probs)
rf_precision = precision_score(y_test, rf_preds)
rf_recall = recall_score(y_test, rf_preds)

# ===============================
# 2️⃣ Gradient Boosting Model
# ===============================
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

gb_preds = gb.predict(X_test)
gb_probs = gb.predict_proba(X_test)[:, 1]

# Metrics
gb_auc = roc_auc_score(y_test, gb_probs)
gb_precision = precision_score(y_test, gb_preds)
gb_recall = recall_score(y_test, gb_preds)

# ===============================
# Print Results
# ===============================
print("===== Random Forest =====")
print("AUC:", rf_auc)
print("Precision:", rf_precision)
print("Recall:", rf_recall)

print("\n===== Gradient Boosting =====")
print("AUC:", gb_auc)
print("Precision:", gb_precision)
print("Recall:", gb_recall)

# ===============================
# Plot ROC Curve
# ===============================
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)

plt.figure()
plt.plot(rf_fpr, rf_tpr, label="Random Forest")
plt.plot(gb_fpr, gb_tpr, label="Gradient Boosting")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()