'''
Experiment Title:

Hyperparameter Optimization of Random Forest and Gradient Boosting Classifiers using GridSearchCV.

Aim:

To implement Random Forest and Gradient Boosting classification algorithms, perform hyperparameter tuning using GridSearchCV with Stratified K-Fold cross-validation, and evaluate model performance using AUC, Precision, and Recall metrics.

Description:

In machine learning, ensemble methods improve predictive performance by combining multiple base learners. Random Forest and Gradient Boosting are two widely used ensemble techniques for classification problems.

In this experiment, both models are trained and optimized using GridSearchCV to identify the best hyperparameter combinations that maximize model performance.

1. Random Forest

Random Forest is a bagging-based ensemble method that constructs multiple decision trees using bootstrapped samples of the dataset. Each tree is trained independently, and the final prediction is obtained using majority voting.

Key characteristics:

->Reduces overfitting
->Handles high-dimensional data
->Robust to noise
->Supports feature importance estimation

Hyperparameters tuned include:

->Number of trees (n_estimators)
->Maximum tree depth (max_depth)
->Minimum samples split (min_samples_split)
->Minimum samples leaf (min_samples_leaf)
->Maximum features considered (max_features)


2. Gradient Boosting

Gradient Boosting is a boosting-based ensemble method that builds trees sequentially. Each new tree corrects the errors of the previous trees by minimizing a loss function using gradient descent optimization.

Key characteristics:

High predictive accuracy
Sequential learning
Minimizes loss function
Sensitive to hyperparameter tuning

Hyperparameters tuned include:

Number of estimators
Learning rate
Maximum depth
Subsample ratio
Minimum samples split and leaf

3. Hyperparameter Tuning using GridSearchCV

GridSearchCV performs an exhaustive search over a predefined parameter grid. It uses cross-validation to evaluate all parameter combinations and selects the best model based on a scoring metric.

In this experiment:

Stratified K-Fold Cross Validation is used to maintain class balance.
ROC-AUC is used as the scoring metric.
The best model is selected based on maximum mean cross-validation AUC.

'''
# ==========================================================
# Hyperparameter Tuning using GridSearchCV
# Random Forest & Gradient Boosting
# Evaluation: AUC, Precision, Recall
# ==========================================================

# ===============================
# 1. Import Required Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    roc_curve,
    classification_report
)

# ===============================
# 2. Load Dataset
# ===============================
# Replace with your dataset path
df = pd.read_csv(r"C:\Users\VIMALA\Downloads\datascience lab\cleaned_data.csv")

# Replace 'target' with your actual target column name
X = df.drop("target", axis=1)
y = df["target"]

# ===============================
# 3. Train-Test Split (Stratified)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ===============================
# 4. Cross Validation Strategy
# ===============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================================
# 5. RANDOM FOREST - Grid Search
# ==========================================================

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)

rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

rf_grid.fit(X_train, y_train)

print("\n==============================")
print("Best Random Forest Parameters")
print("==============================")
print(rf_grid.best_params_)
print("Best CV AUC:", rf_grid.best_score_)

# Best RF Model
best_rf = rf_grid.best_estimator_

# Predictions
rf_preds = best_rf.predict(X_test)
rf_probs = best_rf.predict_proba(X_test)[:, 1]

# Evaluation Metrics
rf_auc = roc_auc_score(y_test, rf_probs)
rf_precision = precision_score(y_test, rf_preds)
rf_recall = recall_score(y_test, rf_preds)

print("\n===== Tuned Random Forest Performance =====")
print("Test AUC:", rf_auc)
print("Precision:", rf_precision)
print("Recall:", rf_recall)

# ==========================================================
# 6. GRADIENT BOOSTING - Grid Search
# ==========================================================

gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)

gb_grid = GridSearchCV(
    estimator=gb,
    param_grid=gb_param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

gb_grid.fit(X_train, y_train)

print("\n==============================")
print("Best Gradient Boosting Parameters")
print("==============================")
print(gb_grid.best_params_)
print("Best CV AUC:", gb_grid.best_score_)

# Best GB Model
best_gb = gb_grid.best_estimator_

# Predictions
gb_preds = best_gb.predict(X_test)
gb_probs = best_gb.predict_proba(X_test)[:, 1]

# Evaluation Metrics
gb_auc = roc_auc_score(y_test, gb_probs)
gb_precision = precision_score(y_test, gb_preds)
gb_recall = recall_score(y_test, gb_preds)

print("\n===== Tuned Gradient Boosting Performance =====")
print("Test AUC:", gb_auc)
print("Precision:", gb_precision)
print("Recall:", gb_recall)

# ==========================================================
# 7. ROC Curve Comparison
# ==========================================================

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)

plt.figure()
plt.plot(rf_fpr, rf_tpr, label="Random Forest")
plt.plot(gb_fpr, gb_tpr, label="Gradient Boosting")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# ==========================================================
# 8. Final Model Comparison
# ==========================================================

results = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting"],
    "AUC": [rf_auc, gb_auc],
    "Precision": [rf_precision, gb_precision],
    "Recall": [rf_recall, gb_recall]
})

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")
print(results)

# ==========================================================
# 9. Classification Reports
# ==========================================================

print("\nRandom Forest Classification Report")
print(classification_report(y_test, rf_preds))

print("\nGradient Boosting Classification Report")
print(classification_report(y_test, gb_preds))