import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your dataset (assumes 'drugs.csv' or replace with correct path)
drugs = pd.read_csv("data.csv")
 # <-- Replace with your actual CSV path
drugs.columns = drugs.columns.str.strip().str.encode('ascii', 'ignore').str.decode('ascii')
# Ensure numeric columns are parsed correctly
columns = ["LogP", "TPSA", "H-Bond Donors", "H-Bond Acceptors", "Binding Affinity", 
           "Target Pro", "Bioavailability", "Toxicity Class (LD50)", "QED Score"]

drugs['Binding Affinity (Ki/IC50)'] = pd.to_numeric(drugs['Binding Affinity (Ki/IC50)'], errors='coerce')

for col in columns:
    drugs[col] = pd.to_numeric(drugs[col], errors='coerce')

# Drop rows with NaN values
drugs = drugs.dropna()

### ---- PCA ANALYSIS (optional, just for visualization) ---- ###
features_for_pca = ['Molecular Weight', 'LogP', 'TPSA', 'H-Bond Donors',
                    'H-Bond Acceptors', 'Binding Affinity (Ki/IC50)',
                    'Target Protein(s)', 'Bioavailability', 'Toxicity Numeric', 'QED Score']

X_pca = drugs[features_for_pca]
X_pca_scaled = StandardScaler().fit_transform(X_pca)

pca = PCA()
pca.fit(X_pca_scaled)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(X_pca.columns))],
    index=X_pca.columns
)
print("PCA Loadings:\n", loadings)

### ---- RFE for feature selection ---- ###
X_rfe = drugs[features_for_pca]
y_rfe = drugs['Binding Affinity (Ki/IC50)']
drugs.drop(columns=['Binding Affinity (Ki/IC50)'],axis=1)
model = LinearRegression()
rfe = RFE(model, n_features_to_select=6)
rfe = rfe.fit(X_rfe, y_rfe)

selected_features = X_rfe.columns[rfe.support_]
feature_rankings = rfe.ranking_

print("Selected Features:", list(selected_features))
print("Feature Rankings:", feature_rankings)

# Drop unimportant features
features_to_drop = X_rfe.columns[feature_rankings > 1]
X_reduced = X_rfe.drop(columns=features_to_drop)
print("Dropped features:", list(features_to_drop))
print("Remaining features:", list(X_reduced.columns))

### ---- CLASSIFICATION (Suitability) ---- ###
classification_features = ['TPSA', 'H-Bond Donors', 
                           'Bioavailability', 'Toxicity Numeric', 'QED Score']

X_cls = drugs[classification_features]
Y_cls = drugs['Suitability']  # Ensure it's binary or multi-class

# Normalize
scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X_cls)

# Train-test split
X_train_cls, X_test_cls, Y_train_cls, Y_test_cls = train_test_split(
    X_cls_scaled, Y_cls, test_size=0.2, random_state=42)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_cls, Y_train_cls)
log_preds = log_model.predict(X_test_cls)

print("\n--- Logistic Regression Classification ---")
print("Accuracy:", accuracy_score(Y_test_cls, log_preds))
print(confusion_matrix(Y_test_cls, log_preds))
print(classification_report(Y_test_cls, log_preds))

### ---- REGRESSION MODELS ---- ###
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reduced, y_rfe, test_size=0.2, random_state=42)

# XGBoost Regressor
xgb_reg = XGBRegressor()
xgb_reg.fit(X_train_reg, y_train_reg)
y_pred_xgb = xgb_reg.predict(X_test_reg)

print("\n--- XGBoost Regression ---")
print("MSE:", mean_squared_error(y_test_reg, y_pred_xgb))
print("R² Score:", r2_score(y_test_reg, y_pred_xgb))

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_reg, y_train_reg)
y_pred_rf = rf.predict(X_test_reg)

print("\n--- Random Forest Regression ---")
print("MSE:", mean_squared_error(y_test_reg, y_pred_rf))
print("R² Score:", r2_score(y_test_reg, y_pred_rf))

### ---- SAVE MODELS ---- ###
joblib.dump(log_model, "logistic_regression_model.pkl")
joblib.dump(scaler_cls, "scaler_cls.pkl")
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(xgb_reg, "xgboost_model.pkl")
joblib.dump(StandardScaler().fit(X_reduced), "scaler_reg.pkl")  # Regression scaler

# Optional getter for Streamlit app
def get_model():
    return log_model  # or change based on your UI model switch
