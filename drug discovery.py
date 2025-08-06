from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
features = ['Molecular Weight','LogP','TPSA','H-Bond Donors','H-Bond Acceptors','Binding Affinity (Ki/IC50)','Target Protein(s)','Bioavailability','Toxicity Numeric','QED Score']                                                               X = drugs[features] 
X_scaled = StandardScaler().fit_transform(X)
pca = PCA()
pca.fit(X_scaled)                                                                                                                                                     loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(X.columns))], index=X.columns)
print(loadings)                                                                                                                                                          X_dropped = X.drop(['H-Bond Donors', 'H-Bond Acceptors', 'Bioavailability'], axis=1, errors='ignore')X_dropped = X.drop(['H-Bond Donors', 'H-Bond Acceptors', 'Bioavailability'], axis=1, errors='ignore')                                                                                                                                            np.seterr(divide='warn', invalid='warn')
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression                                                                                              y = drugs['Binding Affinity (Ki/IC50)'] 
model = LinearRegression()

rfe = RFE(model, n_features_to_select=6)
rfe = rfe.fit(X, y)

print("Selected Features:", X.columns[rfe.support_])
print("Feature Rankings:", rfe.ranking_)                                                                                                     eature_rankings = rfe.ranking_

# Get names of features to drop (those with ranking > 1)
features_to_drop = X.columns[feature_rankings > 1]

# Drop those features
X_reduced = X.drop(columns=features_to_drop)

# Optional: Update main dataframe (if applicable)
# df = df.drop(columns=features_to_drop)

# Output
print("Dropped features:", list(features_to_drop))
print("Remaining features:", list(X_reduced.columns))                                                                                           drugs['TPSA'] = pd.to_numeric(drugs['TPSA'], errors='coerce')
drugs['H-Bond Donors'] = pd.to_numeric(drugs['H-Bond Donors'], errors='coerce')
drugs['Binding Affinity (Ki/IC50)'] = pd.to_numeric(drugs['Binding Affinity (Ki/IC50)'], errors='coerce')
drugs['Bioavailability'] = pd.to_numeric(drugs['Bioavailability'], errors='coerce')
drugs['QED Score'] = pd.to_numeric(drugs['QED Score'], errors='coerce')                                                 from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report                                      features = ['TPSA', 'H-Bond Donors', 'Binding Affinity (Ki/IC50)',
       'Bioavailability', 'Toxicity Numeric', 'QED Score']
X = drugs[features]
Y= drugs['Suitability']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)                      from sklearn.linear_model import LogisticRegressionlog_model = LogisticRegression()
log_model.fit(X_train, Y_train)

log_preds = log_model.predict(X_test)

print("Logistic Regression")
print("Accuracy:", accuracy_score(Y_test, log_preds))
print(confusion_matrix(Y_test, log_preds))
print(classification_report(Y_test, log_preds))                                                                                                   !pip install xgboost                                                                                                                                        from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split                                                                                        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Initialize model
xgb_reg = XGBRegressor()
xgb_reg.fit(X_train, y_train)

y_pred = xgb_reg.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("XGBoost Regression MSE:", mse)
print("XGBoost R2 Score:", r2)                                                                                                                         from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Regression MSE:", mse)
print("Random Forest R2 Score:", r2)          import joblib

# Save classification (suitability) model
joblib.dump(log_model, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler_cls.pkl")

# Save regression models
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(xgb_reg, "xgboost_model.pkl")
joblib.dump(scaler, "scaler_reg.pkl")  # Assuming same scaler for regression
