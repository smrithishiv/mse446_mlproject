import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ✅ Load Dataset
merged_data = pd.read_csv("data/merged_data.csv")

# ✅ Drop unnecessary columns
columns_to_drop = ["Dividends", "Stock Splits", "Year_x", "Year_y", "Election_Date", "End_of_Term"]
merged_data.drop(columns=[col for col in columns_to_drop if col in merged_data.columns], errors="ignore", inplace=True)

# ✅ Convert Dates to Datetime and Remove
if "Date" in merged_data.columns:
    merged_data["Date"] = pd.to_datetime(merged_data["Date"], errors="coerce")
    merged_data.drop(columns=["Date"], inplace=True)

# ✅ Convert Categorical Columns to Numeric
categorical_cols = ["Party", "Opponent_Party", "Industry_Tag", "Country"]
merged_data = pd.get_dummies(merged_data, columns=categorical_cols, drop_first=True)

# ✅ Fix missing 'Capital Gains'
if "Capital Gains" in merged_data.columns:
    merged_data["Capital Gains"] = pd.to_numeric(merged_data["Capital Gains"], errors="coerce").fillna(0)

# ✅ Predicting **Stock Price Change %** Instead of Absolute Price
merged_data["Stock_Change_Percent"] = (merged_data["Close"] - merged_data["Open"]) / merged_data["Open"] * 100

# ✅ Apply **Log Scaling** to Stock Change to Reduce Noise
merged_data["Stock_Change_Log"] = np.sign(merged_data["Stock_Change_Percent"]) * np.log1p(np.abs(merged_data["Stock_Change_Percent"]))
merged_data.replace([np.inf, -np.inf], np.nan, inplace=True)
merged_data["Stock_Change_Log"] = merged_data["Stock_Change_Log"].fillna(0)

# ✅ **Feature Selection Using Correlation Analysis**
correlation_threshold = 0.05
numeric_data = merged_data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
target_corr = correlation_matrix["Stock_Change_Log"].abs().sort_values(ascending=False)
selected_features = target_corr[target_corr > correlation_threshold].index.tolist()
selected_features.remove("Stock_Change_Log")

print(f"Selected Features: {selected_features}")

# ✅ Scale Features
scaler = MinMaxScaler()
merged_data[selected_features] = scaler.fit_transform(merged_data[selected_features])

# ✅ Define Features & Target
X = merged_data[selected_features]
y = merged_data["Stock_Change_Log"]

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ **Hyperparameter Tuning (Less Aggressive)**
param_grid = {
    'n_estimators': [50, 150, 250],  # Lower trees to prevent overfitting
    'max_depth': [4, 8, 12],  # Reduce tree depth
    'min_samples_split': [15, 25, 35],  # More samples per split
    'min_samples_leaf': [10, 15, 20],  # Larger leaf nodes
    'max_features': ['sqrt'],  # Avoid overfitting on all features
    'bootstrap': [True],  # Increase generalization
}

rf_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_
print("Best RF Parameters:", rf_search.best_params_)

# ✅ K-Fold Cross Validation (k=10)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(best_rf, X, y, cv=kf, scoring="r2")
print(f"Mean R²: {np.mean(cross_val_scores):.3f} ± {np.std(cross_val_scores):.3f}")

# ✅ Train Final Model & Evaluate
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

print(f"Final Model Performance:")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# ✅ **Gradient Boosting (Regularized)**
gb_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.07, max_depth=6,  # Lower learning rate
    subsample=0.7, random_state=42  # Lower subsample for more randomness
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print(f"Gradient Boosting R²: {r2_score(y_test, y_pred_gb):.2f}")

# ✅ Graph: Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, label="Random Forest Predictions", alpha=0.6)
plt.scatter(y_test, y_pred_gb, label="Gradient Boosting Predictions", alpha=0.6, marker="x")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Prediction")
plt.xlabel("Actual Log Stock Price Change")
plt.ylabel("Predicted Log Stock Price Change")
plt.legend()
plt.title("Predicted vs. Actual Stock Price Changes")
plt.show()
