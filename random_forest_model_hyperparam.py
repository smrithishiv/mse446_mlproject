import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ✅ Load merged data (single CSV)
merged_data = pd.read_csv("data/merged_data.csv")

# ✅ Select features (including election-related ones)
features = ["Electoral_Vote_Winner", "Popular_Vote_Margin", "Election_Year_Inflation_Rate",
            "Election_Year_Interest_Rate", "Election_Year_Unemployment_Rate", "Total voted",
            "Percent voted", "Open", "High", "Low", "Volume"]

# ✅ Define input (X) and target variable (y)
X = merged_data[features]
y = merged_data["Close"]  # Predicting stock closing price

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# ✅ Define Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 200, 300],                # Number of trees
    'max_depth': [10, 20, 30, None],                # Tree depth
    'min_samples_split': [2, 5, 10],                # Min samples to split
    'min_samples_leaf': [1, 2, 4],                  # Min samples in leaf
    'max_features': ['sqrt', 'log2', None]          # Features used per split
}

# ✅ Run Grid Search for Hyperparameter Tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='r2')

# ✅ Train the optimized model
grid_search.fit(X_train, y_train)

# ✅ Best parameters from GridSearchCV
best_params = grid_search.best_params_
print("\n🚀 Best Hyperparameters:", best_params)

# ✅ Train Random Forest with best parameters
optimized_rf = RandomForestRegressor(**best_params, random_state=42)
optimized_rf.fit(X_train, y_train)

# ✅ Make predictions with optimized model
y_pred = optimized_rf.predict(X_test)

# ✅ Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Optimized Random Forest Performance:")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# ✅ Feature Importance Graph
importances = optimized_rf.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in indices], importances[indices], color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("🔍 Feature Importance in Optimized Random Forest")
plt.gca().invert_yaxis()
plt.show()
