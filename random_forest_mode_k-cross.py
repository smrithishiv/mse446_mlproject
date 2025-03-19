import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
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

# ✅ Define Random Forest Regressor (Optimized for Speed)
rf = RandomForestRegressor(
    n_estimators=50,         # Reduce number of trees
    max_depth=15,            # Limit tree depth
    min_samples_split=5,     # Prevent small splits
    min_samples_leaf=3,      # Minimum samples in each leaf
    max_features='sqrt',     # Use sqrt of features for each tree
    n_jobs=-1,               # Use all CPU cores for faster processing
    random_state=42
)

# ✅ Define K-Fold Cross Validation (k=3)
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# ✅ Perform Cross Validation
r2_scores = cross_val_score(rf, X, y, cv=kf, scoring="r2")

# ✅ Train the final model on the entire dataset
rf.fit(X, y)

# ✅ Make predictions
y_pred = rf.predict(X)

# ✅ Evaluate Model Performance
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\n📊 Random Forest Performance (K=3 Cross Validation):")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ R² Score (Final Model): {r2:.2f}")
print(f"✅ Mean R² (Cross Validation): {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# ✅ Feature Importance Graph
importances = rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in indices], importances[indices], color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("🔍 Feature Importance in Random Forest (K=3 CV)")
plt.gca().invert_yaxis()
plt.show()
