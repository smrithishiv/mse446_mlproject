import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Load the merged data
merged_data = pd.read_csv("data/merged_data.csv")

# ✅ Drop rows with missing essential data
merged_data = merged_data.dropna(subset=["Close", "Party", "Industry_Tag", 
                                         "Election_Year_Unemployment_Rate", "Election_Year_Inflation_Rate", 
                                         "Election_Year_Interest_Rate"])

# ✅ Remove outliers based on the 3-sigma rule for stock prices
mean_close = merged_data['Close'].mean()
std_close = merged_data['Close'].std()
upper_bound = mean_close + 3 * std_close
lower_bound = mean_close - 3 * std_close
cleaned_data = merged_data[(merged_data['Close'] >= lower_bound) & (merged_data['Close'] <= upper_bound)]

# ✅ Compute the average stock price for each industry during each presidential term
term_avg_prices = cleaned_data.groupby(['Industry_Tag', 'Election_Date']).agg(
    avg_term_close=('Close', 'mean')
).reset_index()

# ✅ Merge back to get pre-election stock price & compute stock change during the term
cleaned_data = cleaned_data.merge(term_avg_prices, on=['Industry_Tag', 'Election_Date'], how='left')
cleaned_data['Stock_Change_During_Term'] = cleaned_data['avg_term_close'] - cleaned_data['Close']

# ✅ Define features (X) and target variable (y)
feature_columns = ['Close', 'Election_Year_Inflation_Rate', 'Election_Year_Interest_Rate', 
                   'Election_Year_Unemployment_Rate', 'Party', 'Industry_Tag']
X = cleaned_data[feature_columns]
y = cleaned_data['Stock_Change_During_Term']  # Target: Stock price change over the term

# ✅ Preprocessing: Scale numeric features & OneHotEncode categorical ones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), ['Close', 'Election_Year_Unemployment_Rate', 
                                 'Election_Year_Inflation_Rate', 'Election_Year_Interest_Rate']),
        ('cat', OneHotEncoder(handle_unknown="ignore"), ['Party', 'Industry_Tag'])  # One-hot encode categorical features
    ]
)

# ✅ Transform the data
X_transformed = preprocessor.fit_transform(X)

# ✅ Split the transformed data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# ✅ Define the Random Forest Model
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# ✅ Hyperparameter Grid (Prevent Overfitting)
param_dist = {
    'n_estimators': [50, 100, 200],        # Limit tree count
    'max_depth': [3, 5, 7],              # Prevent deep trees
    'min_samples_split': [5, 10, 20],      # Avoid tiny splits
    'min_samples_leaf': [5, 10, 20],        # Ensure minimum samples per leaf
    'max_features': ['sqrt', 'log2'],      # Prevent memorization of all features
    'bootstrap': [True, False]             # Use bootstrap aggregation
}

# ✅ Run RandomizedSearchCV (OUTSIDE PIPELINE)
rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,  # Only 10 iterations to prevent excessive tuning
    cv=5,       # K-Fold Cross Validation (k=5)
    scoring='r2',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# ✅ Train with Hyperparameter Tuning
rf_search.fit(X_train, y_train)

# ✅ Retrieve the best Random Forest model
best_rf = rf_search.best_estimator_
print("\n🚀 Best Random Forest Hyperparameters:", rf_search.best_params_)

# ✅ K-Fold Cross Validation (k=5) on best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring="r2")

# ✅ Make predictions using the best model
y_pred = best_rf.predict(X_test)

# ✅ Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 K-Fold Cross Validation Results (k=5):")
print(f"Mean R²: {np.mean(cross_val_scores):.3f} ± {np.std(cross_val_scores):.3f}")
print("\n📊 Final Random Forest Model Performance:")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# ✅ Monitor Train vs. Test Overfitting
y_train_pred = best_rf.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)
print(f"Train R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")

# ✅ Scatter Plot: Training vs. Test Predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.5, label="Training Data")
plt.scatter(y_test, y_pred, color='red', alpha=0.5, label="Test Data")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Prediction")
plt.xlabel("Actual Stock Change")
plt.ylabel("Predicted Stock Change")
plt.title("Training vs. Test Predictions")
plt.legend()
plt.show()

# ✅ Feature Importance Extraction
importances = best_rf.feature_importances_
feature_names = preprocessor.get_feature_names_out()
sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

print("\n🔍 Feature Importance Ranking:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")
