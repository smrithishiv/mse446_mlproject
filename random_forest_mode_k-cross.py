import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# Load datasets
elections = pd.read_csv("data/us_presidential_elections_2000_2024.csv")
stocks = pd.read_csv("data/USA_Stock_Prices.csv")
voters = pd.read_csv("data/voter_demographics_data.csv")

# Convert dates to datetime format
elections["Election_Date"] = pd.to_datetime(elections["Election_Date"], errors="coerce", utc=True)
stocks["Date"] = pd.to_datetime(stocks["Date"], errors="coerce", utc=True)

# Extract year from stock dates for merging
stocks["Year"] = stocks["Date"].dt.year

# Extract election year from election date for merging
elections["Election_Year"] = elections["Election_Date"].dt.year

# Merge stock data with elections (Forward-fill election data)
merged_data = pd.merge(stocks, elections, how="left", left_on="Year", right_on="Election_Year")
merged_data = merged_data.sort_values(by="Date").ffill()  # Forward fill missing election data

# Ensure Year is still in merged_data
if "Election_Year" in merged_data.columns and "Year" not in merged_data.columns:
    merged_data.rename(columns={"Election_Year": "Year"}, inplace=True)

# Convert voter demographics to numeric (Fix String-to-Float Errors)
voters["Years"] = pd.to_numeric(voters["Years"], errors="coerce").astype(int)  # Ensure numeric
for col in voters.columns:
    if col != "Years":
        voters[col] = voters[col].astype(str).str.replace(",", "").astype(float)  # Remove commas & convert to float

# Merge voter data and forward-fill for non-election years
merged_data = pd.merge(merged_data, voters, how="left", left_on="Year", right_on="Years")
merged_data = merged_data.sort_values(by="Date").ffill()

# Drop unnecessary columns
columns_to_drop = ["Election_Date", "Inaugration_Date", "End_of_Term", "Election_Year", "Years"]
merged_data.drop(columns=[col for col in columns_to_drop if col in merged_data.columns], errors="ignore", inplace=True)

# Fix missing 'Capital Gains' by ensuring it’s numeric and filling NaNs
if "Capital Gains" in merged_data.columns:
    if merged_data["Capital Gains"].dtype == object:  # Convert string to numeric if necessary
        merged_data["Capital Gains"] = merged_data["Capital Gains"].astype(str).str.replace(",", "").astype(float)
    merged_data["Capital Gains"] = merged_data["Capital Gains"].fillna(0)  # Fill missing values with 0

# Final check for null values
null_counts = merged_data.isnull().sum()[merged_data.isnull().sum() > 0]
if null_counts.empty:
    print("No null values remain in merged_data!")
else:
    print("Null values still exist:", null_counts)

# ✅ **Forcing Election Data by Increasing Feature Importance**
election_features = ["Electoral_Vote_Winner", "Popular_Vote_Margin", "Election_Year_Inflation_Rate",
                     "Election_Year_Interest_Rate", "Election_Year_Unemployment_Rate", "Total voted",
                     "Percent voted"]

# Duplicate election features to increase importance
for feature in election_features:
    merged_data[f"{feature}_Weighted"] = merged_data[feature] * 100

scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # Ensure election data remains more dominant
merged_data[election_features] = scaler.fit_transform(merged_data[election_features])

from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ✅ Define K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold CV

# Store results
results = {}

# Store predictions for actual vs predicted DataFrame
all_predictions = []

for model_name, selected_features in models.items():
    print(f"\n🔹 Training {model_name} with K-Fold Cross Validation")

    # Define input (X) and target variable (y)
    X = merged_data[selected_features]
    y = merged_data["Close"]  # Predicting stock closing price

    # Initialize Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42)

    # Lists to store cross-validation scores
    mae_scores = []
    mse_scores = []
    r2_scores = []
    y_tests = []
    y_preds = []

    # Perform K-Fold Cross Validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Store actual and predicted values
        y_tests.extend(y_test)
        y_preds.extend(y_pred)

        # Evaluate Model Performance
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    # Store the average cross-validation results
    results[model_name] = {
        "MAE (Mean)": np.mean(mae_scores),
        "MSE (Mean)": np.mean(mse_scores),
        "R² Score (Mean)": np.mean(r2_scores),
        "MAE (Std Dev)": np.std(mae_scores),
        "MSE (Std Dev)": np.std(mse_scores),
        "R² Score (Std Dev)": np.std(r2_scores),
        "rf_model": rf_model,
        "features": selected_features,
        "y_tests": np.array(y_tests),
        "y_preds": np.array(y_preds),
    }

    print(f"📊 {model_name} Cross-Validation Performance:")
    print(f"✅ Mean Absolute Error (MAE): {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")
    print(f"✅ Mean Squared Error (MSE): {np.mean(mse_scores):.2f} ± {np.std(mse_scores):.2f}")
    print(f"✅ R² Score: {np.mean(r2_scores):.2f} ± {np.std(r2_scores):.2f}")

    # Collect actual vs predicted values for DataFrame
    model_predictions = pd.DataFrame({"Model": model_name, "Actual Price": y_tests, "Predicted Price": y_preds})
    all_predictions.append(model_predictions)

# ✅ Convert results into a DataFrame for visualization
results_df = pd.DataFrame.from_dict(results, orient="index")

# ✅ Display the results DataFrame
import ace_tools as tools
tools.display_dataframe_to_user(name="K-Fold Cross Validation Results", dataframe=results_df)

# ✅ Create a DataFrame to compare actual vs predicted stock prices for each model
predictions_df = pd.concat(all_predictions, ignore_index=True)

# ✅ Display Predictions DataFrame
tools.display_dataframe_to_user(name="Actual vs Predicted Prices", dataframe=predictions_df)


# ✅ Graph 1️⃣: Actual vs Predicted Prices
# plt.figure(figsize=(12, 5))

# for i, (model_name, data) in enumerate(results.items()):
#     plt.subplot(1, 2, i+1)
#     plt.plot(data["y_test"].values, label="Actual Prices", color="blue", alpha=0.6)
#     plt.plot(data["y_pred"], label="Predicted Prices", color="red", linestyle="dashed", alpha=0.7)
#     plt.xlabel("Test Sample Index")
#     plt.ylabel("Stock Closing Price")
#     plt.title(f"📈 Actual vs Predicted ({model_name})")
#     plt.legend()

# plt.tight_layout()
# plt.show()

# # ✅ Graph 2️⃣: Residual Distribution (Errors)
# plt.figure(figsize=(12, 5))

# for i, (model_name, data) in enumerate(results.items()):
#     residuals = data["y_test"] - data["y_pred"]
#     plt.subplot(1, 2, i+1)
#     sns.histplot(residuals, kde=True, bins=30, color="purple")
#     plt.xlabel("Prediction Error (Residual)")
#     plt.ylabel("Frequency")
#     plt.title(f"📊 Residual Distribution ({model_name})")

# plt.tight_layout()
# plt.show()

# # ✅ Graph 3️⃣: Feature Importance
# plt.figure(figsize=(12, 5))

# for i, (model_name, data) in enumerate(results.items()):
#     importances = data["rf_model"].feature_importances_
#     feature_names = data["X_train"].columns
#     indices = np.argsort(importances)[::-1]

#     plt.subplot(1, 2, i+1)
#     plt.barh([feature_names[i] for i in indices], importances[indices], color="skyblue")
#     plt.xlabel("Importance Score")
#     plt.ylabel("Feature")
#     plt.title(f"🔍 Feature Importance ({model_name})")
#     plt.gca().invert_yaxis()

# plt.tight_layout()
# plt.show()

# ✅ Convert results dictionary into a DataFrame for better visualization
results_df = pd.DataFrame([
    {
        "Model": model_name,
        "Mean Absolute Error (MAE)": data["mae"],
        "Mean Squared Error (MSE)": data["mse"],
        "R² Score": data["r2"]
    }
    for model_name, data in results.items()
])

# ✅ Create a DataFrame to compare actual vs predicted stock prices for each model
predictions_df_list = []

for model_name, data in results.items():
    comparison_df = pd.DataFrame({
        "Model": model_name,
        "Actual Price": data["y_test"].values,
        "Predicted Price": data["y_pred"]
    })
    predictions_df_list.append(comparison_df)

# Combine results from all models into a single DataFrame
predictions_df = pd.concat(predictions_df_list, ignore_index=True)

# ✅ Display DataFrames
print("\n📊 Model Performance Metrics:")
print(results_df)

print("\n📈 Predicted vs. Actual Stock Prices:")
print(predictions_df)

# ✅ (Optional) Save DataFrames to CSV for future analysis
results_df.to_csv("random_forest_results.csv", index=False)
predictions_df.to_csv("predicted_vs_actual_stock_prices.csv", index=False)
