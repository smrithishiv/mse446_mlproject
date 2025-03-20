import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

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

# Forcing Election Data by Increasing Feature Importance
election_features = ["Electoral_Vote_Winner", "Popular_Vote_Margin", "Election_Year_Inflation_Rate",
                     "Election_Year_Interest_Rate", "Election_Year_Unemployment_Rate", "Total voted",
                     "Percent voted"]

# Duplicate election features to increase importance
for feature in election_features:
    merged_data[f"{feature}_Weighted"] = merged_data[feature] * 100

scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # Ensure election data remains more dominant
merged_data[election_features] = scaler.fit_transform(merged_data[election_features])

# Define models to compare
models = {
    "Random Forest (All Features)": election_features + ["Open", "High", "Low", "Volume"],
    "Random Forest (Election-Only)": election_features + [f"{feature}_Weighted" for feature in election_features]
}

# Store results
results = {}

for model_name, selected_features in models.items():
    print(f"\n Training {model_name}")

    # Define input (X) and target variable (y)
    X = merged_data[selected_features]
    y = merged_data["Close"]  # Predicting stock closing price

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate Model Performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[model_name] = {
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "rf_model": rf_model,
        "X_train": X_train,
        "X_test": X_test
    }

    print(f" {model_name} Performance:")
    print(f" Mean Absolute Error (MAE): {mae:.2f}")
    print(f" Mean Squared Error (MSE): {mse:.2f}")
    print(f" R² Score: {r2:.2f}")

# Graph 1: Actual vs Predicted Prices
# plt.figure(figsize=(12, 5))

# for i, (model_name, data) in enumerate(results.items()):
#     plt.subplot(1, 2, i+1)
#     plt.plot(data["y_test"].values, label="Actual Prices", color="blue", alpha=0.6)
#     plt.plot(data["y_pred"], label="Predicted Prices", color="red", linestyle="dashed", alpha=0.7)
#     plt.xlabel("Test Sample Index")
#     plt.ylabel("Stock Closing Price")
#     plt.title(f" Actual vs Predicted ({model_name})")
#     plt.legend()

# plt.tight_layout()
# plt.show()

# # Graph 2: Residual Distribution (Errors)
# plt.figure(figsize=(12, 5))

# for i, (model_name, data) in enumerate(results.items()):
#     residuals = data["y_test"] - data["y_pred"]
#     plt.subplot(1, 2, i+1)
#     sns.histplot(residuals, kde=True, bins=30, color="purple")
#     plt.xlabel("Prediction Error (Residual)")
#     plt.ylabel("Frequency")
#     plt.title(f" Residual Distribution ({model_name})")

# plt.tight_layout()
# plt.show()

# # Graph 3: Feature Importance
# plt.figure(figsize=(12, 5))

# for i, (model_name, data) in enumerate(results.items()):
#     importances = data["rf_model"].feature_importances_
#     feature_names = data["X_train"].columns
#     indices = np.argsort(importances)[::-1]

#     plt.subplot(1, 2, i+1)
#     plt.barh([feature_names[i] for i in indices], importances[indices], color="skyblue")
#     plt.xlabel("Importance Score")
#     plt.ylabel("Feature")
#     plt.title(f" Feature Importance ({model_name})")
#     plt.gca().invert_yaxis()

# plt.tight_layout()
# plt.show()

# Convert results dictionary into a DataFrame for better visualization
results_df = pd.DataFrame([
    {
        "Model": model_name,
        "Mean Absolute Error (MAE)": data["mae"],
        "Mean Squared Error (MSE)": data["mse"],
        "R² Score": data["r2"]
    }
    for model_name, data in results.items()
])

# Create a DataFrame to compare actual vs predicted stock prices for each model
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

# Display DataFrames
print("\n Model Performance Metrics:")
print(results_df)

print("\n Predicted vs. Actual Stock Prices:")
print(predictions_df)

# (Optional) Save DataFrames to CSV for future analysis
results_df.to_csv("random_forest/random_forest_results.csv", index=False)
predictions_df.to_csv("random_forest/predicted_vs_actual_stock_prices.csv", index=False)
