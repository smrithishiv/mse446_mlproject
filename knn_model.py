import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# Load datasets
elections = pd.read_csv("data/us_presidential_elections_2000_2024.csv")
stocks = pd.read_csv("data/USA_Stock_Prices.csv")
voters = pd.read_csv("data/voter_demographics_data.csv")

# Convert dates to datetime format
elections["Election_Date"] = pd.to_datetime(elections["Election_Date"], errors="coerce", utc=True)
stocks["Date"] = pd.to_datetime(stocks["Date"], errors="coerce", utc=True)

# Merge stock data with election results (using past election impact)
merged_data = pd.merge(stocks, elections, left_on="Date", right_on="Election_Date", how="left")

# Merge voter demographics on Election Year
voters["Years"] = pd.to_numeric(voters["Years"], errors="coerce")  # Ensure numeric years
voters["Years"] = voters["Years"].ffill().astype(int)  # Fix the deprecation warning
merged_data = pd.merge(merged_data, voters, left_on="Year", right_on="Years", how="left")

# Drop unnecessary columns
merged_data.drop(columns=["Election_Date", "Inaugration_Date", "End_of_Term", "Years"], inplace=True)

# ✅ Remove Non-Numeric Columns Before Filling NaNs
non_numeric_cols = ["Brand_Name", "Ticker", "Industry_Tag", "Country"]
merged_data.drop(columns=[col for col in non_numeric_cols if col in merged_data.columns], inplace=True)

# ✅ Fill missing values for numerical columns using median
numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].median())

# Select features (Including election-related ones)
features = ["Electoral_Vote_Winner", "Popular_Vote_Margin", "Election_Year_Inflation_Rate",
            "Election_Year_Interest_Rate", "Election_Year_Unemployment_Rate", "Total voted",
            "Percent voted", "Open", "High", "Low", "Volume"]

# Ensure categorical encoding for political parties
merged_data = pd.get_dummies(merged_data, columns=["Party", "Opponent_Party"], drop_first=True)

# ✅ Feature Scaling (Apply MinMaxScaler to normalize all features)
scaler = MinMaxScaler()
merged_data[features] = scaler.fit_transform(merged_data[features])

# Define input (X) and target variable (y)
X = merged_data[features]
y = merged_data["Close"]  # Predicting stock closing price

# ✅ Ensure X has no missing values
if X.isnull().sum().sum() > 0:
    print("Warning: X contains NaN values after preprocessing!")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Ensure X_train and X_test have no NaN values before training
if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
    raise ValueError("Error: X_train or X_test still contains NaN values!")

# ✅ Train KNN Model
knn_model = KNeighborsRegressor(n_neighbors=5, weights="distance")  # Weighted by inverse distance
knn_model.fit(X_train, y_train)

# Make predictions
y_pred = knn_model.predict(X_test)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (KNN): {mae:.2f}")

# ✅ Plot Actual vs Predicted Prices (2020 Onwards)
X_test["Date"] = merged_data.loc[X_test.index, "Date"]
X_test_filtered = X_test[X_test["Date"] >= "2020-01-01"]

# Sort test set by Date
X_test_sorted = X_test_filtered.sort_values(by="Date")
y_test_sorted = y_test.loc[X_test_sorted.index]
y_pred_sorted = pd.Series(y_pred, index=X_test.index).loc[X_test_sorted.index]

plt.figure(figsize=(12, 6))
plt.plot(X_test_sorted["Date"], y_test_sorted, label="Actual Prices", color="blue", alpha=0.6)
plt.plot(X_test_sorted["Date"], y_pred_sorted, label="Predicted Prices", color="red", linestyle="dashed", alpha=0.3)
plt.xlabel("Date")
plt.ylabel("Stock Price (Close)")
plt.title("Actual vs Predicted Stock Prices (KNN, 2020 Onwards)")
plt.legend()
plt.show()

# ✅ Scatter Plot (Actual vs Predicted)
plt.figure(figsize=(8, 8))
plt.scatter(y_test_sorted, y_pred_sorted, alpha=0.6, color="purple")
plt.xlabel("Actual Stock Price")
plt.ylabel("Predicted Stock Price")
plt.title("Actual vs Predicted Stock Prices (KNN)")
plt.axline((0, 0), slope=1, color="black", linestyle="dashed")  # Perfect prediction line
plt.show()

print(merged_data.isnull().sum()[merged_data.isnull().sum() > 0])
