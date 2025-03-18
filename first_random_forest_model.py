import pandas as pd
import numpy as np

# Load datasets
elections = pd.read_csv("data/us_presidential_elections_2000_2024.csv")
stocks = pd.read_csv("data/USA_Stock_Prices.csv")
voters = pd.read_csv("data/voter_demographics_data.csv")

# Convert dates to datetime format
elections["Election_Date"] = pd.to_datetime(elections['Election_Date'], errors='coerce', utc=True)
stocks["Date"] = pd.to_datetime(stocks["Date"], errors='coerce', utc=True)

# Merge stock data with election results (using past election impact)
merged_data = pd.merge(stocks, elections, left_on="Date", right_on="Election_Date", how="left")

# Merge voter demographics on Election Year
voters["Years"] = voters["Years"].astype(int)
merged_data = pd.merge(merged_data, voters, left_on="Year", right_on="Years", how="left")

# Drop unnecessary columns (like End_of_Term)
merged_data.drop(columns=["Election_Date", "Inaugration_Date", "End_of_Term", "Years"], inplace=True)

# Fill missing values
merged_data.fillna(method="ffill", inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Select features
features = ["Electoral_Vote_Winner", "Popular_Vote_Margin", "Election_Year_Inflation_Rate",
            "Election_Year_Interest_Rate", "Election_Year_Unemployment_Rate", "Total voted",
            "Percent voted", "Open", "High", "Low", "Volume"]

# Encode categorical variables
merged_data = pd.get_dummies(merged_data, columns=["Party", "Opponent_Party"], drop_first=True)

X = merged_data[features]
y = merged_data["Close"]  # Predicting closing price

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

import matplotlib.pyplot as plt

# Ensure Date column is retrieved and in datetime format
X_test["Date"] = pd.to_datetime(merged_data.loc[X_test.index, "Date"])

# Filter data from 2020 onwards
X_test_filtered = X_test[X_test["Date"] >= "2020-01-01"]

# Sort filtered test set by Date
X_test_sorted = X_test_filtered.sort_values(by="Date")
y_test_sorted = y_test.loc[X_test_sorted.index]
y_pred_sorted = pd.Series(y_pred, index=X_test.index).loc[X_test_sorted.index]

# Plot Actual vs Predicted Prices (2020 Onwards)
plt.figure(figsize=(12, 6))
plt.plot(X_test_sorted["Date"], y_test_sorted, label="Actual Prices", color="blue", alpha=0.6)
plt.plot(X_test_sorted["Date"], y_pred_sorted, label="Predicted Prices", color="red", linestyle="dashed", alpha=0.3)
plt.xlabel("Date")
plt.ylabel("Stock Price (Close)")
plt.title("Actual vs Predicted Stock Prices (2020 Onwards)")
plt.legend()
plt.show()

# Scatter Plot: Actual vs Predicted Stock Prices
plt.figure(figsize=(8, 8))
plt.scatter(y_test_sorted, y_pred_sorted, alpha=0.6, color="purple")
plt.xlabel("Actual Stock Price")
plt.ylabel("Predicted Stock Price")
plt.title("Actual vs Predicted Stock Prices")
plt.axline((0, 0), slope=1, color="black", linestyle="dashed")  # Perfect prediction line
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Stock Price Prediction")
plt.show()

errors = y_test - y_pred  # Compute residuals

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, errors, alpha=0.5, color="red")
plt.axhline(y=0, color="black", linestyle="dashed")
plt.xlabel("Predicted Stock Price")
plt.ylabel("Residual (Error)")
plt.title("Residual Plot")
plt.show()


# Compute moving averages
merged_data["Close_MA"] = merged_data["Close"].rolling(window=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(X_test["Date"], y_test, label="Actual", color="blue")
plt.plot(X_test["Date"], y_pred, label="Predicted", color="red", linestyle="dashed")
plt.plot(X_test["Date"], merged_data.loc[X_test.index, "Close_MA"], label="10-Day Moving Avg", color="green")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction vs Moving Average")
plt.legend()
plt.show()
