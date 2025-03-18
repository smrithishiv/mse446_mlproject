import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

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
voters["Years"] = voters["Years"].fillna(method="ffill").astype(int)  # Fill missing values
merged_data = pd.merge(merged_data, voters, left_on="Year", right_on="Years", how="left")

# Drop unnecessary columns
merged_data.drop(columns=["Election_Date", "Inaugration_Date", "End_of_Term", "Years"], inplace=True)

# Fill missing values in all columns to ensure no dropped data
merged_data.fillna(method="ffill", inplace=True)

# Select features (Including election-related ones)
features = ["Electoral_Vote_Winner", "Popular_Vote_Margin", "Election_Year_Inflation_Rate",
            "Election_Year_Interest_Rate", "Election_Year_Unemployment_Rate", "Total voted",
            "Percent voted", "Open", "High", "Low", "Volume"]

# Ensure categorical encoding for political parties
merged_data = pd.get_dummies(merged_data, columns=["Party", "Opponent_Party"], drop_first=True)

# ✅ Feature Scaling (Apply MinMaxScaler to election-related features)
scaler = MinMaxScaler()
election_features = ["Electoral_Vote_Winner", "Popular_Vote_Margin", "Election_Year_Inflation_Rate",
                     "Election_Year_Interest_Rate", "Election_Year_Unemployment_Rate", "Total voted",
                     "Percent voted"]

merged_data[election_features] = scaler.fit_transform(merged_data[election_features])

# Define input (X) and target variable (y)
X = merged_data[features]
y = merged_data["Close"]  # Predicting stock closing price

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train XGBoost Model (Better for mixed features)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (XGBoost): {mae:.2f}")

# ✅ Feature Importance (After Scaling)
importances = xgb_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in indices], importances[indices], color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Stock Price Prediction (XGBoost)")
plt.gca().invert_yaxis()  # Show most important first
plt.show()
