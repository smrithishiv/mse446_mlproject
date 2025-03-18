import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

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
voters["Years"] = voters["Years"].astype(int)
merged_data = pd.merge(merged_data, voters, left_on="Year", right_on="Years", how="left")

# Drop unnecessary columns
merged_data.drop(columns=["Election_Date", "Inaugration_Date", "End_of_Term", "Years"], inplace=True)

# Define features list
features = [
    "Electoral_Vote_Winner", "Popular_Vote_Margin", "Election_Year_Inflation_Rate",
    "Election_Year_Interest_Rate", "Election_Year_Unemployment_Rate", "Total voted",
    "Percent voted", "Open", "High", "Low", "Volume"
]

# Keep only columns that have at least one non-null value
valid_features = [col for col in features if merged_data[col].notna().sum() > 0]

# Ensure target variable (y) exists and drop NaNs in "Close"
merged_data = merged_data.dropna(subset=["Close"])
y = merged_data["Close"]

# Impute missing values in valid features
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(merged_data[valid_features])

# Convert back to DataFrame after imputation
X = pd.DataFrame(X_imputed, columns=valid_features)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for NaNs before training
print("Any NaN in X_train:", np.isnan(X_train).any())
print("Any NaN in y_train:", np.isnan(y_train).any())

# Train the SVM Model
svm_model = SVR(kernel="rbf")
svm_model.fit(X_train, y_train)

# Predict
y_pred = svm_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test.values, label="Actual Prices", color="blue", alpha=0.6)
plt.plot(range(len(y_test)), y_pred, label="Predicted Prices", color="red", linestyle="dashed", alpha=0.3)
plt.xlabel("Index")
plt.ylabel("Stock Price (Close)")
plt.title("Actual vs Predicted Stock Prices")
plt.legend()
plt.show()

# Scatter Plot: Actual vs Predicted Stock Prices
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color="purple")
plt.xlabel("Actual Stock Price")
plt.ylabel("Predicted Stock Price")
plt.title("Actual vs Predicted Stock Prices")
plt.axline((0, 0), slope=1, color="black", linestyle="dashed")  # Perfect prediction line
plt.show()
