import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    print("⚠Null values still exist:", null_counts)

# Select features (Including election-related ones)
features = ["Electoral_Vote_Winner", "Popular_Vote_Margin", "Election_Year_Inflation_Rate",
            "Election_Year_Interest_Rate", "Election_Year_Unemployment_Rate", "Total voted",
            "Percent voted", "Open", "High", "Low", "Volume"]

# Ensure categorical encoding for political parties
merged_data = pd.get_dummies(merged_data, columns=["Party", "Opponent_Party"], drop_first=True)

# Feature Scaling (Apply MinMaxScaler to election-related features)
scaler = MinMaxScaler()
election_features = ["Electoral_Vote_Winner", "Popular_Vote_Margin", "Election_Year_Inflation_Rate",
                     "Election_Year_Interest_Rate", "Election_Year_Unemployment_Rate", "Total voted",
                     "Percent voted"]

merged_data[election_features] = scaler.fit_transform(merged_data[election_features])