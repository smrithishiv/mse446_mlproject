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
elections["End_of_Term"] = pd.to_datetime(elections["End_of_Term"], errors="coerce", utc=True)  # Ensure End_of_Term is a datetime
stocks["Date"] = pd.to_datetime(stocks["Date"], errors="coerce", utc=True)

# Extract year from stock dates for merging
stocks["Year"] = stocks["Date"].dt.year

# Extract election year from election date for merging
elections["Election_Year"] = elections["Election_Date"].dt.year

# Merge stock data with elections (Only include elections where stock dates fall between Election_Date and End_of_Term)
def is_within_term(stock_date, election_date, end_of_term):
    return election_date <= stock_date <= end_of_term

# Merge based on the date condition
merged_data = pd.merge(stocks, elections, how="left", left_on="Year", right_on="Election_Year")

# Filter rows where the stock date is within the election term (Election_Date <= Stock Date <= End_of_Term)
merged_data = merged_data[merged_data.apply(
    lambda row: is_within_term(row["Date"], row["Election_Date"], row["End_of_Term"]), axis=1)]

# Forward-fill missing election data if necessary
merged_data = merged_data.sort_values(by="Date").ffill()

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
columns_to_drop = ["Inaugration_Date", "Election_Year", "Years"]
merged_data.drop(columns=[col for col in columns_to_drop if col in merged_data.columns], errors="ignore", inplace=True)

# Fix missing 'Capital Gains' by ensuring it’s numeric and filling NaNs
if "Capital Gains" in merged_data.columns:
    if merged_data["Capital Gains"].dtype == object:  # Convert string to numeric if necessary
        merged_data["Capital Gains"] = merged_data["Capital Gains"].astype(str).str.replace(",", "").astype(float)
    merged_data["Capital Gains"] = merged_data["Capital Gains"].fillna(0)  # Fill missing values with 0

# Final check for null values
null_counts = merged_data.isnull().sum()[merged_data.isnull().sum() > 0]
if null_counts.empty:
    print("No null values remain in merged_data")
else:
    print("Null values still exist:", null_counts)

merged_data.to_csv("data/merged_data.csv", index=False)