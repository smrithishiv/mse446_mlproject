import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Election Data Organization
stock_df = pd.read_csv("data/USA_Stock_Prices.csv")

stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce", utc=True)
stock_df["Date"] = stock_df["Date"].dt.tz_convert(None)

stock_df["Year"] = stock_df["Date"].dt.year

# Stock Data Organization
election_df = pd.read_csv("data/us_presidential_elections_2000_2024.csv")

merged_df = stock_df.groupby("Year").agg(
    start_price=("Close", "first"),
    end_price=("Close", "last")
).reset_index()

merged_df["stock_return"] = ((merged_df["end_price"] - merged_df["start_price"]) / merged_df["start_price"]) * 100


# Election Data and Stock Data Merge
merged_df = merged_df.merge(election_df, on="Year", how="left")

# Mark Election Year (1 if an election occurred that year, else 0)
merged_df["election_year"] = merged_df["Party"].notna().astype(int)

# Ensure Party names are properly mapped
merged_df["Party"] = merged_df["Party"].replace({"D": "Democrat", "R": "Republican"})
sns.set_style("whitegrid")

### 1. Boxplot: Stock returns in election vs. non-election years
plt.figure(figsize=(8, 5))
sns.boxplot(x="election_year", y="stock_return", hue="election_year", data=merged_df, dodge=False, legend=False, palette={0: "blue", 1: "red"})
plt.xticks([0, 1], ["Non-Election Year", "Election Year"])
plt.xlabel("Year Type")
plt.ylabel("Stock Market Return (%)")
plt.title("Stock Returns: Election vs. Non-Election Years")
plt.show()

### 2. Bar Chart: Market Performance under Each Party
plt.figure(figsize=(8, 5))
sns.barplot(x="Party", y="stock_return", hue="Party", data=merged_df, errorbar=None, legend=False, palette={"Democrat": "blue", "Republican": "red"})
plt.xlabel("Political Party")
plt.ylabel("Average Stock Market Return (%)")
plt.title("Stock Market Performance Under Different Political Parties")
plt.show()