import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

stock_df = pd.read_csv("data/USA_Stock_Prices.csv")
election_df = pd.read_csv("data/us_presidential_elections_2000_2024.csv")

stock_df['Date'] = stock_df['Date'].astype(str)
stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce', utc=True)
stock_df['Year'] = stock_df['Date'].dt.year


## Heatmap: Correlation Between Economic Factors & Stock Prices

# merged_df = election_df[['Year', 'Election_Year_Inflation_Rate', 'Election_Year_Interest_Rate', 'Election_Year_Unemployment_Rate']].merge(
#     stock_df.groupby("Year")["Close"].mean().reset_index(), on="Year")

# corr_matrix = merged_df.drop(columns=["Year"]).corr()

# plt.figure(figsize=(6, 4))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Between Economic Factors & Stock Prices")
# plt.show() 

## Time-Series Plot: Stock market movement 6 months before and after each election

# stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce', utc=True)
# election_df['Election_Date'] = pd.to_datetime(election_df['Election_Date'], errors='coerce', utc=True)
# window = timedelta(days=180)

# plt.figure(figsize=(12, 6))

# for election_date in election_df["Election_Date"].dropna():
#     subset = stock_df[(stock_df["Date"] >= election_date - window) & (stock_df["Date"] <= election_date + window)]
    
#     if not subset.empty:
#         plt.plot(subset["Date"], subset["Close"], label=f"Election {election_date.year}")

# plt.xlabel("Date")
# plt.ylabel("Closing Price")
# plt.title("Stock Market Trends Before & After Elections")
# plt.legend(loc="best", fontsize=8)
# plt.xticks(rotation=45)
# plt.grid(True)

# plt.show()

stock_df['Daily_Return'] = stock_df['Close'].pct_change()
election_df['Election_Date'] = pd.to_datetime(election_df['Election_Date'], errors='coerce', utc=True)
stock_prices = stock_df.merge(election_df, how='left', left_on='Date', right_on='Election_Date')
def get_period_returns(stock_prices, election_date, window=30):
    before = stock_prices[(stock_prices['Date'] < election_date) & 
                          (stock_prices['Date'] >= election_date - pd.Timedelta(days=window))]
    after = stock_prices[(stock_prices['Date'] >= election_date) & 
                         (stock_prices['Date'] <= election_date + pd.Timedelta(days=window))]
    
    return before['Daily_Return'], after['Daily_Return']

plt.figure(figsize=(12, 6))

for i, election in election_df.iterrows():
    before_returns, after_returns = get_period_returns(stock_prices, election['Election_Date'])
    
    plt.hist(before_returns, bins=20, alpha=0.5, label=f'Before Election {election["Year"]}')
    plt.hist(after_returns, bins=20, alpha=0.5, label=f'After Election {election["Year"]}')

plt.title("Distribution of Daily Returns Before vs After Elections")
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.legend()
plt.show()
