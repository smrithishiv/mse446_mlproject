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

## Scatter Plot with Regression: Popular vote margin vs. stock market reaction 
if 'Year' not in election_df.columns:
    election_df['Election_Date'] = pd.to_datetime(election_df['Election_Date'])
    election_df['Year'] = election_df['Election_Date'].dt.year

annual_returns = stock_df.groupby('Year').agg(
    first_price=('Close', 'first'),
    last_price=('Close', 'last')
).reset_index()

annual_returns['Yearly_Return'] = (annual_returns['last_price'] - annual_returns['first_price']) / annual_returns['first_price']

merged_df = election_df[['Year', 'Popular_Vote_Margin']].merge(annual_returns[['Year', 'Yearly_Return']], on="Year", how="inner")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x="Popular_Vote_Margin", y="Yearly_Return", color='blue', label="Data points")

sns.regplot(data=merged_df, x="Popular_Vote_Margin", y="Yearly_Return", scatter=False, color='red', line_kws={'label': 'Regression Line'})

plt.title("Popular Vote Margin vs. Yearly Stock Return by Election Year")
plt.xlabel("Popular Vote Margin")
plt.ylabel("Yearly Stock Return")
plt.legend(title="Legend")
plt.grid(True)

plt.tight_layout()
plt.show()

## Bar Chart comparing market returns for close elections vs. landslide victories
if 'Year' not in election_df.columns:
    election_df['Election_Date'] = pd.to_datetime(election_df['Election_Date'])
    election_df['Year'] = election_df['Election_Date'].dt.year

annual_returns = stock_df.groupby('Year').agg(
    first_price=('Close', 'first'),
    last_price=('Close', 'last')
).reset_index()

annual_returns['Yearly_Return'] = (annual_returns['last_price'] - annual_returns['first_price']) / annual_returns['first_price']

merged_df = election_df[['Year', 'Popular_Vote_Margin']].merge(annual_returns[['Year', 'Yearly_Return']], on="Year", how="inner")

merged_df['Election_Type'] = np.where(merged_df['Popular_Vote_Margin'] < 5, 'Close Election', 'Landslide Victory')

print(merged_df[['Year', 'Popular_Vote_Margin', 'Yearly_Return', 'Election_Type']])

plt.figure(figsize=(10, 6))
sns.barplot(data=merged_df, x='Election_Type', y='Yearly_Return', palette='coolwarm')

plt.title("Market Return Comparison for Close Elections vs. Landslide Victories")
plt.xlabel("Election Type")
plt.ylabel("Yearly Stock Return")
plt.grid(True)

plt.tight_layout()
plt.show()
