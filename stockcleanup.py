import pandas as pd 

df = pd.read_csv('data/World-Stock-Prices-Dataset.csv')

df_usa = df[df["Country"] == "usa"]

df_usa.to_csv("data/USA_Stock_Prices.csv", index=False)
