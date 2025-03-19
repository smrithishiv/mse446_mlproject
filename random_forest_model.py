import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    print("✅ No null values remain in merged_data!")
else:
    print("⚠ Null values still exist:", null_counts)

# ✅ **Forcing Election Data by Increasing Feature Importance**
election_features = ["Electoral_Vote_Winner", "Popular_Vote_Margin", "Election_Year_Inflation_Rate",
                     "Election_Year_Interest_Rate", "Election_Year_Unemployment_Rate", "Total voted",
                     "Percent voted"]

# **Increase importance by duplicating election features**
for feature in election_features:
    merged_data[f"{feature}_Weighted"] = merged_data[feature] * 10

# **All features (Stock + Election)**
all_features = election_features + ["Open", "High", "Low", "Volume"]

# **Only Election Features**
only_election_features = election_features + [f"{feature}_Weighted" for feature in election_features]

# Ensure categorical encoding for political parties
merged_data = pd.get_dummies(merged_data, columns=["Party", "Opponent_Party"], drop_first=True)

# Feature Scaling (Apply MinMaxScaler to election-related features)
scaler = MinMaxScaler()
merged_data[election_features] = scaler.fit_transform(merged_data[election_features])

# **Train Two Models**
models = {
    "Random Forest (All Features)": all_features,
    "Random Forest (Election-Only)": only_election_features
}

for model_name, selected_features in models.items():
    print(f"\n🔹 Training {model_name}")

    # Define input (X) and target variable (y)
    X = merged_data[selected_features]
    y = merged_data["Close"]  # Predicting stock closing price

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42)  # sqrt forces more feature diversity
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate Model Performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 {model_name} Performance:")
    print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
    print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
    print(f"✅ R² Score: {r2:.2f}")

    # ✅ Feature Importance
    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh([feature_names[i] for i in indices], importances[indices], color="skyblue")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(f"🔍 Feature Importance ({model_name})")
    plt.gca().invert_yaxis()
    plt.show()
