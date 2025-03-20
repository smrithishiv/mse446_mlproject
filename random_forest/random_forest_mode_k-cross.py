import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load the merged data
merged_data = pd.read_csv("data/merged_data.csv")

# Drop rows with missing essential data
merged_data = merged_data.dropna(subset=["Close", "Party", "Industry_Tag", 
                                         "Election_Year_Unemployment_Rate", "Election_Year_Inflation_Rate", 
                                         "Election_Year_Interest_Rate"])

# Remove outliers based on the 3-sigma rule for stock prices
mean_close = merged_data['Close'].mean()
std_close = merged_data['Close'].std()
upper_bound = mean_close + 3 * std_close
lower_bound = mean_close - 3 * std_close
cleaned_data = merged_data[(merged_data['Close'] >= lower_bound) & (merged_data['Close'] <= upper_bound)]

# Compute the average stock price for each industry during each presidential term
term_avg_prices = cleaned_data.groupby(['Industry_Tag', 'Election_Date']).agg(
    avg_term_close=('Close', 'mean')
).reset_index()

# Merge back to get pre-election stock price & compute stock change during the term
cleaned_data = cleaned_data.merge(term_avg_prices, on=['Industry_Tag', 'Election_Date'], how='left')
cleaned_data['Stock_Change_During_Term'] = cleaned_data['avg_term_close'] - cleaned_data['Close']

# Define features (X) and target variable (y)
feature_columns = ['Close', 'Election_Year_Inflation_Rate', 'Election_Year_Interest_Rate', 
                   'Election_Year_Unemployment_Rate', 'Party', 'Industry_Tag']
X = cleaned_data[feature_columns]
y = cleaned_data['Stock_Change_During_Term']  # Target: Stock price change over the term

# Preprocessing: Scale numeric features & OneHotEncode categorical ones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), ['Close', 'Election_Year_Unemployment_Rate', 
                                 'Election_Year_Inflation_Rate', 'Election_Year_Interest_Rate']),
        ('cat', OneHotEncoder(), ['Party', 'Industry_Tag'])
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with preprocessing and RandomForestRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])

# K-Fold Cross Validation (k=3)
kf = KFold(n_splits=3, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2")

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions using the trained model
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"K-Fold Cross Validation Results (k=3):")
print(f"Mean R²: {np.mean(cross_val_scores):.3f} ± {np.std(cross_val_scores):.3f}")
print("\n Final Random Forest Model Performance:")
print(f" Mean Squared Error (MSE): {mse:.2f}")
print(f" Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f" R² Score: {r2:.2f}")

# Predict stock change for a future scenario
future_data = pd.DataFrame({
    'Close': [150.0],  # Example starting stock price
    'Election_Year_Inflation_Rate': [3.0],  # Example inflation rate
    'Election_Year_Interest_Rate': [5.0],   # Example interest rate
    'Election_Year_Unemployment_Rate': [4.5],  # Example unemployment rate
    'Party': ['R'],  # Example party ('D' for Democrat, 'R' for Republican)
    'Industry_Tag': ['apparel']   # Example industry
})

# Apply the same transformations to future data
future_prediction = pipeline.predict(future_data)

print(f"\n Predicted Stock Change Over Presidential Term: ${future_prediction[0]:.2f}")

# Randomly select 10 test samples for comparison
random_indices = random.sample(range(len(y_test)), 10)
comparison_df = pd.DataFrame({
    'Actual': y_test.iloc[random_indices].values,
    'Predicted': y_pred[random_indices]
})
print("\n Random Sample of Actual vs Predicted Values:")
print(comparison_df)

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)

# Add a line for perfect predictions (y = x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

# Labels and title
plt.title('Actual vs. Predicted Stock Change During Presidential Term', fontsize=14)
plt.xlabel('Actual Stock Change', fontsize=12)
plt.ylabel('Predicted Stock Change', fontsize=12)

# Show plot
plt.show()
