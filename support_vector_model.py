import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import random
import matplotlib.pyplot as plt

# Load the merged data
merged_data = pd.read_csv("data/merged_data.csv")

# Drop rows with missing essential data
merged_data = merged_data.dropna(subset=["Close", "Party", "Industry_Tag", 
                                         "Election_Year_Unemployment_Rate", "Election_Year_Inflation_Rate", 
                                         "Election_Year_Interest_Rate"])

# Remove outliers based on 3-sigma rule for stock prices
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

# Create a pipeline with preprocessing and SVR regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('svr', SVR())
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'svr__C': [0.1, 1, 10],
    'svr__epsilon': [0.01, 0.1, 1],
    'svr__kernel': ['rbf', 'linear', 'poly'],
    'svr__degree': [2, 3]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search.fit(X_train, y_train)

# Get the best parameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Make predictions using the best model
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

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
future_prediction = grid_search.best_estimator_.predict(future_data)

print(f"Predicted Stock Change Over Presidential Term: ${future_prediction[0]:.2f}")

# Randomly select 10 actual vs. predicted values for comparison
random_indices = random.sample(range(len(y_test)), 10) 
comparison_df = pd.DataFrame({
    'Actual': y_test.iloc[random_indices].values,
    'Predicted': y_pred[random_indices]
})

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
