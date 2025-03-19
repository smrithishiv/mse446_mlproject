import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Load the merged data (from the previous step)
merged_data = pd.read_csv("data/merged_data.csv")
merged_data = merged_data.dropna(subset=["Close", "Party", "Industry_Tag", 
                                         "Election_Year_Unemployment_Rate", "Election_Year_Inflation_Rate", 
                                         "Election_Year_Interest_Rate"])


mean_close = merged_data['Close'].mean()
std_close = merged_data['Close'].std()

# Define the upper and lower bounds for 3-sigma
upper_bound = mean_close + 3 * std_close
lower_bound = mean_close - 3 * std_close
# Filter out outliers
cleaned_data = merged_data[(merged_data['Close'] >= lower_bound) & (merged_data['Close'] <= upper_bound)]


feature_columns = ['Close', 'Election_Year_Inflation_Rate', 'Election_Year_Interest_Rate', 
                   'Election_Year_Unemployment_Rate', 'Party', 'Industry_Tag']

# Select relevant features (X) and target variable (y)
X = merged_data[feature_columns]
y = merged_data['Close']  # Target: Stock closing price

# Preprocess categorical and numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), ['Election_Year_Unemployment_Rate', 'Election_Year_Inflation_Rate', 'Election_Year_Interest_Rate']),
        ('cat', OneHotEncoder(), ['Party', 'Industry_Tag'])
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and KNN regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values (if any)
    ('knn', KNeighborsRegressor())
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'knn__n_neighbors': [1200,500,1000],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'chebyshev'],
    'knn__p': [1, 2]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search model
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Make predictions using the best model
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

random_indices = np.random.choice(len(y_test), size=5, replace=False)

# Get the actual and predicted values for those random indices
random_actual = y_test.iloc[random_indices]
random_predicted = y_pred[random_indices]

# Display the comparison
print("Random Predictions vs Actuals:")
for i in range(len(random_indices)):
    print(f"Actual: {random_actual.iloc[i]:.2f}, Predicted: {random_predicted[i]:.2f}")

# Calculate the Mean Squared Error
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display the metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

future_data = pd.DataFrame({
    'Election_Year_Inflation_Rate': [3.0],  # Example inflation rate
    'Election_Year_Interest_Rate': [5.0],   # Example interest rate
    'Election_Year_Unemployment_Rate': [4.5],  # Example unemployment rate
    'Party': ['R'],  # Example party ('D' for Democrat, 'R' for Republican)
    'Industry_Tag': ['apparel']   # Example industry
})

# Ensure the same transformations are applied to future data
future_prediction = grid_search.best_estimator_.predict(future_data)

print(f"Predicted Future Stock Price: ${future_prediction[0]:.2f}")
