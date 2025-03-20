import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import random

# Load the merged dataset
merged_data = pd.read_csv("data/merged_data.csv")

# Drop rows with missing essential data
merged_data = merged_data.dropna(subset=["Close", "Party", "Industry_Tag", 
                                         "Election_Year_Unemployment_Rate", "Election_Year_Inflation_Rate", 
                                         "Election_Year_Interest_Rate"])

# Remove Outliers (3-Sigma Rule for Stock Prices)
mean_close = merged_data['Close'].mean()
std_close = merged_data['Close'].std()
upper_bound = mean_close + 3 * std_close
lower_bound = mean_close - 3 * std_close
cleaned_data = merged_data[(merged_data['Close'] >= lower_bound) & (merged_data['Close'] <= upper_bound)]

# Define Features for Clustering
feature_columns = ['Close', 'Election_Year_Inflation_Rate', 'Election_Year_Interest_Rate', 
                   'Election_Year_Unemployment_Rate', 'Party', 'Industry_Tag']
X = cleaned_data[feature_columns]

# Preprocessing Pipeline: Scaling & Encoding (Forces Dense Output)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), ['Close', 'Election_Year_Unemployment_Rate', 
                                 'Election_Year_Inflation_Rate', 'Election_Year_Interest_Rate']),
        ('cat', OneHotEncoder(sparse_output=False), ['Party', 'Industry_Tag'])
    ]
)

# Apply Preprocessing
X_processed = preprocessor.fit_transform(X)

# Determine Optimal Clusters Using Elbow Method
model = KMeans(init="k-means++", random_state=42)
visualizer = KElbowVisualizer(model, k=(2, 10)) 
visualizer.fit(X_processed)  
visualizer.show()

# Select Optimal Clusters
optimal_clusters = visualizer.elbow_value_

# Train K-Means Model
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, init="k-means++")
cluster_labels = kmeans.fit_predict(X_processed)

# Add Cluster Labels to Data
cleaned_data["Cluster"] = cluster_labels

# Apply PCA for Visualization (Reducing to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)
cleaned_data["PCA1"] = X_pca[:, 0]
cleaned_data["PCA2"] = X_pca[:, 1]

# Plot Clusters
plt.figure(figsize=(10, 6))
plt.scatter(cleaned_data["PCA1"], cleaned_data["PCA2"], c=cleaned_data["Cluster"], cmap="viridis", alpha=0.6)
plt.colorbar(label="Cluster Label")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Stock Market Clustering Based on Elections")
plt.show()

# Analyze Cluster Results
print("Cluster Counts:")
print(cleaned_data["Cluster"].value_counts())

# Show Average Stock Prices by Cluster
cluster_analysis = cleaned_data.groupby("Cluster")["Close"].mean()
print("\n💰 Average Stock Prices by Cluster:")
print(cluster_analysis)

# Compute Average Stock Change for Each Cluster
cluster_stock_change = cleaned_data.groupby("Cluster")["Stock_Change_During_Term"].mean()

# Assign Predicted Stock Change to Each Data Point Based on Cluster
cleaned_data["Predicted_Stock_Change"] = cleaned_data["Cluster"].map(cluster_stock_change)

# Select Random Data Points for Comparison
random_indices = random.sample(range(len(cleaned_data)), min(300, len(cleaned_data)))

comparison_df = pd.DataFrame({
    'Actual': cleaned_data.iloc[random_indices]["Stock_Change_During_Term"].values,
    'Predicted': cleaned_data.iloc[random_indices]["Predicted_Stock_Change"].values
})

print(comparison_df.head())

# Scatter Plot of Actual vs Predicted Stock Change
plt.figure(figsize=(8, 6))
plt.scatter(cleaned_data["Stock_Change_During_Term"], cleaned_data["Predicted_Stock_Change"], color='blue', alpha=0.6)

# Add a diagonal line (perfect predictions)
plt.plot([cleaned_data["Stock_Change_During_Term"].min(), cleaned_data["Stock_Change_During_Term"].max()],
         [cleaned_data["Stock_Change_During_Term"].min(), cleaned_data["Stock_Change_During_Term"].max()], 'r--', lw=2)

# Labels and title
plt.title('Actual vs. Predicted Stock Change (K-Means Clustering)', fontsize=14)
plt.xlabel('Actual Stock Change', fontsize=12)
plt.ylabel('Predicted Stock Change (Cluster Mean)', fontsize=12)

# Show plot
plt.show()
