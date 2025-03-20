# MSE 446 - Predicting Stock Market Prices based on Presidential Election Results

This repository contains the data, scripts, and models developed for predicting stock market prices based on U.S. presidential election results. Our project aimed to analyze the impact of political shifts on stock market behavior, leveraging machine learning techniques to create a predictive model.  

**Objectives:**  
- Identify potential correlations between election results and stock price movements.  
- Develop a machine learning model to predict stock price changes during a presidential term.  
- Evaluate the effectiveness of different ML techniques to find the most optimal approach.  

## Repository Structure
 **`data/`**: Contains all the CSV files used in this project. These datasets include:  
  - Raw data sourced from Kaggle and independent research.  
  - Cleaned datasets prepared for model training and testing.  

**`dataset_creation/`**: Includes Python scripts used to preprocess, clean, and transform raw data into structured datasets.  

**`eda/`**: Scripts dedicated to exploratory data analysis, extracting valuable insights from the datasets, and guiding the modeling process.  

**`trial_models/`**: Our experimentation phase, where we tested various machine learning models, including:  
  - Random Forest  
  - K-Means Clustering  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Machines (SVM)  
  This folder showcases our approach to selecting the most suitable model for prediction.  

**`random_forest/`**: Our final model, a Random Forest Regressor enhanced with:  
  - K-Fold Cross Validation for robust model evaluation.  
  - Hyperparameter Tuning to optimize performance.  
  - XGBoost for boosting accuracy.  
  - CSV files comparing predicted vs. actual stock changes and a summary of statistical results.  

