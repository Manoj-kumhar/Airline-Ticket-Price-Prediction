# Airline-Ticket-Price-Prediction

This script performs the following tasks:
Data Loading and Cleaning: Loads an Excel dataset containing airline ticket prices and performs basic data cleaning by dropping null values.
Feature Engineering: Extracts features like day, month, hour, and minute from date columns and categorizes departure times.
Data Preprocessing: Preprocesses the 'Duration' column to convert it into hours and minutes, and preprocesses other categorical columns for model training.
Data Visualization: Visualizes the distribution of prices, destinations, and airlines using various plots like boxplots, pie charts, and bar graphs.
Model Training: Uses a Random Forest Regressor to train a model for predicting ticket prices.
Model Evaluation: Evaluate the trained model using metrics like R-squared score, Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
Hyperparameter Tuning: Performs hyperparameter tuning using RandomizedSearchCV to improve the model's performance.

The Random Forest Regressor model achieved a reasonable R-squared score on the test data, indicating its effectiveness in predicting ticket prices.
Hyperparameter tuning further improved the model's performance, suggesting that fine-tuning the model can lead to better predictions.
Further analysis and optimization could be done to improve the model's performance even more, such as trying different algorithms, feature engineering techniques, or ensembling methods.
