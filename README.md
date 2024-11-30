# Hawkeye-cricket-predictor

Submission for The Hakweye job application. 

Criteria were as follows: 
Objective: Build a model to predict the final innings score for a cricket team.

Instructions:

-Parse and analyse innings data (runs, wickets, balls).

-Train a predictive model for final score estimation.

-Measure model performance and explore improvements.

The Dataset was obtained using crichseet (https://cricsheet.org/downloads/)

## The code:
The CricketScorePredictor class leverages the Random Forest Regressor to analyze cricket innings data and provide predictions based on current match progress.

# Features

Innings Data Parsing: Extracts and preprocesses cricket match data from JSON files.

Training Data Preparation: Converts historical match data into features and targets for model training.

Model Training: Trains a Random Forest Regressor with robust train-validation-test splits.

Score Prediction: Provides a predicted final score based on the current innings progress.

Validation and Test Metrics: Evaluates the model with metrics such as MSE, MAE, and R².

# How It Works

Load cricket match data from a JSON file using the load_match_data function.

Parse and process innings data into meaningful statistical features.

Train the model using historical data and evaluate its performance.

Predict the final score of an ongoing inning using live data.

As per the instructions above

# Requirements
Python 3.7 or later

Pandas

NumPy

scikit-learn

# usage

Place the match data file in the project directory.

Run the script using `python3 python.main`

View the model's metrics and the predicted score in the console.

# Reults
Below is a screenshot of the results on the last run before submission

<img width="355" alt="image" src="https://github.com/user-attachments/assets/beff60da-9b4c-4ef2-b4f0-87d43d08dd51">



