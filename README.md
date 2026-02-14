# bank_marketing_model_analysis
The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y)

## Problem statement
The objective of this project is to build and evaluate multiple machine learning models to predict whether a customer will subscribe to a term deposit based on the given Bank Marketing dataset.
This is a binary classification problem where the target variable indicates whether the client has subscribed to a term deposit (Yes/No).
By comparing different models using multiple evaluation metrics, the aim is to identify the model that gives the best performance for this dataset

## Dataset description 

The dataset used in this project is the Bank Marketing dataset, which contains customer information collected during a direct marketing campaign conducted by a banking institution.
Dataset details:
* The dataset includes demographic, financial and campaign-related attributes.
* The target variable is y, which represents whether the customer subscribed to a term deposit.
	Types of features in the dataset:
Demographic features
* Age
* Job
* Marital status
* Education
Financial features
* Balance
* Housing loan
* Personal loan
Campaign-related features
* Contact type
* Duration
* Previous campaign outcome
Data preprocessing steps performed:
* Handling missing values
* Encoding categorical variables
* Feature scaling (where required)
* Splitting the data into training and testing sets


## Models used: 

Make a Comparison Table with the evaluation metrics calculated for all the 6 models as below: (Derivation shown in python notebook as well)

ML ModelAccuracyAUCPrecisionRecallF1MCCLogistic Regression0.8900000.6500000.1700000.2700000.7400000.300000Decision Tree0.8900000.6400000.1900000.3000000.7200000.310000K-Nearest Neighbours0.8800000.4900000.2000000.2800000.6600000.260000Naive Bayes (Gaussian)0.7900000.2800000.4800000.3500000.7300000.250000Random Forest0.8900000.5700000.2100000.3100000.7300000.300000XGBoost0.8900000.6900000.1700000.2800000.7700000.310000
-Observations on the performance of each model on the chosen
dataset. 

ML ModelObservation about model performanceLogistic RegressionHigh accuracy (0.89) but very low recall (0.17) shows strong majority-class bias and poor detection of actual subscribers.Decision TreeSlight improvement in recall (0.19) and F1 (0.30) over Logistic Regression, but lower ROC (0.72) indicates weaker class separability.K-Nearest NeighboursLowest ROC (0.66) and MCC (0.26) suggest poor overall class discrimination for this dataset.Naive Bayes (Gaussian)Highest recall (0.48) but very low precision (0.28) and accuracy (0.79), meaning it detects most subscribers but with many false positives.Random ForestBest F1 score (0.31) with stable MCC (0.30), providing the most balanced precision–recall trade-off.XGBoostHighest ROC (0.77) and precision (0.69) with top MCC (0.31), making it the most reliable overall classifier.
XGBoost performs best overall because it achieves the highest ROC (0.77) and joint-highest MCC (0.31), indicating the strongest class separability and most reliable balanced classification despite the class imbalance.
