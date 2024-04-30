# Predicting Loan Default 

We build a classifier to predict default for a given loan. We use financial and loan application data of borrowers as features to help predict default. We also analyze feature importance to describe which financial metrics are most important for predicting loan default.  

## Data

We use loan data from Kaggle that was obtained by Lending Club from 2007-2018, which can be found [here]([https://www.kaggle.com/datasets/wordsforthewise/lending-club/data]). 

The dataset contains 150 explanatory variables and over 2 million rows for which we will use feature selection. 

## Libraries and Tools

- Polars for data handling and cleaning. 
- Optuna for model calibration and hyperparameter tuning. 
- Seaborn for data visualization. 
- Sklearn for machine learning models. 

## Methods

We make use of the fast data handling package Polars to clean and process the data. Having removed missing values, removed useless features, modified and encoded categorical variables and scaled numerical features, we then perform cross-validation on several models and choose the best three to tune parameters. Once parameters have been tuned using Optuna, we evaluate the performance of the models on the unseen test set using accuracy, recall and ROC score as our metrics. We also perform a feature importance analysis on our highest performing models. 

## Summary of Notebooks 

- [1-Cleaning](/notebooks/1-Cleaning.ipynb): Import the raw data. Remove missing values. Perform feature elimination. 
- [2-EDA](/notebooks/2-EDA.ipynb): Perform feature selection based on correlation with default variable. Plot categorical variables and select features with default impact. Log-transform skewed numerical variables. Label and one hot encode categorical variables. Train-test split data and scale numerical features. 
- [3-Model-Selection](/notebooks/3-Model-Selection.ipynb): Perform 5-fold cross validation on eight models with default parameters. Analyse accuracy of models on training data. Select top three performing models for hyperparameter tuning with Optuna. Evaluate model performance on the training set. 
- [4-Evaluation](/notebooks/4-Evaluation.ipynb): Generate predictions for the top three performing models on the test set. Evaluate performance of each model using accuracy, ROC score and recall. Investigate feature importance of the models. Conclude with findings. 

## Models 

We initially use the following models with default parameters during the model selection phase: 

- Logistic Regression 
- K Nearest Neighbors 
- Decision Tree Classifier
- Gaussian Naive Bayes Classifier
- Random Forest Classifier
- Gradient Boosted Classifier
- Multi-layer Perceptron Classifier

Of these models, the following performed best and were chosen for hyperparameter tuning: 

- Random Forest Classifier
- Gradient Boosted Classifier
- Multi-layer Perceptron

The success of these models suggests a highly non-linear relationship between the features and default. 

