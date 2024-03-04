'''Exercise 3
Given the data set Bank-data.csv, use logistic regression to find the odds, and predict whether
a client will subscribe a term deposit or not given the following two data points: [1.335, 0, 1, 0, 0,
109], [1.25, 0, 0, 1, 0, 279]. Print the probabilites for each one of the two data points
Note: Drop the first column, the target variable is the last column'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

df = pd.read_csv('Bank-data.csv')
#print(df)
df=df.drop(df.columns[0],axis=1)
#print(df)
X = np.array(df.iloc[:, 0:6])
y = np.array(df.iloc[:, 6])
#print(X)
logReg = linear_model.LogisticRegression().fit(X, y)
print(f'Coef: {logReg.coef_}')
print(f'Coef shape: {logReg.coef_.shape}')
print(f'Intercept: {logReg.intercept_}')
log_odds = logReg.coef_
odds = np.exp(log_odds)
print(f'Odds: {odds}\n')
pred = np.array([[1.335, 0, 1, 0, 0,109], 
                 [1.25, 0, 0, 1, 0, 279]])
Ypred = logReg.predict(pred)
print(f'Predicted: {Ypred}')
Ymodel = logReg.intercept_ + logReg.coef_[0, 0]*pred[0, 0] + logReg.coef_[0, 
1]*pred[0, 1]
Ymodel2 = logReg.intercept_ + logReg.coef_[0, 0]*pred[1, 0] + logReg.coef_[0, 
1]*pred[1, 1]
probability = np.exp(Ymodel) / (np.exp(Ymodel) + 1)
print(f'Probabilities: {probability}')
probability2 = np.exp(Ymodel2) / (np.exp(Ymodel2) + 1)
print(f'Probabilities: {probability2}')