'''Exercise 1
Given the data set Student-Pass-Fail.csv, use logistic regression to find the odds, and predict
whether a student will Pass or Fail given the following three data points: [7, 28], [10, 34], [2, 39].
Print the probabilites for each one of the three data points'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

df = pd.read_csv('Student-Pass-Fail.csv')
#print(df)
X = np.array(df.iloc[:, 0:2])
y = np.array(df.iloc[:, 2])
#print(X)
logReg = linear_model.LogisticRegression().fit(X, y)
print(f'Coef: {logReg.coef_}')
print(f'Coef shape: {logReg.coef_.shape}')
print(f'Intercept: {logReg.intercept_}')
log_odds = logReg.coef_
odds = np.exp(log_odds)
print(f'Odds: {odds}\n')
pred = np.array([[7, 28],
                [10, 34],
                [2, 39]])
Ypred = logReg.predict(pred)
print(f'Predicted: {Ypred}')
Ymodel = logReg.intercept_ + logReg.coef_[0, 0]*pred[0, 0] + logReg.coef_[0, 
1]*pred[0, 1]
Ymodel2 = logReg.intercept_ + logReg.coef_[0, 0]*pred[1, 0] + logReg.coef_[0, 
1]*pred[1, 1]
Ymodel3 = logReg.intercept_ + logReg.coef_[0, 0]*pred[2, 0] + logReg.coef_[0, 
1]*pred[2, 1]
probability = np.exp(Ymodel) / (np.exp(Ymodel) + 1)
print(f'Probabilities: {probability}')
probability2 = np.exp(Ymodel2) / (np.exp(Ymodel2) + 1)
print(f'Probabilities: {probability2}')
probability3 = np.exp(Ymodel3) / (np.exp(Ymodel3) + 1)
print(f'Probabilities: {probability3}')