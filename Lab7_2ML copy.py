'''Exercise 2
In continuation of Ex. 1, estimate the accuracy of the model and print the Confusion Matrix.
You will have to create your own functions to split the data and compute the Confusion Matrix
Note: Do not use any built-in functions for splitting the data nor any built-in functions to
compute the Confusion Matrix'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import classification_report, confusion_matrix

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

Ypredicted = logReg.predict(X)
confusionMatrix = confusion_matrix(y, Ypredicted)
print(confusionMatrix)
'''for i in range(len(y)):
    print(f'Actual: {y[i]}\tPredicted: {Ypredicted[i]}')
print(classification_report(y, Ypredicted))'''