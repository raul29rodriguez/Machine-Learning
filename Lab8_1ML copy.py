'''Exercise 1
Given the data sets fashion-mnist_train.csv and fashion-mnist_test.csv, use logistic regression to
estimate the accuracy score of the model and print the Confusion Matrix both in text and as a
heatmap'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import seaborn as sns

dfTrain=pd.read_csv('fashion-mnist_train.csv')
dfTest=pd.read_csv('fashion-mnist_test.csv')
xTrain=np.array(dfTrain.iloc[:,1:])
yTrain=np.array(dfTrain.iloc[:,0])
xTest=np.array(dfTest.iloc[:,1:])
yTest=np.array(dfTest.iloc[:,0])
'''print(dfTrain)
print(xTrain)
print(yTrain)
print(dfTest)
print(xTest)
print(yTest)'''
model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(xTrain, yTrain)
pred=model.predict(xTest)
confusionMatrix = confusion_matrix(yTest, pred)
print(confusionMatrix)
print(f'Accuracy score: {accuracy_score(yTest, pred)}')
sns.heatmap(confusionMatrix,annot=True,cmap='plasma')
plt.show()