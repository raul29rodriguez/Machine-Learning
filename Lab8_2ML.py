# -*- coding: utf-8 -*-
"""Lab8_2ML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y9NbwWSIpSYm8RbIP_5xGCwDiYr4Y2H1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, ConfusionMatrixDisplay
#uncomment if using google colab:
from google.colab.patches import cv2_imshow
import seaborn as sns

dfTest=pd.read_csv('/content/fashion-mnist_test.csv')
dfTrain=pd.read_csv('/content/fashion-mnist_train.csv')
xTrain=np.array(dfTrain.iloc[:,1:])
yTrain=np.array(dfTrain.iloc[:,0])
xTest=np.array(dfTest.iloc[:,1:])
yTest=np.array(dfTest.iloc[:,0])
model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(xTrain, yTrain)
number = cv2.cvtColor(cv2.imread('/content/bag.jpg'), cv2.COLOR_BGR2GRAY)
cv2_imshow(number)
number = cv2.resize(number, (28, 28))
number = number.reshape(1, 28 * 28)
number2 = cv2.cvtColor(cv2.imread('/content/trousers.bmp'), cv2.COLOR_BGR2GRAY)
cv2_imshow(number2)
number2 = cv2.resize(number2, (28, 28))
number2 = number2.reshape(1, 28 * 28)
print(f'Predicted input digit for bag:{model.predict(number)}')
print(f'Predicted input digit for trousers:{model.predict(number2)}')