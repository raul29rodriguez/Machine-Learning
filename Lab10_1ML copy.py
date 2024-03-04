import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

df = pd.read_csv('speedLimits.csv')
X = np.array(df.Speed)
y = np.array(df.Ticket)
X = X.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=0)
modelSVC = SVC(kernel='linear').fit(X_train, y_train)
predL = modelSVC.predict(X_test)
print(f'Accuracy score using linear kernel: {accuracy_score(y_test, predL)}')
modelSVC = SVC(kernel='poly').fit(X_train, y_train)
predP = modelSVC.predict(X_test)
print(f'Accuracy score using poly kernel: {accuracy_score(y_test, predP)}')
modelSVC = SVC(kernel='rbf').fit(X_train, y_train)
predRBF = modelSVC.predict(X_test)
print(f'Accuracy score using rbf kernel: {accuracy_score(y_test, predRBF)}')
modelSVC = SVC(kernel='sigmoid').fit(X_train, y_train)
predS = modelSVC.predict(X_test)
print(f'Accuracy score using sigmoid kernel: {accuracy_score(y_test, predS)}')
print(f'RBF is the best kernel')
for i in range(len(X)):
    if y[i]=="T":
        plt.scatter(X[i],y[i],c='r')
    else:
        plt.scatter(X[i],y[i],c='g')
plt.show()