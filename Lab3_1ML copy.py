'''Exercise 1
Download and rename to wine.data.csv the following dataset: wine
The dataset contains 13 attributes (columns 2-14) that contribute to the quality of wine. The
dataset contains data for three types of wines, identified by the category values, 1, 2, and 3
(column 1). The dataset contains 178 records. Plot a graph of the testing accuracy of the dataset
for different values of K (use K=1-10), using a test size of 20%
Note: The names of all columns are:
names = [’class’, ’Alcohol’,’Malic Acid’,’Ash’,’Acadlinity’,’Magnisium’,’Total Phenols’,’Flavanoids’,
’NonFlavanoid Phenols’, ’Proanthocyanins’, ’Color Intensity’, ’Hue’, ’OD280/OD315’, ’Proline’ ]'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
names = ['class', 'Alcohol','Malic Acid','Ash','Acadlinity','Magnisium','Total Phenols','Flavanoids',
'NonFlavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315', 'Proline' ]
df = pd.read_csv('wine.data.csv', names=names)
#print(df)
X = np.array(df.loc[:, 'Alcohol' : 'Proline'])
y = np.array(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
scores=[]
K_range=range(1,11)
print(K_range)
for K in K_range:
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))
plt.plot(K_range,scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Testing Accuracy")
plt.show()
