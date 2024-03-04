'''Exercise 2
Go to the following Machine Learning repository: ML Repository, click on Data Folder,
and download the files: breast-cancer-wisconsin.data and breast-cancer-wisconsin.names. Rename the
former to: breast-cancer-wisconsin.data.csv. At the end of the latter, you can see the description of
all the attributes.
Remove all rows that contain a ? (this should reduce the number of rows from 699 to 683).
Ignore the first column (ID number) and assign the remaining columns to X (features) and
Y (label). Perform classification using the KNN supervised learning algorithm with K=5 and
test_size=0.30. Print the Confusion Matrix'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
names = ['Sample code number', 'Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion',
'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class' ]
df = pd.read_csv('breast-cancer-wisconsin.data.csv', names=names)
df=df.replace('?',np.nan)
df=df.dropna()
df=df.drop("Sample code number",axis=1)
X = np.array(df.loc[:, 'Clump Thickness' : 'Mitoses'])
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print('Model accuracy score: ', accuracy_score(y_test, pred))
print('Index\tPredicted\tActual')
for i in range(len(pred)):
    if pred[i] != y_test[i]:
        print(i, '\t', pred[i], '\t\t', y_test[i], '***')
print(f'\nConfusion Matrix: \n{confusion_matrix(y_test, pred)}')
#print(df)