'''Exercise 1
Given the following data set: hsbdemo.csv, perform classification using the KNN supervised
learning algorithm with K=5 and test_size=0.10. Print the accuracy of the model as well as the
Confusion Matrix. For features (i.e., predictor variables) use all columns apart from columns: id,
prog, cid and for target (i.e., y or response variable) use column prog
Note 1: All features (i.e., predictor variables) will have to be converted from categorical to
numeric, e.g., convert the values of honors column to: 0, 1
Note 2: You may wish to use in function train_test_split() as a last parameter the: ran-
dom_state=0 to select, in every run, the same random samples'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv('hsbdemo.csv')
y=np.array(df['prog'])
#print(y)
#print(df)
df=df.drop(columns=['id','prog','cid'])
#print(df)
df['gender'].replace(['female','male'],[0,1],inplace=True)
#print(df)
df['ses'].replace(['low','middle','high'],[0,1,2],inplace=True)
df['schtyp'].replace(['public','private'],[0,1],inplace=True)
df['honors'].replace(['not enrolled','enrolled'],[0,1],inplace=True)
x=np.array(df[:])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10,random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print('Model accuracy score: ', accuracy_score(y_test, pred))
print(f'\nConfusion Matrix: \n{confusion_matrix(y_test, pred)}')