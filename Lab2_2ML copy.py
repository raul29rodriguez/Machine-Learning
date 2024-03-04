'''Exercise 2
Given the iris.data.csv, produce the following two plots as shown in the two figures below'''

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.decomposition import PCA
fig, ax=plt.subplots(nrows=2,ncols=1)
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv('iris.data.csv', names=names)
print(df)
x1 = np.array(df['sepal_length'])
y1 = np.array(df['sepal_width'])
x2 = np.array(df['petal_length'])
y2 = np.array(df['petal_width'])
color=list(df['class'])
for i in range(len(color)):
    if color[i]=='Iris-setosa':
        ax[0].scatter(x1[i],y1[i],c='purple')
    elif color[i]=='Iris-versicolor':
        ax[0].scatter(x1[i],y1[i],c='green')
    elif color[i]=='Iris-virginica':
        ax[0].scatter(x1[i],y1[i],c='yellow')

for i in range(len(color)):
    if color[i]=='Iris-setosa':
        ax[1].scatter(x2[i],y2[i],c='purple')
    elif color[i]=='Iris-versicolor':
        ax[1].scatter(x2[i],y2[i],c='green')
    elif color[i]=='Iris-virginica':
        ax[1].scatter(x2[i],y2[i],c='yellow')
ax[0].set_xlabel('Sepal Length')
ax[0].set_ylabel('Sepal Width')
ax[1].set_xlabel('Petal Length')
ax[1].set_ylabel('Petal Width')
plt.show()