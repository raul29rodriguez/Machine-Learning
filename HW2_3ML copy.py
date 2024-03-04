'''Exercise 3
Go to the following Machine Learning repository: ML Repository, click on Data Folder, and
download the files: wdbc.data and wdbc.names. Rename the former to: wdbc.data.csv. In the latter
you can read the descriptions of all the attributes.
Ignore the first column (ID number), assign the second column to y and the rest of the
columns to X. Scale your data using the Standardization method (you can use built-in functions),
and perform Principal Component Analysis with 2 Principal Components. Plot the 2 Principal
Components and print the variance ratio as shown in the Figure below'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pandas import DataFrame

names=['ID number','radius','texture','perimeter','area','smoothness','compactness','concavity','concave points',
'symmetry','fractal dimension']
df=pd.read_csv('wdbc.data.csv',names=names)
df=df.drop(df.columns[0],axis=1)
x = np.array(df.loc[:, 'texture' : 'fractal dimension'])
y = np.array(df['radius'])
x=StandardScaler().fit_transform(x)
pca=PCA(n_components=2)
principalComponents=pca.fit_transform(x)
explained_variance=pca.explained_variance_ratio_
df_comp=pd.DataFrame(pca.components_)
pc1=principalComponents[:,0]
pc2=principalComponents[:,1]
plt.scatter(principalComponents[:,0],principalComponents[:,1],c='g')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.title(f'PCA=2 Variance: {explained_variance}')
plt.show()