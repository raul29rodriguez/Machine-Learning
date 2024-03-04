'''Exercise 2
Given the following data set: hsbdemo.csv, apply the Principal Component Analysis (PCA)
algorithm and print the variance ratio and plot the cumulative sum of the variance ratio for all
10 features. Use the same columns for x and y as in Ex. 1'''
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df=pd.read_csv('hsbdemo.csv')
y=np.array(df['prog'])
df=df.drop(columns=['id','prog','cid'])
df['gender'].replace(['female','male'],[0,1],inplace=True)
df['ses'].replace(['low','middle','high'],[0,1,2],inplace=True)
df['schtyp'].replace(['public','private'],[0,1],inplace=True)
df['honors'].replace(['not enrolled','enrolled'],[0,1],inplace=True)
x=np.array(df[:])
x=StandardScaler().fit_transform(x)
pca=PCA(n_components=2)
principalComponents=pca.fit_transform(x)
explained_variance=pca.explained_variance_ratio_
print(explained_variance)
cum_sum=np.cumsum(explained_variance)
plt.plot(cum_sum)
plt.show()