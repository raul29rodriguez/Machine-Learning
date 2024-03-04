'''Exercise 1
Given the recipes_muffins_cupcakes_scones.csv create a scatter plot between PC1 and PC2 (as
shown in slide 96). Print the variance ratio and plot the cumulative sum of the variance ration.
In addition, plot a heatmap with the features with the largest variation in PC1 and PC2 (as shown
in slide 99). Furthermore, find the two features with the highest variation in PC1 and PC2 as well as the
features with the lowest. Finally, plot a covariance heatmap (as shown in slide 98)'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pandas import DataFrame
df=pd.read_csv('recipes_muffins_cupcakes_scones.csv')
features=df.keys()
features=features.drop('Type')
x=np.array(df.loc[:,'Flour':'Salt'])
y=np.array(df['Type'])
fig, ax=plt.subplots(nrows=1,ncols=2)
x=StandardScaler().fit_transform(x)
pca=PCA(n_components=2)
principalComponents=pca.fit_transform(x)
explained_variance=pca.explained_variance_ratio_
print(f'variance ratio: {explained_variance}')
for i in range(len(df)):
    if i<=9:
        ax[0].scatter(principalComponents[0:i,0],principalComponents[0:i,1],c='y')
    elif i>9 and i<=19:
        ax[0].scatter(principalComponents[10:i,0],principalComponents[10:i,1],c='g')
    else:
        ax[0].scatter(principalComponents[20:i,0],principalComponents[20:i,1],c='b')
ax[1].plot(np.cumsum(explained_variance))
plt.show()
df_comp=pd.DataFrame(pca.components_)
sns.heatmap(df_comp,cmap='plasma')
plt.show()
pca1=abs(df_comp.iloc[0])
pca2=abs(df_comp.iloc[1])
max1=pca1.idxmax()
min1=pca1.idxmin()
max2=pca2.idxmax()
min2=pca2.idxmin()
print(f'max var pca1: {features[max1]} min var pca1: {features[min1]}\nmax var pca2: {features[max2]} min var pca2: {features[min2]}')
cov_mat=np.cov(x.T)
sns.heatmap(cov_mat)
plt.show()
