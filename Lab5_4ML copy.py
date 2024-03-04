'''Exercise 4
After dropping the Longitude and Latitude features, create a pairplot between the 6 remain-
ing features of the data set'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('california_housing.csv')
df=df.drop(df.columns[0],axis=1)
df=df.drop(['Longitude','Latitude'],axis=1)
df=df.iloc[::20,:]
print(df)
x=np.array(df.loc[:,:'AveOccup'])
print(x)
sns.pairplot(data=df,vars=df.loc[:,:'AveOccup'],hue='MedHouseVal')
plt.show()