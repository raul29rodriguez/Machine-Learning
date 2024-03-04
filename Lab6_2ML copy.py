'''Exercise 2
Using the same data set from Ex. 1, after having dropped the necessary rows and columns,
scale your data using Standardization. Find the three most important weighted coefficients'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

df = pd.read_csv('auto-mpg.csv')
df=df.replace('?',np.nan)
df=df.dropna()
df=df.drop("origin",axis=1)
df=df.drop("car name",axis=1)
df=df.astype(float)
print(df)
y=np.array(df['mpg'])
x=np.array(df.loc[:,'cylinders':'model year'])
xScaled = StandardScaler().fit_transform(x)
#print(xScaled)
li=[]
for i in range(6):
    slope, intercept, r, p, std_error = stats.linregress(xScaled[:,i], y)
    li.append(r)
li=np.array(li)
li=abs(li)
li=list(li)
print(li)
sortedLi=sorted(li,reverse=True)
#print(type(sortedLi))
topThree=sortedLi[0:3]
#print(twoAndThree)
index1=li.index(topThree[0])
index2=li.index(topThree[1])
index3=li.index(topThree[2])
print(index1,index2,index3)
print(f'most important features are: {df.columns[index1+1]}, {df.columns[index2+1]}, {df.columns[index3+1]}')