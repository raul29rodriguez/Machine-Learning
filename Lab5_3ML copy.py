'''Exercise 3
Find, using a loop, that is, without hard-coding, the most important feature in the data set'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

df=pd.read_csv('california_housing.csv')
df=df.drop(df.columns[0],axis=1)
y=np.array(df.iloc[:,8])
x=np.array(df.iloc[:,:8])
li=[]
for i in range(len(x)):
    if i==8:
        break
    slope, intercept, r, p, std_error = stats.linregress(x[:,i], y)
    li.append(r)
li=np.array(li)
li=abs(li)
max=li.argmax()
print(f'most important feature is: {df.columns[max]}')