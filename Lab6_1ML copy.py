'''Exercise 1
Given the data set auto-mpg, drop the last two columns of the data set, that is, origin and
car name and drop all rows that contain a ?. Estimate the coefficients and intercept using the
equation from slides 148-149. You can use the built-in functions for comparison only'''
import pandas as pd
import numpy as np

df = pd.read_csv('auto-mpg.csv')
df=df.replace('?',np.nan)
df=df.dropna()
df=df.drop("origin",axis=1)
df=df.drop("car name",axis=1)
df=df.astype(float)
#print(df)
#print(df.dtypes)
y=np.array(df['mpg'])
x=np.array(df.loc[:,'cylinders':'model year'])
#print(y)
n=len(x)
x1=np.ones((n),dtype=int).reshape(n,1)
xArr=np.hstack((x1,x))
yArr=np.array(y).reshape(n,1)
xArrT=xArr.transpose()
#print(xArrT)
yArrT=yArr.transpose()
xMul=np.matmul(xArrT,xArr)
#print(xMul)
xyMul=np.matmul(xArrT,yArr)
xMulInv=np.linalg.inv(xMul)
A=np.matmul(xMulInv,xyMul)
#print(A)
intercept=A[0]
print(f'coefficients= {A[1:7]}, intercept= {intercept}')
