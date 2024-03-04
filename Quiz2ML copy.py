'''Exercise 1
Given the data set materials.csv, predict the Strength given the values: 33.5, 40.5, 133.2, for
Time, Pressure, Temperature, respectively. The response variable is Strength'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas import DataFrame

df=pd.read_csv('materials.csv')
#print(df)
y=np.array(df['Strength'])
x=np.array(df.loc[:,'Time':'Temperature'])
p=np.array([33.5, 40.5, 133.2])
#print(x)
#print(y)
reg=LinearRegression()
reg.fit(x,y)
c=np.array(reg.coef_)
print(f'coefficients: {c}')
yIntercept=reg.intercept_
print(f'Intercept: {yIntercept}')
predict=c[0]*p[0]+c[1]*p[1]+c[2]*p[2]+yIntercept
print(f'predtion for strenght is {predict}')