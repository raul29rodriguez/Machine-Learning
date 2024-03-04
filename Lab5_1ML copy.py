'''Exercise 1
From sklearn datasets import the California housing dataset (see link below). Read the 8
features that are the independent, i.e., predictor variables. The last column is the target, i.e.,
response variable. Using Multiple Linear Regression, compute the linear regression coefficients
and y-intercept. Predict the Median House Value given:
8.3153, 41.0, 6.894423, 1.053714, 323.0, 2.533576, 37.88, −122.23
Note 1: California housing dataset
Note 2: Linear Regression method'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas import DataFrame

df=pd.read_csv('california_housing.csv')
df=df.drop(df.columns[0],axis=1)
y=np.array(df.iloc[:,8])
x=np.array(df.iloc[:,:8])
m=np.array([8.3153, 41.0, 6.894423, 1.053714, 323.0, 2.533576, 37.88, -122.23])
reg=LinearRegression()
reg.fit(x,y)
c=np.array(reg.coef_)
print(f'coefficients: {c}')
yIntercept=reg.intercept_
print(f'Intercept: {yIntercept}')
predict=c[0]*m[0]+c[1]*m[1]+c[2]*m[2]+c[3]*m[3]+c[4]*m[4]+c[5]*m[5]+c[6]*m[6]+c[7]*m[7]+yIntercept
print(predict)