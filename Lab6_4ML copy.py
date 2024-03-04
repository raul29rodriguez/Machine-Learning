'''Exercise 4
Predict the mpg given: 8, 306, 129, 3508, 11, 70'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('auto-mpg.csv')
df=df.replace('?',np.nan)
df=df.dropna()
df=df.drop("origin",axis=1)
df=df.drop("car name",axis=1)
df=df.astype(float)
print(df)
#print(df.dtypes)
y=np.array(df['mpg'])
x=np.array(df.loc[:,'cylinders':'model year'])
p=np.array([8, 306, 129, 3508, 11, 70])

reg=LinearRegression() 
reg.fit(x,y)
c=np.array(reg.coef_)
print(f'coefficients: {c}')
yIntercept=reg.intercept_
print(f'Intercept: {yIntercept}')

predict=0
for i in range(len(p)):
    predict+=c[i]*p[i]

predict+=yIntercept
print(f'prediction with given data are\n{predict}')