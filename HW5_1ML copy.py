'''Given the following data set: materials.csv, scale your data by using Standardization and find
the most important features by finding the correlation coefficient, r, between your response
variable, that is, Strength, and each one of the predictor variables (you should use absolute
values). Using Multiple Linear Regression, perform prediction for the following two data points
for Time, Pressure, Temperature, respectively:
32.1, 37.5, 128.95
36.9, 35.37, 130.03
Note: For prediction, do not use any built-in functions. Yet, do not hard-code the coefficients
(you can use, for example, a loop). In addition, do not use the scaled data set for prediction'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
df=pd.read_csv('materials.csv')
#print(df)
y=np.array(df['Strength'])
x=np.array(df.loc[:,'Time':'Temperature'])
#print(x)
xScaled = StandardScaler().fit_transform(x)
#print(xScaled)
li=[]
for i in range(3):
    slope, intercept, r, p, std_error = stats.linregress(xScaled[:,i], y)
    #print(xScaled[:,i])
    li.append(r)
li=np.array(li)
li=abs(li)
print(li)
max=li.argmax()
print(f'most important feature is: {df.columns[max+1]}') #+1 because index values include the response variable 
reg=LinearRegression()
reg.fit(x,y)
c=np.array(reg.coef_)
print(f'coefficients: {c}')
yIntercept=reg.intercept_
print(f'Intercept: {yIntercept}')
p1=np.array([32.1, 37.5, 128.95])
p2=np.array([36.9, 35.37, 130.03])
predict1=0
predict2=0
for i in range(len(p1)):
    predict1+=c[i]*p1[i]
    predict2+=c[i]*p2[i]
predict1+=yIntercept
predict2+=yIntercept
print(f'predictions with given data are\n{predict1}\n{predict2}')