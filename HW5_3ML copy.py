'''Given the following data set: materialsOutliers.csv, use RANSAC to detect and remove outliers
between each one of the independent variables and the dependent variable. Remove rows that
contain outliers and perform Multiple Linear Regression
Note 1: You will need to swap x with y and then apply RANSAC. You should also use the fol-
lowing two values in the RANSACRegressor() method: residual_threshold=15, stop_probability=1.00
Note 2: For more info refer to: RANSAC Regressor'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from functools import reduce
from sklearn import linear_model, datasets
df=pd.read_csv('materialsOutliers.csv')
x=np.array(df['Strength'])
x=x.reshape(-1,1)
y1=np.array(df['Time'])
y2=np.array(df['Pressure'])
y3=np.array(df['Temperature'])
# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor(residual_threshold=15,stop_probability=1.00)
ransac.fit(x, y1)
inlier_mask1 = ransac.inlier_mask_
outlier_mask1 = np.logical_not(inlier_mask1)
#print(outlier_mask1)
res=[]
res.append(list(filter(lambda i: outlier_mask1[i],range(len(outlier_mask1)))))
ransac.fit(x, y2)
inlier_mask2 = ransac.inlier_mask_
outlier_mask2 = np.logical_not(inlier_mask2)
res.append(list(filter(lambda i: outlier_mask2[i],range(len(outlier_mask2)))))
ransac.fit(x, y3)
inlier_mask3 = ransac.inlier_mask_
outlier_mask3 = np.logical_not(inlier_mask3)
res.append(list(filter(lambda i: outlier_mask3[i],range(len(outlier_mask3)))))
resFlat= reduce(lambda a,b: a+b, res)
df=df.drop(df.index[resFlat])
#print(df)
y=np.array(df['Strength'])
x=np.array(df.loc[:,'Time':'Temperature'])
reg=LinearRegression()
reg.fit(x,y)
c=np.array(reg.coef_)
print(f'coefficients: {c}')
yIntercept=reg.intercept_
print(f'Intercept: {yIntercept}')