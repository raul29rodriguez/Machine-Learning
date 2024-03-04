'''Exercise 3
Using the following data set, Square Feet: 100, 150, 185, 235, 310, 370, 420, 430, 440, 530, 600,
634, 718, 750, 850, 903, 978, 1010, 1050, 1990 and Price ($): 12300, 18150, 20100, 23500, 31005,
359000, 44359, 52000, 53853, 61328, 68000, 72300, 77000, 89379, 93200, 97150, 102750, 115358,
119330, 323989. Find the inliers and outliers in the data set using the RANSAC algorithm, as
shown in slide 120. Print the parameters of the line before and after RANSAC implementation.
You can use any built-in functions you wish'''
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
squareFeet=[100, 150, 185, 235, 310, 370, 420, 430, 440, 530, 600, 634, 718, 750, 850, 903, 978, 1010, 1050, 1990]
price=[12300, 18150, 20100, 23500, 31005, 359000, 44359, 52000, 53853, 61328, 68000, 72300, 77000, 89379, 93200, 97150, 102750, 115358, 119330, 323989]
# Add outlier data
np.random.seed(0)
X=np.array(squareFeet)
X=X.reshape(-1,1)
y=np.array(price)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=1)
# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(X, y)
# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X_train, y_train)
inlier_mask = ransac.inlier_mask_
for i in range(len(inlier_mask)):
    if inlier_mask[i]==False:
        print(f'outliers are x={squareFeet[i]} y={price[i]}')
outlier_mask = np.logical_not(inlier_mask)
# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)
