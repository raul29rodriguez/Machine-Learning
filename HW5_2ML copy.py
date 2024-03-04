'''Using the same data set and, having kept only the two most important features from Ex. 1,
perform Multiple Linear Regression using two most important features and create a 3D (mesh-
grid) plot between the two most important features and the response variable. In addition, on
the same graph, create a 3D scatter plot between the same independent and dependent variables.'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt

df=pd.read_csv('materials.csv')
#print(df)
y=np.array(df['Strength'])
x=np.array(df.loc[:,'Pressure':'Temperature'])

reg=LinearRegression()
reg.fit(x,y)
c=np.array(reg.coef_)
yIntercept=reg.intercept_
#print(x[:,0])
X1, X2 = np.meshgrid(x[:,0], x[:,1])
print(yIntercept)
Z = yIntercept + c[0]*X1 + c[1]*X2
#print(Z)
#3D plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X1, X2, Z, color = 'blue')
#3D scattet plot (data points)
ax.scatter3D(x[:,0], x[:,1], y, c=y, cmap='Greens')
ax.set_title('3D Graph')
plt.show()