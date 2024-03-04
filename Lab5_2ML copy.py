'''Exercise 2
Plot the 3D graph of any 2 independent variables from the Multiple Linear Regression equa-
tion of Ex. 1'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d

df=pd.read_csv('california_housing.csv')
df=df.drop(df.columns[0],axis=1)
y=np.array(df.iloc[::20,8])
x=np.array(df.iloc[::20,:8])
reg=LinearRegression()
reg.fit(x,y)
c=np.array(reg.coef_)
yIntercept=reg.intercept_
print(x[:,0])
X1, X2 = np.meshgrid(x[:,0], x[:,1])
print(yIntercept)
Z = yIntercept + c[0]*X1 + c[1]*X2
print(Z)
#3D plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X1, X2, Z, color = 'blue')
#3D scattet plot (data points)
ax.scatter3D(x[:,0], x[:,1], y, c=y, cmap='Greens')
ax.set_title('3D Graph')
plt.show()